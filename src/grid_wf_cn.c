
/*
 * Crank-Nicolson propagation routines.
 *
 */

#include "grid.h"

/*
 * @FUNC{grid_wf_energy_cn, "Wavefunction energy (finite difference/CN)"}
 * @DESC{"Auxiliary routine for calculating the energy (finite difference/Crank-Nicolson).
          Users should rather call grid_wf_energy()"}
 * @ARG1{wf *gwf, "Wavefunction for the energy calculation"}
 * @ARG2{cgrid *potential, "Potential grid"}
 * @RVAL{REAL, "Returns the energy"}
 *
 */

EXPORT REAL grid_wf_energy_cn(wf *gwf, rgrid *potential) {  

  REAL en;

  en = grid_wf_kinetic_energy_cn(gwf);
  if(potential) en += grid_wf_potential_energy(gwf, potential);
  return en;    
}

/*
 * @FUNC{grid_wf_kinetic_energy_cn, "Wavefunction kinetic energy (finite difference)"}
 * @DESC{"Auxiliary routine for calculating the kinetic energy (finite difference/Crank-Nicolson).
          Users should rather call grid_wf_energy()"}
 * @ARG1{wf *gwf, "Wavefunction for the energy calculation"}
 * @RVAL{REAL, "Returns the energy"}
 *
 */

EXPORT REAL grid_wf_kinetic_energy_cn(wf *gwf) {  

  cgrid *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");

  /* (-2m/hbar^2) T psi */
  cgrid_fd_laplace(gwf->grid, gwf->cworkspace);
  cgrid_multiply(gwf->cworkspace, -HBAR * HBAR / (2.0 * gwf->mass));

  /* int psi^* (T + V) psi d^3r */
  return CREAL(cgrid_integral_of_conjugate_product(gwf->grid, gwf->cworkspace));
}

/*
 * @FUNC{grid_wf_propagate_kinetic_cn, "Propagate wavefunction (Crank-Nicolson)"}
 * @DESC{"This is the main function for propagating given wavefunction using Crank-Nicolson.
          Note that in dimensions higher than 1, the potential should not be included.
          It must be propagated separately"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex tstep, "Base time step length"}
 * @RVAL{void, "No return value"}
 *
 * exp( -i (Tx + Ty + Tz) dt / hbar ) 
 *   = exp( -i (Tx+V) dt / hbar ) exp( -i (Ty+V) dt / hbar ) exp( -i (Tz+V) dt / hbar ) + O(dt^2)
 * 
 */

EXPORT void grid_wf_propagate_kinetic_cn(wf *gwf, REAL complex tstep) {

  cgrid *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
  // Workspace 3 only needed for periodic CN
  if(gwf->boundary == WF_PERIODIC_BOUNDARY) {
    if(!gwf->cworkspace3) gwf->cworkspace3 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 3");
  }

  if(gwf->grid->nx != 1) grid_wf_propagate_cn_x(gwf, tstep, gwf->cworkspace, gwf->cworkspace2, gwf->cworkspace3);
  if(gwf->grid->ny != 1) grid_wf_propagate_cn_y(gwf, tstep, gwf->cworkspace, gwf->cworkspace2, gwf->cworkspace3);
  if(gwf->grid->nz != 1) grid_wf_propagate_cn_z(gwf, tstep, gwf->cworkspace, gwf->cworkspace2, gwf->cworkspace3);
}

/*
 * @FUNC{grid_wf_propagate_cn_x, "Propagate wavefunction x (Crank-Nicolson)"}
 * @DESC{"Auxiliary routine for propagating along x subject to given BC (Crank-Nicolson)"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex tstep, "Base time step length"}
 * @ARG3{cgrid *workspace, "Additional storage space"}
 * @ARG4{cgrid *workspace2, "Additional storage space"}
 * @ARG5{cgrid *workspace3, "Additional storage space needed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_cn_x(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  REAL complex c, c2, c3, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, kx0 = gwf->grid->kx0, y, y0 = gwf->grid->y0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx;
  INT j, ny = gwf->grid->ny, ny2 = ny / 2;
  INT k, jk;
  INT nz = gwf->grid->nz, nyz = ny * nz;
  REAL complex *d, *b, *pwrk, tim, cp, *wrk, *wrk2, *wrk3;
  REAL (*time)(INT, INT, INT, INT, INT, INT, INT, INT, INT) = gwf->ts_func, tmp;
  INT lx = gwf->lx, hx = gwf->hx, ly = gwf->ly, hy = gwf->hy, lz = gwf->lz, hz = gwf->hz;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    abort();
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_x(gwf, tstep, workspace, workspace2, workspace3)) return;
#endif

  /*
   * (1 + .5 i (T - \omega Lz - v0 px) dt / hbar) psi(t+dt) = (1 - .5 i (T - \omega Lz - v0 px) dt / hbar) psi(t) <=> A x = b
   * (C + dx^2 laplace + C2 grad + C3 y grad) psi(t+dt) 
   *         = (C - dx^2 laplace - C2 grad - C3 y grad) psi(t)
   * where C = 4 i m dx^2 / (hbar dt), C2 = -i dx kx, C3 = m \omega i dx / hbar
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  c2 = -I * step * kx0; // coeff for moving background; TODO: do we need 2 X for C2????
  c3 = gwf->mass * gwf->grid->omega * I * step / HBAR; // coeff for rotating liquid around Z
  wrk = workspace->value;
  wrk2 = workspace2->value;
  wrk3 = workspace3->value;

#pragma omp parallel for firstprivate(c2,c3,nx,ny,ny2,nz,nyz,y0,psi,wrk,wrk2,wrk3,c,step,time,tstep,gwf,lx,hx,ly,hy,lz,hz) private(tmp,tid,i,j,k,jk,d,b,ind,tim,cp,pwrk,y) default(none) schedule(runtime)
  for (jk = 0; jk < nyz; jk++) {  /* for each (y,z) */
    j = jk / nz;
    k = jk % nz;  
    y = ((REAL) (j - ny2)) * step - y0;
    tid = omp_get_thread_num();
    d = &wrk[nx * tid];
    b = &wrk2[nx * tid];
    pwrk = &wrk3[nx * tid];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(i = 1; i < nx - 1; i++) {
      if(time) {
        tmp = (*time)(i, j, k, lx, hx, ly, hy, lz, hz);
        tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
      } else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dx^2 laplace + C2 grad + C3 y grad) -- only C and laplace have diag elements */
      d[i] = cp - 2.0; // -2 from Laplacian, C = cp = c / dt
      /* Right-hand side (-) */
      /* (C - dx^2 laplace - CB V - C2 grad - C3 y grad) */
      b[i] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind - nyz]) 
             - c2 * (psi[ind + nyz] - psi[ind - nyz]) - c3 * y * (psi[ind + nyz] - psi[ind - nyz]);
    }

    // Boundary conditions 
    ind = j * nz + k; // i = 0 - left boundary
    if(time) {
      tmp = (*time)(0, j, k, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    /* Right-hand side (-) */
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind - nyz] = 0
        b[0] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind]) - c2 * psi[ind + nyz] - c3 * y * psi[ind + nyz];
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind - nyz] = psi[ind + nyz]
        b[0] = cp * psi[ind] - (2.0 * psi[ind + nyz] - 2.0 * psi[ind]);
      break;
      case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind - nyz] = psi[ind + (nx-1)*nyz]
        b[0] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind + (nx-1)*nyz])
               - c2 * (psi[ind + nyz] - psi[ind + (nx-1) * nyz]) - c3 * y * (psi[ind + nyz] - psi[ind + (nx-1) * nyz]);
      break;
    }
    d[0] = cp - 2.0;  // LHS: -2 diag elem from Laplacian, cp = c / dt

    ind = (nx - 1) * nyz + j * nz + k;  // i = nx - 1, right boundary
    if(time) {
      tmp = (*time)(nx-1, j, k, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    /* Right-hand side (-) */
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind + nyz] = 0
        b[nx-1] = cp * psi[ind] - (psi[ind - nyz] - 2.0 * psi[ind]) + c2 * psi[ind - nyz] + c3 * y * psi[ind - nyz];  // c2,c3: - from difference
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind + nyz] = psi[ind - nyz]
        b[nx-1] = cp * psi[ind] - (2.0 * psi[ind - nyz] - 2.0 * psi[ind]);
      break;
      case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind + nyz] = psi[ind - (nx-1)*nyz]
        b[nx-1] = cp * psi[ind] - (psi[ind - nyz] - 2.0 * psi[ind] + psi[ind - (nx-1) * nyz])
               - c2 * (psi[ind - (nx-1) * nyz] - psi[ind - nyz]) - c3 * y * (psi[ind - (nx-1) * nyz] - psi[ind - nyz]);
      break;
    }      
    d[nx-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt

// TODO: There are other than periodic and neumann.
    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz, pwrk);
    else
      grid_solve_tridiagonal_system2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz);
  }
}

/*
 * @FUNC{grid_wf_propagate_cn_y, "Propagate wavefunction y (Crank-Nicolson)"}
 * @DESC{"Auxiliary routine for propagating along y subject to given BC (Crank-Nicolson)"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex tstep, "Base time step length"}
 * @ARG3{cgrid *workspace, "Additional storage space"}
 * @ARG4{cgrid *workspace2, "Additional storage space"}
 * @ARG5{cgrid *workspace3, "Additional storage space needed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_cn_y(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  REAL complex c, c2, c3, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, ky0 = gwf->grid->ky0, x, x0 = gwf->grid->x0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx, nx2 = nx / 2;
  INT j, ny = gwf->grid->ny;
  INT k, ik;
  INT nz = gwf->grid->nz, nyz = ny * nz, nxz = nx * nz;
  REAL complex *d, *b, *pwrk, tim, cp, *wrk, *wrk2, *wrk3;
  REAL (*time)(INT, INT, INT, INT, INT, INT, INT, INT, INT) = gwf->ts_func, tmp;
  INT lx = gwf->lx, hx = gwf->hx, ly = gwf->ly, hy = gwf->hy, lz = gwf->lz, hz = gwf->hz;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    abort();
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_y(gwf, tstep, workspace, workspace2, workspace3)) return;
#endif

  /*
   * (1 + .5 i (T - \omega Lz - v0 py) dt / hbar) psi(t+dt) = (1 - .5 i (T - \omega Lz - v0 py) dt / hbar) psi(t) <=> A x = b
   * (C + dy^2 laplace + C2 grad + C3 x grad) psi(t+dt) 
   *         = (C - dx^2 laplace - C2 grad - C3 x grad) psi(t)
   * where C = 4 i m dy^2 / (hbar dt), C2 = -i dy ky, C3 = -m \omega i dy / hbar
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  c2 = -I * step * ky0; // TODO: Do we need 2X for C2?
  c3 = -gwf->mass * gwf->grid->omega * I * step / HBAR;
  wrk = workspace->value;
  wrk2 = workspace2->value;
  wrk3 = workspace3->value;
  
#pragma omp parallel for firstprivate(c2,c3,nx,nx2,ny,nz,nyz,nxz,x0,psi,wrk,wrk2,wrk3,c,step,time,tstep,gwf,lx,hx,ly,hy,lz,hz) private(tmp,tid,i,j,k,ik,d,b,ind,tim,cp,x,pwrk) default(none) schedule(runtime)
  for (ik = 0; ik < nxz; ik++) {  /* for each (x,z) */
    i = ik / nz;
    k = ik % nz;
    x = ((REAL) (i - nx2)) * step - x0;    
    tid = omp_get_thread_num();
    d = &wrk[ny * tid];
    b = &wrk2[ny * tid];
    pwrk = &wrk3[ny * tid];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(j = 1; j < ny - 1; j++) {
      if(time) {
        tmp = (*time)(i, j, k, lx, hx, ly, hy, lz, hz);
        tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
      } else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dy^2 laplace + C2 grad + C3 x grad) */
      d[j] = cp - 2.0; // -2 from Laplacian, cp = c / dt, LHS(+)
      /* Right-hand side (-) */
      /* (C - dx^2 laplace - C2 grad - C3 x grad) */
      b[j] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind - nz]) 
             - c2 * (psi[ind + nz] - psi[ind - nz]) - c3 * x * (psi[ind + nz] - psi[ind - nz]);
    }

    // Boundary conditions
 
    ind = i * nyz + k; // j = 0 - left boundary
    if(time) {
      tmp = (*time)(i, 0, k, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    /* Right-hand side (-) */
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind - nz] = 0
        b[0] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind]) - c2 * psi[ind + nz] - c3 * x * psi[ind + nz];
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind - nz] = psi[ind + nz]
        b[0] = cp * psi[ind] - (2.0 * psi[ind + nz] - 2.0 * psi[ind]);
      break;
     case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind - nz] = psi[ind + (ny-1)*nz]
        b[0] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind + (ny-1)*nz])
               - c2 * (psi[ind + nz] - psi[ind + (ny-1) * nz]) - c3 * x * (psi[ind + nz] - psi[ind + (ny-1) * nz]);
     break;
    }
    d[0] = cp - 2.0;  // -2 from Laplacian, cp = c / dt

    ind = i * nyz + (ny-1) * nz + k;  // j = ny - 1 - right boundary
    if(time) {
      tmp = (*time)(i, ny-1, k, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    /* Right-hand side (-) */
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind + nz] = 0
        b[ny-1] = cp * psi[ind] - (psi[ind - nz] - 2.0 * psi[ind]) + c2 * psi[ind - nz] + c3 * x * psi[ind - nz];
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind + nz] = psi[ind - nz]
        b[ny-1] = cp * psi[ind] - (2.0 * psi[ind - nz] - 2.0 * psi[ind]);
      break;
      case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind + nz] = psi[ind - (ny-1)*nz]
        b[ny-1] = cp * psi[ind] - (psi[ind - nz] - 2.0 * psi[ind] + psi[ind - (ny-1)*nz])
               - c2 * (psi[ind - (ny-1) * nz] - psi[ind - nz]) - c3 * x * (psi[ind - (ny-1) * nz] - psi[ind - nz]);
      break;
    }
    d[ny-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt

    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz, pwrk);
    else
      grid_solve_tridiagonal_system2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz);
  }
}

/*
 * @FUNC{grid_wf_propagate_cn_z, "Propagate wavefunction z (Crank-Nicolson)"}
 * @DESC{"Auxiliary routine for propagating along z subject to given BC (Crank-Nicolson)"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex tstep, "Base time step length"}
 * @ARG3{cgrid *workspace, "Additional storage space"}
 * @ARG4{cgrid *workspace2, "Additional storage space"}
 * @ARG5{cgrid *workspace3, "Additional storage space needed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_cn_z(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  REAL complex c, c2, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, kz0 = gwf->grid->kz0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx;
  INT j, ny = gwf->grid->ny;
  INT k, ij;
  INT nz = gwf->grid->nz, nyz = ny * nz, nxy = nx * ny;
  REAL complex *d, *b, *pwrk, tim, cp, *wrk, *wrk2, *wrk3;
  REAL (*time)(INT, INT, INT, INT, INT, INT, INT, INT, INT) = gwf->ts_func, tmp;
  INT lx = gwf->lx, hx = gwf->hx, ly = gwf->ly, hy = gwf->hy, lz = gwf->lz, hz = gwf->hz;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    abort();
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_z(gwf, tstep, workspace, workspace2, workspace3)) return;
#endif

  /*
   * (1 + .5 i (T - v0 pz) dt / hbar) psi(t+dt) = (1 - .5 i (T - v0 pz) dt / hbar) psi(t) <=> A x = b
   * (C + dz^2 laplace + C2 grad) psi(t+dt) 
   *         = (C - dz^2 laplace - C2 grad) psi(t)
   * where C = 4 i m dz^2 / (hbar dt), C2 = -i dz kz.
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  c2 = -I * step * kz0;  // TODO: Do we need 2X for C2?
  wrk = workspace->value;
  wrk2 = workspace2->value;
  wrk3 = workspace3->value;

#pragma omp parallel for firstprivate(c2,nx,ny,nz,nyz,nxy,psi,wrk,wrk2,wrk3,c,step,time,tstep,gwf,lx,hx,ly,hy,lz,hz) private(tmp,tid,i,j,k,ij,d,b,ind,tim,cp,pwrk) default(none) schedule(runtime)
  for (ij = 0; ij < nxy; ij++) {  /* for each (x,y) */
    i = ij / ny;
    j = ij % ny;
    tid = omp_get_thread_num();
    d = &wrk[nz * tid];
    b = &wrk2[nz * tid];
    pwrk = &wrk3[nz * tid];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(k = 1; k < nz - 1; k++) {
      if(time) {
        tmp = (*time)(i, j, k, lx, hx, ly, hy, lz, hz);
        tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
      } else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dz^2 laplace + C2 grad) */
      d[k] = cp - 2.0; // Diagonal: -2 from Laplacian, cp = c / dt
      /* Right-hand side (-) */
      /* (C - dz^2 laplace - C2 grad) */
      b[k] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind - 1]) 
             - c2 * (psi[ind + 1] - psi[ind - 1]); // NOTE: No rotation term as the rotation is about the z-axis
    }

    // Boundary conditions
 
    ind = i * nyz + j * nz; // k = 0 - left boundary
    if(time) {
      tmp = (*time)(i, j, 0, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    /* Right-hand side (-) */
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind - 1] = 0
        b[0] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind]) - c2 * psi[ind + 1];
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind - 1] = psi[ind + 1]
        b[0] = cp * psi[ind] - (2.0 * psi[ind + 1] - 2.0 * psi[ind]);
      break;
     case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind - 1] = psi[ind + (nz-1)]
        b[0] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind + (nz-1)])
               - c2 * (psi[ind + 1] - psi[ind + (nz-1)]);
     break;
    }
    d[0] = cp - 2.0;  // -2 from Laplacian, cp = c / dt

    ind = i * nyz + j * nz + (nz - 1);  // k = nz-1 - right boundary
    if(time) {
      tmp = (*time)(i, j, nz-1, lx, hx, ly, hy, lz, hz);
      tim = (1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp;
    } else tim = tstep;
    cp = c / tim;
    switch(gwf->boundary) {
      case WF_DIRICHLET_BOUNDARY:
        // Dirichlet: psi[ind + 1] = 0
        b[nz-1] = cp * psi[ind] - (psi[ind - 1] - 2.0 * psi[ind]) + c2 * psi[ind - 1];
      break;
      case WF_NEUMANN_BOUNDARY:
        // Neumann: psi[ind + 1] = psi[ind - 1]
        b[nz-1] = cp * psi[ind] - (2.0 * psi[ind - 1] - 2.0 * psi[ind]);
      break;
      case WF_PERIODIC_BOUNDARY:
        // Periodic: psi[ind + 1] = psi[ind - (nz-1)]
        b[nz-1] = cp * psi[ind] - (psi[ind - 1] - 2.0 * psi[ind] + psi[ind - (nz-1)])
               - c2 * (psi[ind - (nz-1)] - psi[ind - 1]);
      break;
    }      
    d[nz-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt

    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(nz, d, b, &psi[i * nyz + j * nz], c2, 1, pwrk);
    else
      grid_solve_tridiagonal_system2(nz, d, b, &psi[i * nyz + j * nz], c2, 1); // c2 has very little effect
  }
}
