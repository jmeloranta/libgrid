
/*
 * Crank-Nicolson propagation routines.
 *
 */

#include "grid.h"
#include "private.h"

/*
 * Auxiliary routine for calculating the energy (Crank-Nicolson).
 * Users should rather call grid_wf_energy().
 *
 * gwf       = Wavefunction for the energy calculation (wf *).
 * potential = Potential grid (cgrid *).
 * 
 * Returns the energy.
 *
 */

EXPORT REAL grid_wf_energy_cn(wf *gwf, rgrid *potential) {  

  REAL en;

  en = grid_wf_energy_cn_kinetic(gwf);
  if(potential) en += grid_wf_potential_energy(gwf, potential);
  return en;    
}

/*
 * Auxiliary routine for calculating the kinetic energy (Crank-Nicolson).
 * Users should rather call grid_wf_energy().
 *
 * gwf       = Wavefunction for the energy calculation (wf *).
 * 
 * Returns the energy.
 *
 */

EXPORT REAL grid_wf_energy_cn_kinetic(wf *gwf) {  

  cgrid *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");

  /* (-2m/hbar^2) T psi */
  cgrid_fd_laplace(gwf->grid, gwf->cworkspace);
  cgrid_multiply(gwf->cworkspace, -HBAR * HBAR / (2.0 * gwf->mass));

  /* int psi^* (T + V) psi d^3r */
  return CREAL(cgrid_integral_of_conjugate_product(gwf->grid, gwf->cworkspace));
}

/*
 * Main routine for propagating using Crank-Nicolson.
 *
 * gwf       = wavefunction to be propagated (wf *).
 * time      = time step function (REAL complex (*time)(INT, INT, INT, void *, REAL complex)). If NULL, tstep will be used.
 * tstep     = base time step length (REAL complex).
 * privdata  = additional private data form time step function (void *).
 * potential = potential grid (cgrid *; NULL if not needed).
 *
 * exp( -i (Tx + Ty + Tz) dt / hbar ) 
 *   = exp( -i (Tx+V) dt / hbar ) exp( -i (Ty+V) dt / hbar ) exp( -i (Tz+V) dt / hbar ) + O(dt^2)
 *   
 * Note: In dimensions higher than 1, the potential should not be included here because
 *       it would be propagate three times (once for each split component). It must be
 *       propagated separately.
 * 
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_cn(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *potential) {

  cgrid *grid = gwf->grid, *cworkspace;
  INT worklen = ((INT) sizeof(REAL complex)) * gwf->cworkspace->nx * gwf->cworkspace->ny * gwf->cworkspace->nz;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  if(gwf->grid->nx != 1) grid_wf_propagate_cn_x(gwf, time, tstep, privdata, potential, cworkspace->value, worklen);
  if(gwf->grid->ny != 1) grid_wf_propagate_cn_y(gwf, time, tstep, privdata, potential, cworkspace->value, worklen);
  if(gwf->grid->nz != 1) grid_wf_propagate_cn_z(gwf, time, tstep, privdata, potential, cworkspace->value, worklen);
}

/*
 * Auxiliary routine for propagating along x subject to given BC (Crank-Nicolson).
 *
 * gwf        = wavefunction to be propagated (wf *; input/output).
 * 
 * time       = time step function (REAL complex (*time)(INT, INT, INT, void *, REAL complex); input). If NULL, tstep will be used.
 * tstep      = base time step length (REAL complex; input).
 * privdata   = additional private data form time step function (void *; input).
 * potential  = potential grid (cgrid *; input). NULL if not needed.
 * workspace  = additional storage space needed (REAL complex *; overwritten) with size at least 3 * nx * (number of threads).
 * worklen    = workspace length (INT; input).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_cn_x(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *potential, REAL complex *workspace, INT worklen) {

  REAL complex c, cb, c2, c3, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, kx0 = gwf->grid->kx0, y, y0 = gwf->grid->y0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx;
  INT j, ny = gwf->grid->ny, ny2 = ny / 2;
  INT k, jk;
  INT nz = gwf->grid->nz, nyz = ny * nz;
  REAL complex *d, *b, *pwrk, *pot, tim, cp;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    exit(1);
  }
  if(worklen < 3 * nx * nyz * (INT) sizeof(REAL complex)) {
    fprintf(stderr, "libgrid(CUDA): CN worskspace too small.\n");
    exit(1);
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_x(gwf, time, tstep, privdata, potential, workspace, worklen)) return;
#endif

  if(worklen < ((INT) sizeof(REAL complex)) * grid_threads() * nx) {
    fprintf(stderr, "libgrid: grid_wf_propagate_cn_x workspace too small.\n");
    exit(1);
  }

  if(potential) pot = potential->value;
  else pot = NULL;

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t) <=> A x = b
   * (C + dx^2 laplace + CB V + C2 grad + C3 y grad) psi(t+dt) 
   *         = (C - dx^2 laplace - CB V - C2 grad - C3 y grad) psi(t)
   * where C = 4 i m dx^2 / (hbar dt), CB = -2m dx^2 / hbar^2, C2 = -i dx kx, C3 = m \omega i dx / hbar
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  cb = -2.0 * gwf->mass * step * step / (HBAR * HBAR);  // diag element coefficient for the potential
  c2 = -I * step * kx0; // coeff for moving background
  c3 = gwf->mass * gwf->grid->omega * I * step / HBAR; // coeff for rotating liquid around Z

#pragma omp parallel for firstprivate(cb,c2,c3,nx,ny,ny2,nz,nyz,y0,psi,workspace,c,step,pot,time,tstep,privdata,gwf) private(tid,i,j,k,jk,d,b,ind,tim,cp,pwrk,y) default(none) schedule(runtime)
  for (jk = 0; jk < nyz; jk++) {  /* for each (y,z) */
    j = jk / nz;
    k = jk % nz;  
    y = ((REAL) (j - ny2)) * step - y0;
    tid = omp_get_thread_num();
    d = &workspace[nx * (3 * tid + 0)];
    b = &workspace[nx * (3 * tid + 1)];
    pwrk = &workspace[nx * (3 * tid + 2)];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(i = 1; i < nx - 1; i++) {
      if(time) tim = (*time)(i, j, k, privdata, tstep);
      else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dx^2 laplace + CB V + C2 grad + C3 y grad) -- only C and laplace have diag elements */
      d[i] = cp - 2.0; // -2 from Laplacian, C = cp = c / dt
      if(pot) d[i] = d[i] + cb * pot[ind];
      /* Right-hand side (-) */
      /* (C - dx^2 laplace - CB V - C2 grad - C3 y grad) */
      b[i] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind - nyz]) 
             - c2 * (psi[ind + nyz] - psi[ind - nyz]) - c3 * y * (psi[ind + nyz] - psi[ind - nyz]);
      if(pot) b[i] = b[i] - cb * pot[ind] * psi[ind];
    }

    // Boundary conditions 
    ind = j * nz + k; // i = 0 - left boundary
    if(time) tim = (*time)(0, j, k, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[0] = b[0] - cb * pot[ind] * psi[ind];  // RHS(-)
      d[0] = d[0] + cb * pot[ind];             // LHS(+)
    }

    ind = (nx - 1) * nyz + j * nz + k;  // i = nx - 1, right boundary
    if(time) tim = (*time)(nx-1, j, k, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[nx-1] = b[nx-1] - cb * pot[ind] * psi[ind]; // RHS(-)
      d[nx-1] = d[nx-1] + cb * pot[ind];            // LHS(+)
    }

    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz, pwrk);
    else
      grid_solve_tridiagonal_system2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz);
  }
}

/*
 * Auxiliary routine for propagating along y subject to given BC (Crank-Nicolson).
 *
 * gwf       = wavefunction to be propagated (wf *).
 * 
 * time      = time step function (REAL complex (*time)(INT, INT, INT, void *, REAL complex)). If NULL, tstep will be used.
 * tstep     = base time step length (REAL complex).
 * privdata  = additional private data form time step function (void *).
 * potential = potential grid (cgrid *; NULL if not needed).
 * workspace = additional storage space needed (REAL complex *) with size at least 3 * ny * (number of threads).
 * worklen   = workspace length (INT).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_cn_y(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *potential, REAL complex *workspace, INT worklen) {

  REAL complex c, cb, c2, c3, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, ky0 = gwf->grid->ky0, x, x0 = gwf->grid->x0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx, nx2 = nx / 2;
  INT j, ny = gwf->grid->ny;
  INT k, ik;
  INT nz = gwf->grid->nz, nyz = ny * nz, nxz = nx * nz;
  REAL complex *d, *b, *pwrk, *pot, tim, cp;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    exit(1);
  }
  if(worklen < 3 * nx * nyz * (INT) sizeof(REAL complex)) {
    fprintf(stderr, "libgrid(CUDA): CN worskspace too small.\n");
    exit(1);
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_y(gwf, time, tstep, privdata, potential, workspace, worklen)) return;
#endif

  if(worklen < ((INT) sizeof(REAL complex)) * grid_threads() * ny) {
    fprintf(stderr, "libgrid: grid_wf_propagate_cn_y workspace too small.\n");
    exit(1);
  }

  if(potential) pot = potential->value;
  else pot = NULL;

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t) <=> A x = b
   * (C + dy^2 laplace + CB V + C2 grad + C3 x grad) psi(t+dt) 
   *         = (C - dx^2 laplace - CB V - C2 grad - C3 x grad) psi(t)
   * where C = 4 i m dy^2 / (hbar dt), CB = -2m dy^2 / hbar^2, C2 = -i dy ky, C3 = -m \omega i dy / hbar
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  cb = -2.0 * gwf->mass * step * step / (HBAR * HBAR);  // diag element coefficient for the potential
  c2 = -I * step * ky0;
  c3 = -gwf->mass * gwf->grid->omega * I * step / HBAR;
  
#pragma omp parallel for firstprivate(cb,c2,c3,nx,nx2,ny,nz,nyz,nxz,x0,psi,workspace,c,step,pot,time,tstep,privdata,gwf) private(tid,i,j,k,ik,d,b,ind,tim,cp,x,pwrk) default(none) schedule(runtime)
  for (ik = 0; ik < nxz; ik++) {  /* for each (x,z) */
    i = ik / nz;
    k = ik % nz;
    x = ((REAL) (i - nx2)) * step - x0;    
    tid = omp_get_thread_num();
    d = &workspace[ny * (3 * tid + 0)];
    b = &workspace[ny * (3 * tid + 1)];
    pwrk = &workspace[ny * (3 * tid + 2)];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(j = 1; j < ny - 1; j++) {
      if(time) tim = (*time)(i, j, k, privdata, tstep);
      else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dy^2 laplace + CB V + C2 grad + C3 x grad) */
      d[j] = cp - 2.0; // -2 from Laplacian, cp = c / dt, LHS(+)
      if(pot) d[j] = d[j] + cb * pot[ind];
      /* Right-hand side (-) */
      /* (C - dx^2 laplace - CB V - C2 grad - C3 x grad) */
      b[j] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind - nz]) 
             - c2 * (psi[ind + nz] - psi[ind - nz]) - c3 * x * (psi[ind + nz] - psi[ind - nz]);
      if(pot) b[j] = b[j] - cb * pot[ind] * psi[ind];
    }

    // Boundary conditions
 
    ind = i * nyz + k; // j = 0 - left boundary
    if(time) tim = (*time)(i, 0, k, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[0] = b[0] - cb * pot[ind] * psi[ind];  // RHS(-)
      d[0] = d[0] + cb * pot[ind];             // LHS(+)
    }

    ind = i * nyz + (ny-1) * nz + k;  // j = ny - 1 - right boundary
    if(time) tim = (*time)(i, ny-1, k, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[ny-1] = b[ny-1] - cb * pot[ind] * psi[ind]; // RHS(-)
      d[ny-1] = d[ny-1] + cb * pot[ind];            // LHS(+)
    }

    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz, pwrk);
    else
      grid_solve_tridiagonal_system2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz);
  }
}

/*
 * Auxiliary routine for propagating along z subject to given BC (Crank-Nicolson).
 *
 * gwf       = wavefunction to be propagated (wf *).
 * 
 * time      = time step function (REAL complex (*time)(INT, INT, INT, void *, REAL complex)). If NULL, tstep will be used.
 * tstep     = base time step length (REAL complex).
 * privdata  = additional private data form time step function (void *).
 * potential = potential grid (cgrid *; NULL if not needed).
 * workspace = additional storage space needed (REAL complex *) with size at least 3 * nz * (number of threads).
 * worklen   = workspace length (INT).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_cn_z(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *potential, REAL complex *workspace, INT worklen) {

  REAL complex c, cb, c2, *psi = gwf->grid->value;
  REAL step = gwf->grid->step, kz0 = gwf->grid->kz0;
  INT tid, ind;
  INT i, nx = gwf->grid->nx;
  INT j, ny = gwf->grid->ny;
  INT k, ij;
  INT nz = gwf->grid->nz, nyz = ny * nz, nxy = nx * ny;
  REAL complex *d, *b, *pwrk, *pot, tim, cp;

#ifdef USE_CUDA
  if(time && time != &grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    exit(1);
  }
  if(worklen < 3 * nx * nyz * (INT) sizeof(REAL complex)) {
    fprintf(stderr, "libgrid(CUDA): CN worskspace too small.\n");
    exit(1);
  }
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cn_z(gwf, time, tstep, privdata, potential, workspace, worklen)) return;
#endif

  if(worklen < ((INT) sizeof(REAL complex)) * grid_threads() * nz) {
    fprintf(stderr, "libgrid: grid_wf_propagate_cn_z workspace too small.\n");
    exit(1);
  }

  if(potential) pot = potential->value;
  else pot = NULL;

  /*
   * (1 + .5 i (T + V - v0 pz) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - v0 pz) dt / hbar) psi(t) <=> A x = b
   * (C + dz^2 laplace + CB V + C2 grad) psi(t+dt) 
   *         = (C - dz^2 laplace - CB V - C2 grad) psi(t)
   * where C = 4 i m dz^2 / (hbar dt), CB = -2m dz^2 / hbar^2, C2 = -i dz kz.
   *
   */
  c = 4.0 * I * gwf->mass * step * step / HBAR; // division by dt included separately below
  cb = -2.0 * gwf->mass * step * step / (HBAR * HBAR);  // diag element coefficient for the potential
  c2 = -I * step * kz0;

#pragma omp parallel for firstprivate(cb,c2,nx,ny,nz,nyz,nxy,psi,workspace,c,step,pot,time,tstep,privdata,gwf) private(tid,i,j,k,ij,d,b,ind,tim,cp,pwrk) default(none) schedule(runtime)
  for (ij = 0; ij < nxy; ij++) {  /* for each (x,y) */
    i = ij / ny;
    j = ij % ny;
    tid = omp_get_thread_num();
    d = &workspace[nz * (3 * tid + 0)];
    b = &workspace[nz * (3 * tid + 1)];
    pwrk = &workspace[nz * (3 * tid + 2)];

    /* create left-hand diagonal element (d) and right-hand vector (b) */
    for(k = 1; k < nz - 1; k++) {
      if(time) tim = (*time)(i, j, k, privdata, tstep);
      else tim = tstep;
      cp = c / tim;
      ind = i * nyz + j * nz + k;
      /* Left-hand side (+) */
      /* (C + dz^2 laplace + CB V + C2 grad) */
      d[k] = cp - 2.0; // Diagonal: -2 from Laplacian, cp = c / dt
      if(pot) d[k] = d[k] + cb * pot[ind]; // add possible potential to diagonal
      /* Right-hand side (-) */
      /* (C - dz^2 laplace - CB V - C2 grad) */
      b[k] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind - 1]) 
             - c2 * (psi[ind + 1] - psi[ind - 1]); // NOTE: No rotation term as the rotation is about the z-axis
      if(pot) b[k] = b[k] - cb * pot[ind] * psi[ind];
    }

    // Boundary conditions
 
    ind = i * nyz + j * nz; // k = 0 - left boundary
    if(time) tim = (*time)(i, j, 0, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[0] = b[0] - cb * pot[ind] * psi[ind];  // RHS(-)
      d[0] = d[0] + cb * pot[ind];             // LHS(+)
    }

    ind = i * nyz + j * nz + (nz - 1);  // k = nz-1 - right boundary
    if(time) tim = (*time)(i, j, nz-1, privdata, tstep);
    else tim = tstep;
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
    if(pot) {
      b[nz-1] = b[nz-1] - cb * pot[ind] * psi[ind]; // RHS(-)
      d[nz-1] = d[nz-1] + cb * pot[ind];            // LHS(+)
    }

    if(gwf->boundary == WF_PERIODIC_BOUNDARY)
      grid_solve_tridiagonal_system_cyclic2(nz, d, b, &psi[i * nyz + j * nz], c2, 1, pwrk);
    else
      grid_solve_tridiagonal_system2(nz, d, b, &psi[i * nyz + j * nz], c2, 1);
  }
}
