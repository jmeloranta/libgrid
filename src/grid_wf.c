/*
 * Routines for handling wavefunctions.
 *
 */

#include "grid.h"
#include "private.h"

/*
 * Allocate wavefunction.
 *
 * nx         = number of spatial grid points along x (INT).
 * ny         = number of spatial grid points along y (INT).
 * nz         = number of spatial grid points along z (INT).
 * step       = spatial step size (REAL).
 * mass       = mass of the particle corresponding to this wavefunction (REAL).
 * boundary   = boundary condition (int):
 *              WF_DIRICHLET_BOUNDARY = Dirichlet boundary condition.
 *              WF_NEUMANN_BOUNDARY   = Neumann boundary condition.
 *              WF_PERIODIC_BOUNDARY  = Periodic boundary condition.
 * propagator = which time propagator to use for this wavefunction:
 *              WF_2ND_ORDER_PROPAGATOR = 2nd order in time.
 *              WF_4TH_ORDER_PROPAGATOR = 4th order in time.
 * id         = String identifier for the grid (for debugging; char *; input).
 *
 * Return value is a pointer to the allocated wavefunction.
 * This routine returns NULL if allocation fails.
 *
 */

EXPORT wf *grid_wf_alloc(INT nx, INT ny, INT nz, REAL step, REAL mass, char boundary, char propagator, char *id) {

  wf *gwf;
  REAL complex (*value_outside)(struct cgrid_struct *grid, INT i, INT j, INT k);
  
  if(boundary != WF_DIRICHLET_BOUNDARY 
     && boundary != WF_NEUMANN_BOUNDARY 
     && boundary != WF_PERIODIC_BOUNDARY
     && boundary != WF_VORTEX_X_BOUNDARY
     && boundary != WF_VORTEX_Y_BOUNDARY
     && boundary != WF_VORTEX_Z_BOUNDARY) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Unknown boundary condition.\n");
    return 0;
  }
  
  if(propagator != WF_2ND_ORDER_PROPAGATOR 
       && propagator != WF_4TH_ORDER_PROPAGATOR) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Unknown propagator.\n");
    return 0;
  }
  
  if ((boundary == WF_DIRICHLET_BOUNDARY || boundary == WF_NEUMANN_BOUNDARY)
      && propagator == WF_4TH_ORDER_PROPAGATOR) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Invalid boundary condition - propagator combination. 4th order propagator can be used only with periodic boundary conditions.\n");
    return 0;
  }
  
  gwf = (wf *) malloc(sizeof(wf));
  if (!gwf) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Could not allocate memory for wf.\n");
    return 0;
  }
  
  value_outside = NULL;
  if (boundary == WF_DIRICHLET_BOUNDARY)
    value_outside = cgrid_value_outside_constantdirichlet;
  else if (boundary == WF_NEUMANN_BOUNDARY)
    value_outside = cgrid_value_outside_neumann;
  else if (boundary == WF_PERIODIC_BOUNDARY)
    value_outside = cgrid_value_outside;
  else if (boundary == WF_VORTEX_X_BOUNDARY)
    value_outside = cgrid_value_outside_vortex_x;
  else if (boundary == WF_VORTEX_Y_BOUNDARY)
    value_outside = cgrid_value_outside_vortex_y;
  else if (boundary == WF_VORTEX_Z_BOUNDARY)
    value_outside = cgrid_value_outside_vortex_z;
  
  gwf->grid = cgrid_alloc(nx, ny, nz, step, value_outside, 0, id);
  
  if (!gwf->grid) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Could not allocate memory for wf->grid.\n");
    free(gwf);
    return 0;
  }
  
  gwf->mass = mass;
  gwf->norm = 1.0;
  gwf->boundary = boundary;
  gwf->propagator = propagator;
  
  return gwf;
}

/*
 * Free wavefunction.
 *
 * gwf = wavefunction to be freed (wf *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_free(wf *gwf) {

  if (gwf) {
    if (gwf->grid) cgrid_free(gwf->grid);
    free(gwf);
  }
}

/* 
 * Calculate (complex) time for implementing absorbing boundaries.
 *
 * Excitations entering the absorbing region will be damped out
 * such that no back reflections occur from the boundary of the finite grid.
 * This is achieved by introducing imaginary time component to propagation
 * in this reagion, which is turned to full gradually (using a linear function).
 * This is to be used with time propagation routines that allow spatially dependent 
 * time (-> excludes kinetic energy by FFT). The absorging region is placed outside 
 * [lx,hx] x [ly,hy] x [lz,hz]. At the corners of the box, the propagation time
 * is (1.0 - I) * tstep (whereas outside the absorbing region the imaginary
 * component is zero). Each x,y,z direction contributes 1/3 to the imaginary component.
 *
 * i           = Current index along X (INT; input).
 * j           = Current index along Y (INT; input).
 * k           = Current index along Z (INT; input).
 * data        = Pointer to struct grid_abs holding values for specifying the absorbing region (void *; INPUT).
 *               This will specify amp, lx, hx, ly, hy, lz, hz.
 * time_step   = Real time step (REAL complex; input). This should be real valued.
 * 
 * Returns the (complex) time.
 *
 */

EXPORT REAL complex grid_wf_absorb(INT i, INT j, INT k, void *data, REAL complex time_step) {

  REAL t;
  struct grid_abs *ab = (struct grid_abs *) data;
  INT lx = ab->data[0], hx = ab->data[1], ly = ab->data[2], hy = ab->data[3], lz = ab->data[4], hz = ab->data[5];
  REAL amp = ab->amp;

  if(CIMAG(time_step) != 0.0) {
    fprintf(stderr, "libgrid: Imaginary time for absorbing boundary - forcing real time.\n");
    time_step = CREAL(time_step);
  }
  if(i >= lx && i <= hx && j >= ly && j <= hy && k >= lz && k <= hz) return (REAL complex) time_step;

  t = 0.0;

  if(i < lx) t -= ((REAL) (lx - i)) / (3.0 * (REAL) lx);
  else if(i > hx) t -= ((REAL) (i - hx)) / (3.0 * (REAL) lx);

  if(j < ly) t -= ((REAL) (ly - j)) / (3.0 * (REAL) ly);
  else if(j > hy) t -= ((REAL) (j - hy)) / (3.0 * (REAL) ly);

  if(k < lz) t -= ((REAL) (lz - k)) / (3.0 * (REAL) lz);
  else if(k > hz) t -= ((REAL) (k - hz)) / (3.0 * (REAL) lz);

  return (1.0 + I * amp * t) * (REAL complex) time_step;
}

/*
 * Calculate the velocity field.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * vy     = y output grid containing the velocity (rgrid *).
 * vz     = z output grid containing the velocity (rgrid *).
 * cutoff = cutoff value for velocity (|v| <= cutoff) (REAL).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz, REAL cutoff) {
  
  grid_wf_velocity_x(gwf, vx, cutoff);
  grid_wf_velocity_y(gwf, vy, cutoff);
  grid_wf_velocity_z(gwf, vz, cutoff);
}

/*
 * Calculate the velocity field x component using:
 * v = -\frac{i\hbar}{2m} (d/dx) ln(\psi/\psi^*).
 *
 */

EXPORT void grid_wf_velocity_x(wf *gwf, rgrid *vx, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_x(gwf, vx, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vx,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i+1, j, k);
      pm = cgrid_value_at_index(grid, i-1, j, k);
      tmp = inv_delta * CIMAG(CLOG(pp * CONJ(pm) / (CONJ(pp) * pm)));
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vx, i, j, k, tmp);      
    }
  }
}

/*
 * Calculate the velocity field y component using:
 * v = -\frac{i\hbar}{2m} (d/dy) ln(\psi/\psi^*).
 *
 */

EXPORT void grid_wf_velocity_y(wf *gwf, rgrid *vy, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_y(gwf, vy, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vy,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j+1, k);
      pm = cgrid_value_at_index(grid, i, j-1, k);
      tmp = inv_delta * CIMAG(CLOG(pp * CONJ(pm) / (CONJ(pp) * pm)));
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vy, i, j, k, tmp);
    }
  }
}

/*
 * Calculate the velocity field z component using:
 * v = -\frac{i\hbar}{2m} (d/dz) ln(\psi/\psi^*).
 *
 */

EXPORT void grid_wf_velocity_z(wf *gwf, rgrid *vz, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_z(gwf, vz, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vz,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j, k+1);
      pm = cgrid_value_at_index(grid, i, j, k-1);
      tmp = inv_delta * CIMAG(CLOG(pp * CONJ(pm) / (CONJ(pp) * pm)));
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vz, i, j, k, tmp);
    }
  }
}

/*
 * Calculate the probability flux x component.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_x    = x output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_x(wf *gwf, rgrid *flux_x) {

  cgrid *grid = gwf->grid;
  INT i, j, ij, k, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_x(gwf, flux_x, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_x,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_x, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k))
        * (cgrid_value_at_index(grid, i+1, j, k) - cgrid_value_at_index(grid, i-1, j, k))) * inv_delta);
  }
}

/*
 * Calculate the probability flux y component.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_y    = y output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_y(wf *gwf, rgrid *flux_y) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_y(gwf, flux_y, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_y,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_y, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k))
        * (cgrid_value_at_index(grid, i, j+1, k) - cgrid_value_at_index(grid, i, j-1, k))) * inv_delta);
  }
}

/*
 * Calculate the probability flux z component.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_z    = z output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_z(wf *gwf, rgrid *flux_z) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_z(gwf, flux_z, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_z,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_z, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) 
        * (cgrid_value_at_index(grid, i, j, k+1) - cgrid_value_at_index(grid, i, j, k-1))) * inv_delta);
  }
}

/*
 * Calculate the probability flux.
 *
 * gwf        = wavefunction for the operation (wf *).
 * flux_x     = x output grid containing the flux (rgrid *).
 * flux_y     = y output grid containing the flux (rgrid *).
 * flux_z     = z output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 * NOTE: This is not the liquid velocity! Divide by density (rho) to get v (velocity):
 *       v_i = flux_i / rho (i = x, y, z).
 *
 */

EXPORT void grid_wf_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {
  
  /*
   * J(r) = -i (hbar/2m) ( psi^* grad psi - psi grad psi^* )
   *      = (hbar/m) Im[ psi^* grad psi ] 
   */
  grid_wf_probability_flux_x(gwf, flux_x);
  grid_wf_probability_flux_y(gwf, flux_y);
  grid_wf_probability_flux_z(gwf, flux_z);
}

/*
 * Calculate energy for the wavefunction. 
 * Includes -E_{kin} * n if the frame of reference has momentum != 0.
 *
 * gwf       = wavefunction for the energy calculation (wf *).
 * potential = grid containing the potential (cgrid *).
 * workspace = additional storage needed (cgrid *).
 *
 * Returns the energy (REAL).
 *
 */

EXPORT REAL grid_wf_energy(wf *gwf, cgrid *potential, cgrid *workspace) {

  REAL mass=gwf->mass, kx = gwf->grid->kx0, ky = gwf->grid->ky0, kz = gwf->grid->kz0;
  REAL ekin = -HBAR * HBAR * (kx * kx + ky * ky + kz * kz) / (2.0 * mass);

  if(ekin != 0.0) ekin *= CREAL(cgrid_integral_of_square(gwf->grid));

  if (gwf->boundary == WF_DIRICHLET_BOUNDARY)
    return grid_wf_energy_cn(gwf, gwf, potential, workspace) + ekin;
  else if (gwf->boundary == WF_PERIODIC_BOUNDARY
		  || gwf->boundary == WF_NEUMANN_BOUNDARY
		  || gwf->boundary == WF_VORTEX_X_BOUNDARY
		  || gwf->boundary == WF_VORTEX_Y_BOUNDARY
		  || gwf->boundary == WF_VORTEX_Z_BOUNDARY)
      	  return grid_wf_energy_fft(gwf, potential, workspace) + ekin;
  else
    abort();
}

/*
 * Auxiliary routine for calculating potential energy.
 * 
 * gwf       = wavefunction for potential energy calculation (wf *).
 * workspace = additional workspace required for the operation (cgrid *).
 *
 * Returns the potential energy.
 *
 */

EXPORT REAL grid_wf_potential_energy(wf *gwf, cgrid *potential) {

  return CREAL(cgrid_grid_expectation_value(gwf->grid, potential));
}

/*
 * Propagate wavefunction in time subject to given potential.
 *
 * gwf         = wavefunction to be propagated (wf *).
 * potential   = grid containing the potential (cgrid *).
 * sq_grad_pot = grid containing square of potential gradient (cgrid *).
 * time        = time step (REAL complex). Note this may be either real or imaginary.
 * workspace   = additional workspace needed for the operation (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate(wf *gwf, cgrid *potential, cgrid *sq_grad_pot, REAL complex time, cgrid *workspace) {  
  
  REAL complex half_time = 0.5 * time;
  REAL complex one_sixth_time = time / 6.0;
  REAL complex two_thirds_time = 2.0 * time / 3.0;
  
  if (gwf->boundary == WF_DIRICHLET_BOUNDARY && gwf->propagator == WF_2ND_ORDER_PROPAGATOR) {    
    grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
    grid_wf_propagate_cn(gwf, NULL, time, NULL, potential, workspace->value, ((INT) sizeof(REAL complex)) * workspace->nx * workspace->ny * workspace->nz);
    grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
  } else if ((gwf->boundary == WF_PERIODIC_BOUNDARY || gwf->boundary == WF_NEUMANN_BOUNDARY || gwf->boundary == WF_VORTEX_X_BOUNDARY || gwf->boundary == WF_VORTEX_Y_BOUNDARY || gwf->boundary == WF_VORTEX_Z_BOUNDARY)
	     && gwf->propagator == WF_2ND_ORDER_PROPAGATOR) {
    grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
    grid_wf_propagate_kinetic_fft(gwf, time);
    grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
  } else if (gwf->boundary == WF_PERIODIC_BOUNDARY
            && gwf->propagator == WF_4TH_ORDER_PROPAGATOR) {
    grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);
    grid_wf_propagate_kinetic_fft(gwf, half_time);    
    cgrid_copy(workspace, potential);
    cgrid_add_scaled(workspace, (1/48.0 * HBAR * HBAR / gwf->mass) * sqnorm(time), sq_grad_pot);	
    grid_wf_propagate_potential(gwf, NULL, two_thirds_time, NULL, workspace);
    
    grid_wf_propagate_kinetic_fft(gwf, half_time);
    grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);
  } else {
    fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator - boundary value combination.\n");
    abort();
  }
}

/*
 * Auxiliary routine to propagate potential energy.
 *
 * gwf       = wavefunction to be propagated (wf *).
 * time      = time step function (REAL complex (*time)(INT, INT, INT, void *, REAL complex)). If NULL, tstep will be used.
 * tstep     = time step (REAL complex).
 * privdata  = private data for time step function (void *).
 * potential = grid containing the potential (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_potential(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *potential) {

  INT i, j, ij, ijnz, k, ny = gwf->grid->ny, nxy = gwf->grid->nx * ny, nz = gwf->grid->nz;
  REAL complex c, *psi = gwf->grid->value, *pot = potential->value;
  
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_propagate_potential(gwf, time, tstep, privdata, potential)) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,psi,pot,time,tstep,privdata) private(i, j, k, ij, ijnz, c) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* psi(t+dt) = exp(- i V dt / hbar) psi(t) */
      if(time) c = -I * (*time)(i, j, k, privdata, tstep) / HBAR;
      else c = -I * tstep / HBAR;
      psi[ijnz + k] *= CEXP(c * pot[ijnz + k]);
    }
  }
}

/*
 * Produce density grid from a given wavefunction.
 *
 * gwf     = wavefunction (wf *).
 * density = output density grid (cgrid *).
 *
 * No return value.
 *
 */

EXPORT inline void grid_wf_density(wf *gwf, rgrid *density) {

  INT ij, k, ijnz, ijnz2, nxy = gwf->grid->nx * gwf->grid->ny, nz = gwf->grid->nz, nzz = density->nz2;
  REAL complex *avalue = gwf->grid->value;
  REAL *cvalue = density->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_density(gwf, density)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,cvalue) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz2 + k] = (REAL) (CONJ(avalue[ijnz + k]) * avalue[ijnz + k]);
  }
}

/*
 * Zero wavefunction.
 *
 * gwf = wavefunction to be zeroed (wf *).
 *
 * No return value.
 *
 */

EXPORT inline void grid_wf_zero(wf *gwf) { 

  cgrid_zero(gwf->grid); 
}

/*
 * Set wavefunction to some constant value.
 *
 * gwf = wavefunction to be set (wf *).
 * c   = value (REAL complex).
 *
 * No return value.
 *
 */

EXPORT inline void grid_wf_constant(wf *gwf, REAL complex c) { 

  cgrid_constant(gwf->grid, c); 
}

/*
 * Map a given function on a wavefunction.
 *
 * gwf  = wavefunction where function will be mapped to (wf *).
 * func = function providing the mapping (REAL complex (*)(void *, REAL)).
 * farg = optional argument for passing parameters to func (void *).
 *
 * No return value.
 *
 */

EXPORT inline void grid_wf_map(wf *gwf, REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg) { 

  cgrid_map(gwf->grid, func, farg); 
}

/*
 * Calculate the norm of the given wavefunction.
 *
 * gwf = wavefunction for the calculation (wf *).
 *
 * Returns the norm (REAL).
 *
 */

EXPORT inline REAL grid_wf_norm(wf *gwf) { 

  return cgrid_integral_of_square(gwf->grid); 
}

/*
 * Normalize wavefunction (to the value given in gwf->norm).
 *
 * gwf = wavefunction to be normalized (wf *).
 *
 * Returns the normalization constant (REAL).
 *
 */

EXPORT inline REAL grid_wf_normalize(wf *gwf) { 

  REAL norm = grid_wf_norm(gwf);

  cgrid_multiply(gwf->grid, SQRT(gwf->norm / norm));
  return norm; 
}

/*
 * Calculate overlap between two wavefunctions.
 *
 * gwfa = 1st wavefunction (wf *).
 * gwfb = 2nd wavefunction (wf *).
 *
 * Returns the overlap (REAL complex).
 *
 */

EXPORT inline REAL complex grid_wf_overlap(wf *gwfa, wf *gwfb) { 

  return cgrid_integral_of_conjugate_product(gwfa->grid, gwfb->grid); 
}

/*
 * Output wavefunction.
 *
 * gwf = wavefunction to be printed (wf *).
 * out = output file pointer (FILE *).
 *
 * No return value.
 *
 */

EXPORT inline void grid_wf_print(wf *gwf, FILE *out) { 

  cgrid_print(gwf->grid, out); 
}
