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
 * boundary   = boundary condition (char):
 *              WF_DIRICHLET_BOUNDARY = Dirichlet boundary condition.
 *              WF_NEUMANN_BOUNDARY   = Neumann boundary condition.
 *              WF_PERIODIC_BOUNDARY  = Periodic boundary condition.
 *              WF_VORTEX_X_BOUNDARY  = Vortex line along X.
 *              WF_VORTEX_Y_BOUNDARY  = Vortex line along Y.
 *              WF_VORTEX_Z_BOUNDARY  = Vortex line along Z.
 * propagator = which time propagator to use for this wavefunction (char):
 *              WF_2ND_ORDER_FFT      = 2nd order in time (FFT).
 *              WF_4TH_ORDER_FFT      = 4th order in time (FFT).
 *              WF_2ND_ORDER_CN       = 2nd order in time with Crank-Nicolson propagator.
 *              WF_4TH_ORDER_CN       = 4th order in time with Crank-Nicolson propagator.
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
  
  if(propagator < WF_2ND_ORDER_FFT || propagator > WF_4TH_ORDER_CN) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Unknown propagator.\n");
    return 0;
  }
  
  if (boundary == WF_DIRICHLET_BOUNDARY && propagator < WF_2ND_ORDER_CN) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Invalid boundary condition - propagator combination. Dirichlet condition can only be used with Crank-Nicolson propagator.\n");
    return 0;
  }
  
  if(!(gwf = (wf *) malloc(sizeof(wf)))) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Could not allocate memory for wf.\n");
    return 0;
  }
  
  switch(boundary) {
    case WF_DIRICHLET_BOUNDARY:
      value_outside = cgrid_value_outside_constantdirichlet;
      break;
    case WF_NEUMANN_BOUNDARY:
      value_outside = cgrid_value_outside_neumann;
      break;
    case WF_PERIODIC_BOUNDARY:
      value_outside = cgrid_value_outside;
      break;
    case WF_VORTEX_X_BOUNDARY:
      value_outside = cgrid_value_outside_vortex_x;
      break;
    case WF_VORTEX_Y_BOUNDARY:
      value_outside = cgrid_value_outside_vortex_y;
      break;
    case WF_VORTEX_Z_BOUNDARY:
      value_outside = cgrid_value_outside_vortex_z;
      break;
    default:
      fprintf(stderr, "libgrid: Unknown boundary condition in grid_wf_alloc().\n");
      exit(1);
  }
  
  if(!(gwf->grid = cgrid_alloc(nx, ny, nz, step, value_outside, 0, id))) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Could not allocate memory for wf->grid.\n");
    free(gwf);
    return NULL;
  }
  
  gwf->mass = mass;
  gwf->norm = 1.0;
  gwf->boundary = boundary;
  gwf->propagator = propagator;
  gwf->cworkspace = NULL;
  gwf->cworkspace2 = NULL;
  gwf->cworkspace3 = NULL;
  gwf->ts_func = NULL;
  
  return gwf;
}

/*
 * "Clone" a wave function: Allocate a wave function with idential parameters.
 *
 * gwf = Wavefunction to be cloned (wf *; input).
 * id  = Comment string describing the WF (char *; input).
 *
 * Returns pointer to new wave function (wf *).
 *
 */

EXPORT wf *grid_wf_clone(wf *gwf, char *id) {

  wf *nwf;
  cgrid *grid = gwf->grid;

  if(!(nwf = (wf *) malloc(sizeof(wf)))) {
    fprintf(stderr, "libgrid: Out of memory in grid_wf_clone().\n");
    exit(1);
  }
  bcopy((void *) gwf, (void *) nwf, sizeof(wf));
  if(!(nwf->grid = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, 0, id))) {
    fprintf(stderr, "libgrid: Error in grid_wf_clone(). Could not allocate memory for nwf->grid.\n");
    free(nwf);
    return NULL;
  }
  return nwf;
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
    if (gwf->cworkspace) cgrid_free(gwf->cworkspace);
    if (gwf->cworkspace2) cgrid_free(gwf->cworkspace2);
    if (gwf->cworkspace3) cgrid_free(gwf->cworkspace3);
    free(gwf);
  }
}

/* 
 * Absorbing boundary amplitude (between zero and one).
 *
 * i           = Current index along X (INT; input).
 * j           = Current index along Y (INT; input).
 * k           = Current index along Z (INT; input).
 * data        = Pointer to struct grid_abs holding values for specifying the absorbing region (void *; INPUT).
 *               This will specify lx, hx, ly, hy, lz, hz.
 * 
 * Returns the scaling factor for imaginary time.
 *
 */

EXPORT REAL grid_wf_absorb(INT i, INT j, INT k, void *data) {

  REAL t;
  struct grid_abs *ab = (struct grid_abs *) data;
  INT lx = ab->data[0], hx = ab->data[1], ly = ab->data[2], hy = ab->data[3], lz = ab->data[4], hz = ab->data[5];

  t = 0.0;

  if(i < lx) t += ((REAL) (lx - i)) / (REAL) lx;
  else if(i > hx) t += ((REAL) (i - hx)) / (REAL) lx;

  if(j < ly) t += ((REAL) (ly - j)) / (REAL) ly;
  else if(j > hy) t += ((REAL) (j - hy)) / (REAL) ly;

  if(k < lz) t += ((REAL) (lz - k)) / (REAL) lz;
  else if(k > hz) t += ((REAL) (k - hz)) / (REAL) lz;

  return t / 3.0;
}

/*
 * Calculate total energy for the wavefunction. 
 * Includes -E_{kin} * n if the frame of reference has momentum != 0.
 *
 * gwf       = wavefunction for the energy calculation (wf *).
 * potential = grid containing the potential (rgrid *).
 *
 * Returns the energy (REAL).
 *
 */

EXPORT REAL grid_wf_energy(wf *gwf, rgrid *potential) {

  if (gwf->propagator == WF_2ND_ORDER_CN || gwf->propagator == WF_4TH_ORDER_CN) return grid_wf_energy_cn(gwf, potential);
  else return grid_wf_energy_fft(gwf, potential); /* else FFT */
}

/*
 * Calculate kinetic energy for the wavefunction. 
 * Includes -E_{kin} * n if the frame of reference has momentum != 0.
 *
 * gwf       = wavefunction for the energy calculation (wf *).
 *
 * Returns the energy (REAL).
 *
 */

EXPORT REAL grid_wf_kinetic_energy(wf *gwf) {

  return grid_wf_energy(gwf, NULL);
}

/*
 * Calcucate potential energy.
 * 
 * gwf       = wavefunction for potential energy calculation (wf *).
 * potential = potential energy (rgrid *).
 *
 * Returns the potential energy.
 *
 */

EXPORT REAL grid_wf_potential_energy(wf *gwf, rgrid *potential) {

  cgrid *grid = gwf->grid;

  // TODO: This should be done better...
  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  grid_real_to_complex_re(gwf->cworkspace, potential);

  return CREAL(cgrid_grid_expectation_value(gwf->grid, gwf->cworkspace));
}

/*
 * Propagate (PREDICT) wavefunction in time subject to given potential.
 *
 * gwf         = wavefunction to be propagated; wf up to kinetic propagation (wf *).
 * gwfp        = wavefunction to be propagated; predicted (wf *).
 * potential   = grid containing the potential (cgrid *).
 * time        = time step (REAL complex). Note this may be either real or imaginary.
 *
 * No return value.
 *
 * NOTE: FFT can only do absorption for the potential part (CN does both).
 *
 */

EXPORT void grid_wf_propagate_predict(wf *gwf, wf *gwfp, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  
  switch(gwfp->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwfp->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        exit(1);
      }
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      cgrid_copy(gwfp->grid, gwf->grid);
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwfp, grid_wf_absorb, time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwfp, NULL, time, NULL, potential);
      /* continue with correct cycle */
      break;
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CN:
      fprintf(stderr, "libgrid: 4th order propagator not implemented for predict-correct.\n");
      exit(1);
    case WF_2ND_ORDER_CN:
      if(gwfp->ts_func) {
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, half_time, &(gwfp->abs_data));
        cgrid_copy(gwfp->grid, gwf->grid);
        grid_wf_propagate_potential(gwfp, grid_wf_absorb, time, &(gwfp->abs_data), potential);
      /* continue with correct cycle */
      } else {
        grid_wf_propagate_kinetic_cn(gwf, NULL, half_time, NULL);
        cgrid_copy(gwfp->grid, gwf->grid);
        grid_wf_propagate_potential(gwfp, NULL, time, NULL, potential);
        /* continue with correct cycle */
      }
      break;        
    default:
      fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator.\n");
      abort();
  }
}

/*
 * Propagate (CORRECT) wavefunction in time subject to given potential.
 *
 * gwf         = wavefunction to be propagated (wf *).
 * potential   = grid containing the potential (cgrid *).
 * time        = time step (REAL complex). Note this may be either real or imaginary.
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_correct(wf *gwf, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  
  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, time, NULL, potential);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      /* end correct cycle */
      break;
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CN:
      fprintf(stderr, "libgrid: 4th order propagator not implemented for predict-correct.\n");
      exit(1);
    case WF_2ND_ORDER_CN:
      if(gwf->ts_func) {
        grid_wf_propagate_potential(gwf, grid_wf_absorb, time, &(gwf->abs_data), potential);
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, half_time, &(gwf->abs_data));
        /* end correct cycle */
      } else {
        grid_wf_propagate_potential(gwf, NULL, time, NULL, potential);
        grid_wf_propagate_kinetic_cn(gwf, NULL, half_time, NULL);
        /* end correct cycle */
      }
      break;        
    default:
      fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator.\n");
      abort();
  }
}

/*
 * Propagate wavefunction in time subject to given potential.
 *
 * gwf         = wavefunction to be propagated (wf *).
 * potential   = grid containing the potential (cgrid *).
 * time        = time step (REAL complex). Note this may be either real or imaginary.
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate(wf *gwf, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  REAL complex one_sixth_time = time / 6.0;
  REAL complex two_thirds_time = 2.0 * time / 3.0;
  cgrid *grid = gwf->grid;
  
  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        exit(1);
      }
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, half_time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
      grid_wf_propagate_kinetic_fft(gwf, time);
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, half_time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
      break;
    case WF_4TH_ORDER_FFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        exit(1);
      }
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace2");
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, one_sixth_time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);
      grid_wf_propagate_kinetic_fft(gwf, half_time);    
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
      cgrid_add_scaled(gwf->cworkspace, (1/48.0 * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, two_thirds_time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, two_thirds_time, NULL, gwf->cworkspace);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      if(gwf->ts_func)
        grid_wf_propagate_potential(gwf, grid_wf_absorb, one_sixth_time, &(gwf->abs_data), potential);
      else
        grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);
      break;
    case WF_4TH_ORDER_CN:
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace2");
      if(gwf->ts_func) {
        grid_wf_propagate_potential(gwf, grid_wf_absorb, one_sixth_time, &(gwf->abs_data), potential);
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, half_time, &(gwf->abs_data));    
        cgrid_copy(gwf->cworkspace, potential);
        grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
        cgrid_add_scaled(gwf->cworkspace, (1/48.0 * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
        grid_wf_propagate_potential(gwf, grid_wf_absorb, two_thirds_time, &(gwf->abs_data), gwf->cworkspace);
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, half_time, &(gwf->abs_data));
        grid_wf_propagate_potential(gwf, grid_wf_absorb, one_sixth_time, &(gwf->abs_data), potential);
      } else {
        grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);
        grid_wf_propagate_kinetic_cn(gwf, NULL, half_time, NULL);    
        cgrid_copy(gwf->cworkspace, potential);
        grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
        cgrid_add_scaled(gwf->cworkspace, (1/48.0 * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
        grid_wf_propagate_potential(gwf, NULL, two_thirds_time, NULL, gwf->cworkspace);
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, half_time, &(gwf->abs_data));
        grid_wf_propagate_potential(gwf, NULL, one_sixth_time, NULL, potential);    
      }
      break;
    case WF_2ND_ORDER_CN:
      if(gwf->ts_func) {
        grid_wf_propagate_potential(gwf, grid_wf_absorb, half_time, &(gwf->abs_data), potential);
        grid_wf_propagate_kinetic_cn(gwf, grid_wf_absorb, time, &(gwf->abs_data));
        grid_wf_propagate_potential(gwf, grid_wf_absorb, half_time, &(gwf->abs_data), potential);
      } else {
        grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
        grid_wf_propagate_kinetic_cn(gwf, NULL, time, NULL);
        grid_wf_propagate_potential(gwf, NULL, half_time, NULL, potential);
      }
      break;        
    default:
      fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator.\n");
      abort();
  }
}

/*
 * Auxiliary routine to propagate potential energy (only used with FFT propagation of kinetic energy; CN includes potential).
 *
 * gwf       = wavefunction to be propagated (wf *).
 * time      = time step function (REAL (*time)(INT, INT, INT, void *, REAL complex)). If NULL, tstep will be used.
 * tstep     = time step (REAL complex).
 * privdata  = private data for time step function (void *).
 * potential = grid containing the potential (cgrid *). If NULL, no propagation needed.
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_potential(wf *gwf, REAL (*time)(INT, INT, INT, void *), REAL complex tstep, void *privdata, cgrid *potential) {

  INT i, j, ij, ijnz, k, ny = gwf->grid->ny, nxy = gwf->grid->nx * ny, nz = gwf->grid->nz;
  REAL complex c, *psi = gwf->grid->value, *pot = potential->value;
  
  if(!potential) return;
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
      if(time) c = (1.0 / HBAR) * (CIMAG(tstep) * grid_wf_absorb(i, j, k, privdata) - I * CREAL(tstep));
      else c = -I * tstep / HBAR;
      psi[ijnz + k] = psi[ijnz + k] * CEXP(c * pot[ijnz + k]);
    }
  }
}

/*
 * Produce density grid from a given wavefunction.
 *
 * gwf     = wavefunction (wf *).
 * density = output density grid (rgrid *).
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
 * func = function providing the mapping (REAL complex (*)(void *, REAL, REAL, REAL)).
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

/*
 * Add complex absorbing potential: -I * amp * (|psi|^2 - rho0).
 *
 * gwf   = current wavefunction (wf *; input).
 * pot   = potential to which the absorption is added (cgrid *; input/output).
 * amp   = amplitude (REAL; input).
 * rho0  = baseline density (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_absorb_potential(wf *gwf, cgrid *pot, REAL amp, REAL rho0) {

  REAL complex *val = pot->value;
  INT i, j, k, ij, ijnz, nxy = pot->nx * pot->ny, ny = pot->ny, nz = pot->nz;
  REAL g;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_absorb_potential(gwf, pot, amp, rho0)) return;
#endif
#pragma omp parallel for firstprivate(nxy, nz, ny, gwf, amp, val, rho0) private(i, j, k, ijnz, g) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      g = grid_wf_absorb(i, j, k, &(gwf->abs_data));
      if(g > 0.0) val[ijnz + k] += -I * g * amp * (sqnorm(gwf->grid->value[ijnz + k]) - rho0);
    }
  }  
}

