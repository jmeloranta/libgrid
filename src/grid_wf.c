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
 *              WF_FFT_EEE_BOUNDARY   = Even/Even/Even boundary condition.
 *              WF_FFT_OEE_BOUNDARY   = Odd/Even/Even boundary condition.
 *              WF_FFT_EOE_BOUNDARY   = Even/Odd/Even boundary condition.
 *              WF_FFT_EEO_BOUNDARY   = Even/Even/Odd boundary condition.
 *              WF_FFT_OOE_BOUNDARY   = Odd/Odd/Even boundary condition.
 *              WF_FFT_EOO_BOUNDARY   = Even/Odd/Odd boundary condition.
 *              WF_FFT_OEO_BOUNDARY   = Odd/Even/Odd boundary condition.
 *              WF_FFT_OOO_BOUNDARY   = Odd/Odd/Odd boundary condition.
 * propagator = which time propagator to use for this wavefunction (char):
 *              WF_2ND_ORDER_FFT      = 2nd order in time (FFT).
 *              WF_4TH_ORDER_FFT      = 4th order in time (FFT).
 *              WF_2ND_ORDER_CFFT     = 2nd order in time (FFT with Cayley's approx).
 *              WF_4TH_ORDER_CFFT     = 4th order in time (FFT with Cayley's approx).
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
  
  if(boundary < WF_DIRICHLET_BOUNDARY || boundary > WF_FFT_OOO_BOUNDARY) {
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
  
  value_outside = NULL;
  switch(boundary) {
    case WF_DIRICHLET_BOUNDARY:
      value_outside = CGRID_DIRICHLET_BOUNDARY;
      break;
    case WF_NEUMANN_BOUNDARY:
      value_outside = CGRID_NEUMANN_BOUNDARY;
      break;
    case WF_PERIODIC_BOUNDARY:
      value_outside = CGRID_PERIODIC_BOUNDARY;
      break;
    case WF_FFT_EEE_BOUNDARY:
      value_outside = CGRID_FFT_EEE_BOUNDARY;
      break;
    case WF_FFT_OEE_BOUNDARY:
      value_outside = CGRID_FFT_OEE_BOUNDARY;
      break;
    case WF_FFT_EOE_BOUNDARY:
      value_outside = CGRID_FFT_EOE_BOUNDARY;
      break;
    case WF_FFT_EEO_BOUNDARY:
      value_outside = CGRID_FFT_EEO_BOUNDARY;
      break;
    case WF_FFT_OOE_BOUNDARY:
      value_outside = CGRID_FFT_OOE_BOUNDARY;
      break;
    case WF_FFT_EOO_BOUNDARY:
      value_outside = CGRID_FFT_EOO_BOUNDARY;
      break;
    case WF_FFT_OEO_BOUNDARY:
      value_outside = CGRID_FFT_OEO_BOUNDARY;
      break;
    case WF_FFT_OOO_BOUNDARY:
      value_outside = CGRID_FFT_OOO_BOUNDARY;
      break;
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
 * Set up absorbing boundaries.
 *
 * gwf  = wavefunction (wf *; input/output).
 * gwfp = predict wavefunction (wf *; input/output). Set to NULL if predict-correct not used.
 * amp  = amplitude for boundary damping (REAL complex; input). Use 0.0 to remove the boundary. Relative to the baseline value defined
 *        in src/defs.h. To get real penalty potential, use amp = I * value (real value corresponds to imaginary potential).
 * rho0 = desired value for |\psi|^2 at the boundary (complex potential only) (REAL; input).
 * lx   = lower bound index (x) for the boundary (INT; input).
 * hx   = upper bound index (x) for the boundary (INT; input).
 * ly   = lower bound index (y) for the boundary (INT; input).
 * hy   = upper bound index (y) for the boundary (INT; input).
 * lz   = lower bound index (z) for the boundary (INT; input).
 * hz   = upper bound index (z) for the boundary (INT; input).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_boundary(wf *gwf, wf *gwfp, REAL complex amp, REAL rho0, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  if(amp == 0.0) {
    gwf->ts_func = NULL;
    if(gwfp) gwfp->ts_func = NULL;
    return;
  }

  if(gwf->propagator < WF_2ND_ORDER_CN) {
    gwf->abs_data.amp = amp * GRID_ABS_BC_FFT;
    gwf->abs_data.rho0 = rho0;
  } else {
    gwf->abs_data.amp = 1.0;  // not used
    gwf->abs_data.rho0 = 0.0;
  }
  gwf->abs_data.data[0] = lx;
  gwf->abs_data.data[1] = hx;
  gwf->abs_data.data[2] = ly;
  gwf->abs_data.data[3] = hy;
  gwf->abs_data.data[4] = lz;
  gwf->abs_data.data[5] = hz;
  if(gwfp) {
    if(gwfp->propagator < WF_2ND_ORDER_CN) {
      gwfp->abs_data.amp = amp * GRID_ABS_BC_FFT;
      gwfp->abs_data.rho0 = rho0;
    } else {
      gwfp->abs_data.amp = 1.0;
      gwfp->abs_data.rho0 = 0.0;
    }
    gwfp->abs_data.data[0] = lx;
    gwfp->abs_data.data[1] = hx;
    gwfp->abs_data.data[2] = ly;
    gwfp->abs_data.data[3] = hy;
    gwfp->abs_data.data[4] = lz;
    gwfp->abs_data.data[5] = hz;
  }

  gwf->ts_func = &grid_wf_absorb;
  if(gwfp) gwfp->ts_func = &grid_wf_absorb;
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
    abort();
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
 *               This will specify amp, lx, hx, ly, hy, lz, hz.
 * 
 * Returns the scaling factor for imaginary time (value between 0 and 1.0).
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

  t *= 2.0 / 3.0;  // average of the three directions and 2x to saturate to full imag time at half point
  if(t > 1.0) return 1.0; // Maximum is one
  return t;
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
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  
  switch(gwfp->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwfp->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, cons);
      /* continue with correct cycle */
      break;
    case WF_2ND_ORDER_CFFT:
      if(gwfp->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, cons);
      /* continue with correct cycle */
      break;
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CFFT:
    case WF_4TH_ORDER_CN:
      fprintf(stderr, "libgrid: 4th order propagator not implemented for predict-correct.\n");
      abort();
    case WF_2ND_ORDER_CN:
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, 0.0);
      /* continue with correct cycle */
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
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  
  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      grid_wf_propagate_potential(gwf, time, potential, cons);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      /* end correct cycle */
      break;
    case WF_2ND_ORDER_CFFT:
      grid_wf_propagate_potential(gwf, time, potential, cons);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      /* end correct cycle */
      break;
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CFFT:
    case WF_4TH_ORDER_CN:
      fprintf(stderr, "libgrid: 4th order propagator not implemented for predict-correct.\n");
      abort();
    case WF_2ND_ORDER_CN:
      grid_wf_propagate_potential(gwf, time, potential, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      /* end correct cycle */
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
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  
  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      grid_wf_propagate_potential(gwf, half_time, potential, cons);
      grid_wf_propagate_kinetic_fft(gwf, time);
      grid_wf_propagate_potential(gwf, half_time, potential, cons);
      break;
    case WF_4TH_ORDER_FFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      grid_wf_propagate_kinetic_fft(gwf, half_time);    
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
      cgrid_add_scaled(gwf->cworkspace, ((1.0 / 48.0) * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
      grid_wf_propagate_potential(gwf, two_thirds_time, potential, cons);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      break;
    case WF_2ND_ORDER_CFFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      grid_wf_propagate_potential(gwf, half_time, potential, cons);
      grid_wf_propagate_kinetic_cfft(gwf, time);
      grid_wf_propagate_potential(gwf, half_time, potential, cons);
      break;
    case WF_4TH_ORDER_CFFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);    
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
      cgrid_add_scaled(gwf->cworkspace, ((1.0 / 48.0) * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
      grid_wf_propagate_potential(gwf, two_thirds_time, potential, cons);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      break;
    case WF_4TH_ORDER_CN:
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace2, potential);
      cgrid_add_scaled(gwf->cworkspace, (1.0/48.0 * HBAR * HBAR / gwf->mass) * sqnorm(time), gwf->cworkspace2);
      grid_wf_propagate_potential(gwf, two_thirds_time, gwf->cworkspace, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, 0.0);
      break;
    case WF_2ND_ORDER_CN:
      grid_wf_propagate_potential(gwf, half_time, potential, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, time);
      grid_wf_propagate_potential(gwf, half_time, potential, 0.0);
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
 * tstep     = time step (REAL complex).
 * potential = grid containing the potential (cgrid *). If NULL, no propagation needed.
 * cons      = constant term to add to potential (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_potential(wf *gwf, REAL complex tstep, cgrid *potential, REAL cons) {

  INT i, j, ij, ijnz, k, ny = gwf->grid->ny, nxy = gwf->grid->nx * ny, nz = gwf->grid->nz;
  REAL complex c, *psi = gwf->grid->value, *pot = potential->value;
  REAL (*time)(INT, INT, INT, void *) = gwf->ts_func, tmp;
  struct grid_abs *privdata = &(gwf->abs_data);
  char add_abs = 0;  
  REAL rho0 = privdata->rho0;
  REAL complex amp = privdata->amp;

  if(!potential) return;
#ifdef USE_CUDA
  if(gwf->ts_func && gwf->ts_func != grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    abort();
  }
  if(cuda_status() && !grid_cuda_wf_propagate_potential(gwf, tstep, potential, cons)) return;
#endif
  if(gwf->propagator < WF_2ND_ORDER_CN) {
    if(time) add_abs = 1;   /* abs potential for FFT */
    time = NULL;            /* Imag time scheme disabled for FFT */
  }

#pragma omp parallel for firstprivate(amp,cons,add_abs,rho0,ny,nz,nxy,psi,pot,time,tstep,privdata) private(tmp, i, j, k, ij, ijnz, c) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* psi(t+dt) = exp(- i V dt / hbar) psi(t) */
      if(time) {
        tmp = ((*time)(i, j, k, privdata));
        c = -I * ((1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp) / HBAR;
      } else c = -I * tstep / HBAR;
      if(add_abs) psi[ijnz + k] = psi[ijnz + k] * 
         CEXP(c * (pot[ijnz + k] - I * amp * grid_wf_absorb(i, j, k, privdata) * (sqnorm(psi[ijnz + k]) - rho0)));
      else psi[ijnz + k] = psi[ijnz + k] * CEXP(c * (cons + pot[ijnz + k]));
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
