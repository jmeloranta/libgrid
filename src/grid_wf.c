/*
 * Routines for handling wavefunctions.
 *
 */

#include "grid.h"
#include "cprivate.h"

extern char grid_analyze_method;

/*
 * @FUNC{grid_wf_alloc, "Allocate wavefunction"}
 * @DESC{"This function allocates memory for wavefunction"}
 * @ARG1{INT nx, "Number of spatial grid points along x (runs slowest in memory)"}
 * @ARG2{INT ny, "Number of spatial grid points along y"}
 * @ARG3{INT nz, "Number of spatial grid points along z (runs fastest in memory)"}
 * @ARG4{REAL step, "Spatial grid step size"}
 * @ARG5{REAL mass, "Mass of the particle corresponding to this wavefunction"}
 * @ARG6{char boundary, "Boundary condition: WF_DIRICHLET_BOUNDARY = Dirichlet boundary condition, WF_NEUMANN_BOUNDARY = Neumann boundary condition, WF_PERIODIC_BOUNDARY = Periodic boundary condition"}
 * @ARG7{char propagator, "Which time propagator to use for this wavefunction: WF_2ND_ORDER_FFT = 2nd order in time (FFT), WF_4TH_ORDER_FFT = 4th order in time (FFT), WF_2ND_ORDER_CFFT = 2nd order in time (FFT with cutoff in k-space), WF_4TH_ORDER_CFFT = 4th order in time (FFT with cutoff in k-space), WF_2ND_ORDER_CN = 2nd order in time with Crank-Nicolson propagator, WF_4TH_ORDER_CN = 4th order in time with Crank-Nicolson propagator, WF_1ST_ORDER_EULER = Explicit Euler"}
 * @ARG8{char *id, "String identifier for the grid"}
 * @RVAL{cgrid *, "Return value is a pointer to the allocated wavefunction. This routine returns NULL if allocation fails"}
 *
 */

EXPORT wf *grid_wf_alloc(INT nx, INT ny, INT nz, REAL step, REAL mass, char boundary, char propagator, char *id) {

  wf *gwf;
  REAL complex (*value_outside)(struct cgrid_struct *grid, INT i, INT j, INT k);
  
  if(boundary < WF_DIRICHLET_BOUNDARY || boundary > WF_PERIODIC_BOUNDARY) {
    fprintf(stderr, "libgrid: Error in grid_wf_alloc(). Unknown boundary condition.\n");
    return 0;
  }
  
  if(propagator < WF_2ND_ORDER_FFT || propagator > WF_1ST_ORDER_EULER) {
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
  gwf->cfft_width = 1.0;  // window width: between 0 and 2.
  gwf->kmax = 9999.0;     // Default not to use

  return gwf;
}

/*
 * @FUNC{grid_wf_boundary, "Set up absorbing boundaries"}
 * @DESC{"Set up absorbing boundaries. To clear the boundaries, call with lx = hx = ly = hy = lz = hz = 0"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{wf *gwfp, "Predict wavefunction. Set to NULL if predict-correct not used"}
 * @ARG3{INT lx, "Lower bound index (x) for the boundary"}
 * @ARG4{INT hx, "Upper bound index (x) for the boundary"}
 * @ARG5{INT ly, "Lower bound index (y) for the boundary"}
 * @ARG6{INT hy, "Upper bound index (y) for the boundary"}
 * @ARG7{INT lz, "Lower bound index (z) for the boundary"}
 * @ARG8{INT hz, "Upper bound index (z) for the boundary"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_boundary(wf *gwf, wf *gwfp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  if(!lx && !hx && !ly && !hy && !lz && !hz) {
    gwf->ts_func = NULL;
    gwf->lx = gwf->hx = gwf->ly = gwf->hy = gwf->lz = gwf->hz = 0;
    if(gwfp) {
      gwfp->ts_func = NULL;
      gwfp->lx = gwf->hx = gwf->ly = gwf->hy = gwf->lz = gwf->hz = 0;
    }
    return;
  }

  gwf->lx = lx;
  gwf->hx = hx;
  gwf->ly = ly;
  gwf->hy = hy;
  gwf->lz = lz;
  gwf->hz = hz;

  gwf->ts_func = &grid_wf_absorb;
  if(gwfp) gwfp->ts_func = &grid_wf_absorb;
}

/*
 * @FUNC{grid_wf_clone, "Clone wavefunction"}
 * @DESC{"Clone a wavefunction. Allocate new wavefunction with idential parameters"}
 * @ARG1{wf *gwf, "Wavefunction to be cloned"}
 * @ARG2{char *id, "Comment string describing the WF"}
 * @RVAL{wf *, "Returns pointer to new wavefunction"}
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
 * @FUNC{grid_wf_free, "Free wavefunction"}
 * @DESC{"Free wavefunction memory"}
 * @ARG1{wf *gwf, "Wavefunction to be freed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_wf_absorb, "Absorbing boundary amplitude"}
 * @DESC{"Return the absorbing boundary amplitude (between zero and one) at a given grid point. This is zero
          in the regular domain (real time) whereas it is non-zero in the absorbing region (imaginary time)"}
 * @ARG1{INT i, "Current index along x"}
 * @ARG2{INT j, "Current index along y"}
 * @ARG3{INT k, "Current index along z"}
 * @ARG4{INT lx, "Lower bound index (x) for the boundary"}
 * @ARG5{INT hx, "Upper bound index (x) for the boundary"}
 * @ARG6{INT ly, "Lower bound index (y) for the boundary"}
 * @ARG7{INT hy, "Upper bound index (y) for the boundary"}
 * @ARG8{INT lz, "Lower bound index (z) for the boundary"}
 * @ARG9{INT hz, "Upper bound index (z) for the boundary"}
 * @RVAL{REAL, "Returns the scaling factor between 0.0 and 1.0"}
 *
 */

EXPORT REAL grid_wf_absorb(INT i, INT j, INT k, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  REAL t;

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
 * @FUNC{grid_wf_energy, "Total energy of wavefunction"}
 * @DESC{"Calculate total energy for the wavefunction. This includes $-E_{kin} * n$ ($n$ is the number
          of particles), if the frame of reference has momentum != 0"}
 * @ARG1{wf *gwf, "Wavefunction for the energy calculation"}
 * @ARG2{rgrid *potential, "Grid containing the potential energy"}
 * @RVAL{REAL, "Returns the total energy"}
 *
 */

EXPORT REAL grid_wf_energy(wf *gwf, rgrid *potential) {

  if (gwf->propagator == WF_2ND_ORDER_CN || gwf->propagator == WF_4TH_ORDER_CN || !grid_analyze_method)
    return grid_wf_energy_cn(gwf, potential);  /* If CN or finite difference analysis requested */
  else
    return grid_wf_energy_fft(gwf, potential); /* else FFT */
}

/*
 * @FUNC{grid_wf_kinetic_energy, "Kinetic energy of wavefunction"}
 * @DESC{"Calculate kinetic energy for the wavefunction. This includes $-E_{kin} * n$ 
          ($n$ is the number of particles), if the frame of reference has momentum != 0"}
 * @ARG1{wf *gwf, "Wavefunction for kinetic energy calculation"}
 * @RVAL{REAL, "Returns the kinetic energy"}
 *
 */

EXPORT REAL grid_wf_kinetic_energy(wf *gwf) {

  return grid_wf_energy(gwf, NULL);
}

/*
 * @FUNC{grid_wf_potential_energy, "Potential energy of wavefunction"}
 * @DESC{"Calcucate potential energy of wavefunction"}
 * @ARG1{wf *gwf, "Wavefunction for potential energy calculation"}
 * @ARG2{rgrid *potential, "Potential energy"}
 * @RVAL{REAL, "Returns the potential energy"}
 *
 */

EXPORT REAL grid_wf_potential_energy(wf *gwf, rgrid *potential) {

  return grid_grid_expectation_value(gwf->grid, potential);
}

/*
 * @FUNC{grid_wf_propagate_predict, "Propagate wavefunction (PREDICT)"}
 * @DESC{"Propagate (PREDICT) wavefunction in time subject to given potential.
         FFT propagation can only do absorption for the potential part whereas
         CN can do it for both"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated; wf up to kinetic propagation"}
 * @ARG2{wf *gwfp, "Wavefunction to be propagated; predicted"}
 * @ARG3{cgrid *potential, "Grid containing the potential"}
 * @ARG4{REAL complex time, "Propagation time step. Note this can include real and imaginary parts"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_predict(wf *gwf, wf *gwfp, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  
  switch(gwfp->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwf->ts_func) {
        fprintf(stderr, "libgrid: Predict-correct not available for FFT absorbing BC.\n");
        abort();
      }
      if(gwfp->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, cons);
      /* continue with correct cycle */
      break;
    case WF_2ND_ORDER_CFFT:
      if(gwf->ts_func) {
        fprintf(stderr, "libgrid: Predict-correct not available for CFFT absorbing BC.\n");
        abort();
      }
      if(gwfp->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, cons);
      /* continue with correct cycle */
      break;
    case WF_2ND_ORDER_CN:
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      cgrid_copy(gwfp->grid, gwf->grid);
      grid_wf_propagate_potential(gwfp, time, potential, 0.0);
      /* continue with correct cycle */
      break;        
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CFFT:
    case WF_4TH_ORDER_CN:
    case WF_1ST_ORDER_EULER:
      fprintf(stderr, "libgrid: Propagator not implemented for predict-correct.\n");
      abort();
    default:
      fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator.\n");
      abort();
  }
}

/*
 * @FUNC{grid_wf_propagate_correct, "Propagate wavefunction (CORRECT)"}
 * @DESC{"Propagate (CORRECT) wavefunction in time subject to given potential"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{cgrid *potential, "Grid containing the potential"}
 * @ARG3{REAL complex time, "Time step. Note that this can have real and imaginary parts"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_correct(wf *gwf, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  
  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      grid_wf_propagate_potential(gwf, time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      /* end correct cycle */
      break;
    case WF_2ND_ORDER_CFFT:
      grid_wf_propagate_potential(gwf, time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      /* end correct cycle */
      break;
    case WF_4TH_ORDER_FFT:
    case WF_4TH_ORDER_CFFT:
    case WF_4TH_ORDER_CN:
    case WF_1ST_ORDER_EULER:
      fprintf(stderr, "libgrid: Propagator not implemented for predict-correct.\n");
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
 * @FUNC{grid_wf_propagate, "Propagate wavefunction"}
 * @DESC{"Propagate wavefunction in time subject to given potential"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{cgrid *potential, "Grid containing the potential"}
 * @ARG3{REAL complex time, "Time step. Note that this may have real and imaginary parts"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate(wf *gwf, cgrid *potential, REAL complex time) {  
  
  REAL complex half_time = 0.5 * time;
  REAL complex one_sixth_time = time / 6.0;
  REAL complex two_thirds_time = 2.0 * time / 3.0;
  cgrid *grid = gwf->grid;
  cgrid *gsave;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL cons = -(HBAR * HBAR / (2.0 * gwf->mass)) * (kx0 * kx0 + ky0 * ky0 + kz0 * kz0);
  REAL (*save)(INT, INT, INT, INT, INT, INT, INT, INT, INT);

  switch(gwf->propagator) {
    case WF_2ND_ORDER_FFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with CN.\n");
        abort();
      }
      if(gwf->ts_func && CREAL(time) != 0.0) { // do not include boundary for pure imaginary time propagation
        save = gwf->ts_func;
        gwf->ts_func = NULL;
        if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
        /* Real time step */
        cgrid_copy(gwf->cworkspace, gwf->grid);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, half_time);
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, time, potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, half_time);
        cgrid_inverse_fft_norm(gwf->grid);
        /* Imaginary time step */
        gsave = gwf->grid;
        gwf->grid = gwf->cworkspace;
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, -I * CREAL(half_time));
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, -I * CREAL(time), potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, -I * CREAL(half_time));
        cgrid_inverse_fft_norm(gwf->grid);
        gwf->grid = gsave;
        gwf->ts_func = save;
        /* merge solutions according to the boundary function grid_wf_boundary() */
        grid_wf_merge(gwf, gwf->grid, gwf->cworkspace);
      } else {
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, half_time);
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, time, potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_fft(gwf, half_time);
        cgrid_inverse_fft_norm(gwf->grid);
      }
      break;
    case WF_4TH_ORDER_FFT:
      if(gwf->ts_func) {
        fprintf(stderr, "libgrid: 4th order propagator not available for FFT absorbing BC.\n");
        abort();
      }
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with CN.\n");
        abort();
      }
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      if(!gwf->cworkspace3) gwf->cworkspace3 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 3");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_fft(gwf, half_time);    
      cgrid_inverse_fft_norm(gwf->grid);
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace3, potential);  // Uses cworkspace and cworkspace2 !
      cgrid_add_scaled(gwf->cworkspace, ((1.0 / 48.0) * HBAR * HBAR / gwf->mass) * csqnorm(time), gwf->cworkspace3);
      grid_wf_propagate_potential(gwf, two_thirds_time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_fft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      break;
    case WF_2ND_ORDER_CFFT:
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      if(gwf->ts_func && CREAL(time) != 0.0) { // do not include boundary for pure imaginary time propagation
        save = gwf->ts_func;
        gwf->ts_func = NULL;
        if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
        /* Real time step */
        cgrid_copy(gwf->cworkspace, gwf->grid);
        grid_wf_propagate_potential(gwf, half_time, potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_cfft(gwf, time);
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, half_time, potential, cons);
        /* Imaginary time step */
        gsave = gwf->grid;
        gwf->grid = gwf->cworkspace;
        grid_wf_propagate_potential(gwf, -I * CREAL(half_time), potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_cfft(gwf, -I * CREAL(time));
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, -I * CREAL(half_time), potential, cons);
        gwf->grid = gsave;
        gwf->ts_func = save;
        /* merge solutions according to the boundary function grid_wf_boundary() */
        grid_wf_merge(gwf, gwf->grid, gwf->cworkspace);
      } else {
        grid_wf_propagate_potential(gwf, half_time, potential, cons);
        cgrid_fft(gwf->grid);
        grid_wf_propagate_kinetic_cfft(gwf, time);
        cgrid_inverse_fft_norm(gwf->grid);
        grid_wf_propagate_potential(gwf, half_time, potential, cons);
      }
      break;
    case WF_4TH_ORDER_CFFT:
      if(gwf->ts_func) {
        fprintf(stderr, "libgrid: 4th order propagator not available for FFT absorbing BC.\n");
        abort();
      }
      if(gwf->grid->omega != 0.0) {
        fprintf(stderr, "libgrid: omega != 0.0 allowed only with WF_XX_ORDER_CN.\n");
        abort();
      }
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      if(!gwf->cworkspace3) gwf->cworkspace3 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 3");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);    
      cgrid_inverse_fft_norm(gwf->grid);
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace3, potential);   // Uses cworkspace and cworkspace2 !
      cgrid_add_scaled(gwf->cworkspace, ((1.0 / 48.0) * HBAR * HBAR / gwf->mass) * csqnorm(time), gwf->cworkspace3);
      grid_wf_propagate_potential(gwf, two_thirds_time, potential, cons);
      cgrid_fft(gwf->grid);
      grid_wf_propagate_kinetic_cfft(gwf, half_time);
      cgrid_inverse_fft_norm(gwf->grid);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, cons);
      break;
    case WF_2ND_ORDER_CN:
      grid_wf_propagate_potential(gwf, half_time, potential, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, time);
      grid_wf_propagate_potential(gwf, half_time, potential, 0.0);
      break;        
    case WF_4TH_ORDER_CN:
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
      if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
      if(!gwf->cworkspace3) gwf->cworkspace3 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 3");
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      cgrid_copy(gwf->cworkspace, potential);
      grid_wf_square_of_potential_gradient(gwf, gwf->cworkspace3, potential);   // Uses cworkspace and cworkspace2 !
      cgrid_add_scaled(gwf->cworkspace, (1.0/48.0 * HBAR * HBAR / gwf->mass) * csqnorm(time), gwf->cworkspace3);
      grid_wf_propagate_potential(gwf, two_thirds_time, gwf->cworkspace, 0.0);
      grid_wf_propagate_kinetic_cn(gwf, half_time);
      grid_wf_propagate_potential(gwf, one_sixth_time, potential, 0.0);
      break;
    case WF_1ST_ORDER_EULER: /* explicit Euler (not stable but simple and no op. splitting) */
      if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 1");
      cgrid_laplace(gwf->grid, gwf->cworkspace);
      cgrid_multiply(gwf->cworkspace, -0.5 * HBAR * HBAR / gwf->mass);
      cgrid_add_scaled_product(gwf->cworkspace, 1.0, potential, gwf->grid);
      cgrid_multiply(gwf->cworkspace, -I * time / HBAR);
      cgrid_sum(gwf->grid, gwf->grid, gwf->cworkspace);
      break;      
    default:
      fprintf(stderr, "libgrid: Error in grid_wf_propagate(). Unknown propagator.\n");
      abort();
  }
}

/*
 * @FUNC{grid_wf_propagate_potential, "Propagate potential operator"}
 * @DESC{"Auxiliary routine to propagate potential energy"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex tstep, "Time step"}
 * @ARG3{cgrid *potential, "Grid containing the potential. If NULL, no propagation needed"}
 * @ARG4{REAL cons, "Constant term to add to potential (usually zero)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_potential(wf *gwf, REAL complex tstep, cgrid *potential, REAL cons) {

  INT i, j, ij, ijnz, k, ny = gwf->grid->ny, nxy = gwf->grid->nx * ny, nz = gwf->grid->nz;
  REAL complex c, *psi = gwf->grid->value, *pot = potential->value;
  REAL (*time)(INT, INT, INT, INT, INT, INT, INT, INT, INT) = gwf->ts_func, tmp;
  INT lx = gwf->lx, hx = gwf->hx, ly = gwf->ly, hy = gwf->hy, lz = gwf->lz, hz = gwf->hz;

  if(!potential) return;
#ifdef USE_CUDA
  if(gwf->ts_func && gwf->ts_func != grid_wf_absorb) {
    fprintf(stderr, "libgrid(CUDA): Only grid_wf_absorb function can be used for time().\n");
    abort();
  }
  if(cuda_status() && !grid_cuda_wf_propagate_potential(gwf, tstep, potential, cons)) return;
#endif
  if(gwf->propagator < WF_2ND_ORDER_CN) time = NULL; /* Imag time scheme disabled for FFT */

#pragma omp parallel for firstprivate(cons,ny,nz,nxy,psi,pot,time,tstep,lx,hx,ly,hy,lz,hz) private(tmp, i, j, k, ij, ijnz, c) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* psi(t+dt) = exp(- i V dt / hbar) psi(t) */
      if(time) {
        tmp = ((*time)(i, j, k, lx, hx, ly, hy, lz, hz));
        c = -I * ((1.0 - tmp) * CREAL(tstep) - I * CREAL(tstep) * tmp) / HBAR;
      } else c = -I * tstep / HBAR;
      psi[ijnz + k] = psi[ijnz + k] * CEXP(c * (cons + pot[ijnz + k]));
    }
  }
}

/*
 * @FUNC{grid_wf_density, "Density of wavefunction"}
 * @DESC{"Produce density grid from a given wavefunction"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{rgrid *density, "Output density grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_wf_zero, "Zero wavefunction"}
 * @DESC{"Zero wavefunction"}
 * @ARG1{wf *gwf, "Wavefunction to be zeroed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT inline void grid_wf_zero(wf *gwf) { 

  cgrid_zero(gwf->grid); 
}

/*
 * @FUNC{grid_wf_constant, "Set wavafunction to constant value"}
 * @DESC{"Set wavefunction to some constant value"}
 * @ARG1{wf *gwf, "Wavefunction to be set"}
 * @ARG2{REAL complex c, "Value"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT inline void grid_wf_constant(wf *gwf, REAL complex c) { 

  cgrid_constant(gwf->grid, c); 
}

/*
 * @FUNC{grid_wf_map, "Map wavefunction"}
 * @DESC{"Map a given function to a wavefunction. The arguments to the user specified function are:
          user parameters (void *; may be NULL) and the current x, y, z coordinates. The function
          must return REAL complex type data"}
 * @ARG1{wf *gwf, "Wavefunction where function will be mapped to"}
 * @ARG2{REAL complex (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Optional argument for passing parameters to func"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT inline void grid_wf_map(wf *gwf, REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg) { 

  cgrid_map(gwf->grid, func, farg); 
}

/*
 * @FUNC{grid_wf_norm, "Wavefunction norm"}
 * @DESC{"Calculate the norm of the given wavefunction"}
 * @ARG1{wf *gwf, "Wavefunction for the calculation"}
 * @RVAL{REAL, "Returns the norm"}
 *
 */

EXPORT inline REAL grid_wf_norm(wf *gwf) { 

  return cgrid_integral_of_square(gwf->grid);
}

/*
 * @FUNC{grid_wf_normalize, "Normalize wavefunction"}
 * @DESC{"Normalize wavefunction (to the value given in gwf-$>$norm)"}
 * @ARG1{wf *gwf, "Wavefunction to be normalized"}
 * @RVAL{REAL, "Returns the normalization constant applied"}
 *
 */

EXPORT inline REAL grid_wf_normalize(wf *gwf) { 

  REAL norm = grid_wf_norm(gwf);

  cgrid_multiply(gwf->grid, SQRT(gwf->norm / norm));
  return norm; 
}

/*
 * @FUNC{grid_wf_overlap, "Overlap between wavefunctions"}
 * @DESC{"Calculate overlap between two wavefunctions"}
 * @ARG1{wf *gwfa, "1st wavefunction"}
 * @ARG2{wf *gwfb, "2nd wavefunction"}
 * @RVAL{REAL complex, "Returns the overlap"}
 *
 */

EXPORT inline REAL complex grid_wf_overlap(wf *gwfa, wf *gwfb) { 

  return cgrid_integral_of_conjugate_product(gwfa->grid, gwfb->grid); 
}

/*
 * @FUNC{grid_wf_merge, "Merge two wavefunctions"}
 * @DESC{"Merge two wavefunctions according to: wf = (1 - alpha(r)) wfr + alpha(r) wfi.
          alpha is computed by grid_wf_absorb(). Used by FFT absorbing BC. 
          Should be no need to call it from user programs"}
 * @ARG1{wf *wf, "Resulting wave functon"}
 * @ARG2{wf *wfr, "Wavefunction (the grid only) from propagating in real time"}
 * @ARG3{wf *wfi, "Wavefunction (the grid only) from propagating in imaginary time"}
 * @RVAL{void, "No return value"}
 *
 */ 

EXPORT void grid_wf_merge(wf *dst, cgrid *wfr, cgrid *wfi) {

  INT i, j, k, ij, ijnz, ny = dst->grid->ny, nz = dst->grid->nz, nxy = dst->grid->nx * ny;
  REAL complex *dval, *rval, *ival;
  REAL alpha;
  INT lx = dst->lx, hx = dst->hx, ly = dst->ly, hy = dst->hy, lz = dst->lz, hz = dst->hz;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_merge(dst, wfr, wfi)) return;
  cuda_remove_block(wfi->value, 1);
  cuda_remove_block(wfr->value, 1);
  cuda_remove_block(dst->grid->value, 0);
#endif
  dval = dst->grid->value;
  rval = wfr->value;
  ival = wfi->value;

#pragma omp parallel for firstprivate(nxy,ny,nz,dval,rval,ival,lx,hx,ly,hy,lz,hz) private(ij,i,j,k,ijnz,alpha) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      alpha = grid_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      dval[ijnz + k] = (1.0 - alpha) * rval[ijnz + k] + alpha * ival[ijnz + k];
    }
  }
}
