/*
 * Routines to find eigenfunctions by the imaginary time propagation method.
 *
 */

#include "grid.h"
#include "private.h"

#ifdef USE_LAPACK

/*
 * Solve eigenfunctions of a Hamiltonian with a linear potential function.
 *
 * gwf            = an array of dimension "states" for storing the resulting eigenfunctions (wf **).
 * states         = number of states requested (INT).
 * virtuals       = number of virtual states for orthogonalization (INT).
 * potential      = potential grid (cgrid *).
 * tau            = initial imaginary time step length (REAL). This will be adjusted to smaller values dynamically.
 *                  If tau is given as negative number, its absolute value will be used and the time step will not
 *                  be adjusted dynamically.
 * threshold      = convergence threshold (dE / E < threshold) (REAL).
 * max_iterations = maximum number of iterations allowed (INT).
 * rtau           = the final (adjusted) imaginary time step (REAL *).
 * riterations    = final number of iterations (INT *).
 * 
 * Return value is the relative error (dE / E).
 *
 */

EXPORT REAL grid_itp_linear(wf **gwf, INT states, INT virtuals, cgrid *potential, REAL tau, REAL threshold, INT max_iterations, REAL *rtau, INT *riterations) {

  INT i, l;
  REAL cfactor = 0.5, max_cfactor = 0.9;
  REAL erel = 0.0, erms_long, erms_short, derms_long, derms_short;
  REAL complex time_step;
  REAL *energy_long = 0, *energy_short = 0;
  REAL *error_long = 0, *error_short = 0;
  REAL error; 

  wf **gwf_short;
  wf **gwf_long = gwf;
  
  cgrid *sq_grad_pot = 0;
  cgrid *workspace  = 0;
  cgrid *workspace2 = 0;
    
  /* if tau is negative, use constant time step */
  if (tau < 0.0) {
    cfactor = 1.0; tau = -tau;
  }
  
  /* allocate another set of wave functions for workspace */
  gwf_short = (wf **) malloc(((size_t) states) * sizeof(wf *));
  for(i = 0; i < states; i++) {
    gwf_short[i] = grid_wf_alloc(gwf[i]->grid->nx, gwf[i]->grid->ny, gwf[i]->grid->nz, gwf[i]->grid->step,
				 gwf[i]->mass, gwf[i]->boundary, gwf[i]->propagator, "ITP gwf");
    
    if (!gwf_short[i]) {
      fprintf(stderr, "libgrid: Error in grid_itp_linear(). Could not allocate memory for workspaces.\n");
      abort();
    }
  }
  
  /* allocate grids for |grad pot|^2 and workspaces */
  sq_grad_pot = cgrid_alloc(potential->nx, potential->ny, potential->nz, potential->step,
			     potential->value_outside, potential->outside_params_ptr, "ITP potential");
  workspace = cgrid_alloc(potential->nx, potential->ny, potential->nz, potential->step,
			   potential->value_outside, potential->outside_params_ptr, "ITP workspace");
  workspace2 = cgrid_alloc(potential->nx, potential->ny, potential->nz, potential->step,
			    potential->value_outside, potential->outside_params_ptr, "ITP workspace2");
  
  if (!sq_grad_pot || !workspace || !workspace2) {
    fprintf(stderr, "libgrid: Error in grid_itp_linear(). Could not allocate memory for workspaces.\n");
    abort();
  }
  
  energy_long = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  error_long = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  energy_short = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  error_short = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  if (!energy_long || !energy_short || !error_short || !error_long) {
    fprintf(stderr, "libgrid: Error in grid_itp_linear(). Could not allocate memory for workspaces.\n");
    abort();
  }
  
  /* copy wave functions */
  for(i = 0; i < states; i++)
    cgrid_copy(gwf_short[i]->grid, gwf_long[i]->grid);
  
  /* |grad potential|^2 */
  grid_wf_square_of_potential_gradient(sq_grad_pot, potential, workspace, workspace2);
  
  /* iteration loop */
  for(l = 0; l < max_iterations; l++) {
    
    /* propagate t = - i tau */
    time_step = -I * tau;
    for(i = 0; i < states; i++)
      grid_wf_propagate(gwf_long[i], potential, sq_grad_pot, time_step, workspace);
    grid_wf_diagonalize(gwf_long, states);
    
    for(i = 0; i < states; i++)
      energy_long[i] = grid_wf_energy_and_error(gwf_long[i], potential, workspace, &error_long[i]); 
    
    /* if constant time step, skip time step adjusting */
    if (cfactor > max_cfactor) {
#if __DEBUG__
      fprintf(stderr, "El %d ", l);
      for(i = 0; i < states; i++)
        fprintf(stderr, "%20.15le ", energy_long[i]);
      fprintf(stderr, "\n");
      fprintf(stderr, "el %d ", l);
      for(i = 0; i < states; i++)
        fprintf(stderr, "%20.15le ", error_long[i]);
      fprintf(stderr, "\n");
#endif      
      continue;
    }
    
    /* propagate t = - i c tau */
    time_step = -I * cfactor * tau;
    for(i = 0; i < states; i++)
      grid_wf_propagate(gwf_short[i], potential, sq_grad_pot, time_step, workspace);
    
    grid_wf_diagonalize(gwf_short, states);
    
    for( i = 0; i < states; i++ ) 
      energy_short[i] = grid_wf_energy_and_error(gwf_short[i], potential, workspace, &error_short[i]);
    
    /* relative error, dE / E */
    erel = 0;
    for(i = 0; i < states - virtuals; i++)
      erel += 2.0 * (error_long[i] / energy_long[i]) * (error_long[i] / energy_long[i]);
    erel = SQRT(erel);
    
    /* check convergence */
    if (erel < threshold) break;
    
    /* rms of absolute energy and error */
    erms_long = erms_short = 0.0;
    derms_long = derms_short = 0.0;
    for(i = 0; i < states - virtuals; i++) {
      erms_long += energy_long[i]  * energy_long[i];
      erms_short += energy_short[i] * energy_short[i];
      
      derms_long  += error_long[i]  * error_long[i];
      derms_short += error_short[i] * error_short[i];
    }
    erms_long = SQRT(erms_long / (REAL) (states - virtuals));
    erms_short = SQRT(erms_short / (REAL) (states - virtuals));
    derms_long = SQRT(derms_long / (REAL) (states - virtuals));
    derms_short = SQRT(derms_short / (REAL) (states - virtuals));
    
    /* if long time step gives better energy or error, use it and corresponding wave function for next iteration */
    if (erms_long < erms_short || derms_long < derms_short) {
      for(i = 0; i < states; i++)
        cgrid_copy(gwf_short[i]->grid, gwf_long[i]->grid);
      
      /* try smaller time step reduce */
      cfactor = SQRT(cfactor);
      if (cfactor > max_cfactor) cfactor = max_cfactor;
    }
    /* else use short time step and corresponing wave function */
    else {
      for(i = 0; i < states; i++)
        cgrid_copy(gwf_long[i]->grid, gwf_short[i]->grid);
      
      /* try shorter time step */
      tau = tau * cfactor;
      /* and larger time step reduce */
      cfactor *= cfactor;
    }
    
#if __DEBUG__
    fprintf(stderr, "T    %d  %lf\n", l, tau);
    fprintf(stderr, "dE   %d  %le\n", l, erel);
    
    fprintf(stderr, "El   %d ", l);
    for(i = 0; i < states; i++)
      fprintf(stderr, "%20.15le ", energy_long[i]);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Errl %d ", l);
    for(i = 0; i < states; i++)
      fprintf(stderr, "%20.15le ", error_long[i]);
    fprintf(stderr, "\n");
#endif
  }
#if __DEBUG__
  fprintf(stderr, "\n");
#endif
  
  /* free workspaces */
  for(i = 0; i < states; i++)
    grid_wf_free(gwf_short[i]);
  free(gwf_short);
  
  cgrid_free(sq_grad_pot);
  cgrid_free(workspace);
  cgrid_free(workspace2);
  
  free(energy_long);
  free(error_long);
  free(energy_short);
  free(error_short);

  error = erel;
  *rtau = tau;
  *riterations = l;
  
  return error;
}

/*
 * Solve eigenfunctions of a Hamiltonian with a nonlinear potential function.
 *
 * gwf                  = an array of dimension "states" for storing the resulting eigenfunctions (wf **).
 * states               = number of states requested (INT).
 * virtuals             = number of virtual states for orthogonalization (INT).
 * calculate_potentials = nonlinear potential, which takes the current grid etc. as argument (void (*)(cgrid **, void *, wf **, INT)).
 *                        Note that this is a pointer to the potential function.
 * tau                  = initial imaginary time step length (REAL). This will be adjusted to smaller values dynamically.
 *                        If tau is given as negative number, its absolute value will be used and the time step will not
 *                        be adjusted dynamically.
 * threshold            = convergence threshold (dE / E < threshold) (REAL).
 * max_iterations       = maximum number of iterations allowed (INT).
 * rtau                 = the final (adjusted) imaginary time step (REAL *).
 * riterations          = final number of iterations (INT *).
 * 
 * Return value is the relative error (dE / E).
 *
 */

EXPORT REAL grid_itp_nonlinear(wf **gwf, INT states, INT virtuals, void (*calculate_potentials)(cgrid **potential, void *arg, wf **gwf, INT states), void *arg, REAL tau, REAL threshold, INT max_iterations, REAL *rtau, INT *riterations) {

  INT i, l;
  REAL erel = 0.0, erms_long, erms_short, derms_long, derms_short;
  REAL cfactor = 0.5, max_cfactor = 0.9;
  REAL complex time_step;
  REAL error;
  
  wf **gwf_short;
  wf **gwf_long = gwf;
  
  cgrid **potential   = 0;
  cgrid **sq_grad_pot = 0;
  cgrid *workspace  = 0;
  cgrid *workspace2 = 0;
  
  REAL *energy_long = 0, *energy_short = 0;
  REAL *error_long = 0, *error_short = 0;
  
  /* if tau is negative, use constant time step */
  if (tau < 0.0) {
    cfactor = 1.0; tau = -tau;
  }
  
  /* allocate grids for another set of wave functions, potential, |grad pot|^2 and workspaces */
  gwf_short = (wf **) malloc(((size_t) states) * sizeof(wf *));
  potential = (cgrid **) malloc(((size_t) states) * sizeof(cgrid *));
  sq_grad_pot = (cgrid **) malloc(((size_t) states) * sizeof(cgrid *));
  
  for(i = 0; i < states; i++) {
    gwf_short[i] = grid_wf_alloc(gwf[i]->grid->nx, gwf[i]->grid->ny, gwf[i]->grid->nz, gwf[i]->grid->step,
				   gwf[i]->mass, gwf[i]->boundary, gwf[i]->propagator, "ITP gwf");
    potential[i] = cgrid_alloc(gwf[i]->grid->nx, gwf[i]->grid->ny, gwf[i]->grid->nz, gwf[i]->grid->step,
				gwf[i]->grid->value_outside, gwf[i]->grid->outside_params_ptr, "ITP potential");
    sq_grad_pot[i] = cgrid_alloc(potential[i]->nx, potential[i]->ny, potential[i]->nz, potential[i]->step,
				  potential[i]->value_outside, potential[i]->outside_params_ptr, "ITP sq grad");
  }
  workspace = cgrid_alloc(potential[0]->nx, potential[0]->ny, potential[0]->nz, potential[0]->step,
			   potential[0]->value_outside, potential[0]->outside_params_ptr, "ITP workspace");
  workspace2 = cgrid_alloc(potential[0]->nx, potential[0]->ny, potential[0]->nz, potential[0]->step,
			    potential[0]->value_outside, potential[0]->outside_params_ptr, "ITP workspace2");
  
  for(i = 0; i < states; i++)
    if (!gwf_short[i] || !potential[i] || !sq_grad_pot[i]) {
      fprintf(stderr, "libgrid: Error in grid_itp_nonlinear(). Could not allocate memory for workspaces.\n");
      abort();
    }
  
  if (!workspace || !workspace2) {
    fprintf(stderr, "libgrid: Error in grid_itp_nonlinear(). Could not allocate memory for workspaces.\n");
    abort();
  }
  
  energy_long = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  error_long = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  energy_short = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  error_short = (REAL *) malloc(((size_t) states) * sizeof(REAL));
  if (!energy_long || !energy_short || !error_short || !error_long) {
    fprintf(stderr, "libgrid: Error in grid_itp_nonlinear(). Could not allocate memory for workspaces.\n");
    abort();
  }
  
  /* copy wave functions */
  for(i = 0; i < states; i++)
    cgrid_copy(gwf_short[i]->grid, gwf_long[i]->grid);
  
  /* iteration loop */
  for(l = 0; l < max_iterations; l++) {
    
    /* calculate potentials */
    (*calculate_potentials)(potential, arg, gwf_long, states);
    
    /* |grad potential|^2 */
    for(i = 0; i < states; i++)
      grid_wf_square_of_potential_gradient(sq_grad_pot[i], potential[i], workspace, workspace2);
    
    /* propagate t = - i tau */
    time_step = -I * tau;
    for(i = 0; i < states; i++)
      grid_wf_propagate(gwf_long[i], potential[i], sq_grad_pot[i], time_step, workspace);
    
    grid_wf_diagonalize(gwf_long, states);
    
    for(i = 0; i < states; i++)
      energy_long[i] = grid_wf_energy_and_error(gwf_long[i], potential[i], workspace, &error_long[i]);
    
    /* if constant time step, skip step adjusting */
    if (cfactor > max_cfactor) {
#if __DEBUG__
      fprintf(stderr,"E  %4d ", l);
      for(i = 0; i < states; i++)
        fprintf(stderr," %20.15lf %20.15lf ", energy_long[i], error_long[i]);
      fprintf(stderr,"\n");
#endif
      continue;
    }
    
    /* propagate t = - i c tau */
    time_step = -I * cfactor * tau;
    for(i = 0; i < states; i++)
      grid_wf_propagate(gwf_short[i], potential[i], sq_grad_pot[i], time_step, workspace);
    
    grid_wf_diagonalize(gwf_short, states);
    
    for(i = 0; i < states; i++)
      energy_short[i] = grid_wf_energy_and_error(gwf_short[i], potential[i], workspace, &error_short[i]);
    
    /* max relative error, dE^2 / E^2 */
    erel = 0;
    for(i = 0; i < states - virtuals; i++)
      erel += 2.0 * (error_long[i] / energy_long[i]) * (error_long[i] / energy_long[i]);
    
    /* check convergence */
    if ( SQRT(erel) < threshold ) break;
   
    /* rms of energy absolute error */
    erms_long = erms_short = 0;
    derms_long = derms_short = 0;
    for( i = 0; i < states - virtuals; i++ ) {
      erms_long += energy_long[i] * energy_long[i];
      erms_short += energy_short[i] * energy_short[i];
      derms_long += error_long[i] * error_long[i];
      derms_short += error_short[i] * error_short[i];
    }
    erms_long = SQRT(erms_long / (REAL) (states - virtuals));
    erms_short = SQRT(erms_short / (REAL) (states - virtuals));
    derms_long = SQRT(derms_long / (REAL) (states - virtuals));
    derms_short = SQRT(derms_short / (REAL) (states - virtuals));
    
    /* if short time step gives better energy AND error, use it and corresponding wave function for next iteration */
    if (erms_short < erms_long && derms_short < derms_long) {
      for(i = 0; i < states; i++)
        cgrid_copy(gwf_long[i]->grid, gwf_short[i]->grid);
      
      /* try shorter time step */
      tau = tau * cfactor;
      /* and larger time step reduce */
      cfactor *= cfactor;
    }
    /* else use long time step and corresponing wave function */
    else {
      for(i = 0; i < states; i++)
        cgrid_copy(gwf_short[i]->grid, gwf_long[i]->grid);
      
      /* try smaller time step reduce */
      cfactor = SQRT(cfactor);
      if (cfactor > max_cfactor) cfactor = max_cfactor;
    }
    
#if __DEBUG__
    fprintf(stderr,"T   %4d %12.4lf %12.4le\n", l, tau, derms_long);
    fprintf(stderr,"El  %4d ", l);
    for(i = 0; i < states; i++)
      fprintf(stderr," %20.15lf %20.15lf ", energy_long[i], error_long[i]);
    fprintf(stderr,"\n");
    fprintf(stderr,"Es  %4d ", l);
    for(i = 0; i < states; i++)
      fprintf(stderr," %20.15lf %20.15lf ", energy_short[i], error_short[i]);
    fprintf(stderr,"\n");
#endif
  }
  
  /* free workspaces */
  for(i = 0; i < states; i++) {
    grid_wf_free(gwf_short[i]);
    cgrid_free(potential[i]);
    cgrid_free(sq_grad_pot[i]);
  }
  free(gwf_short);
  free(potential);
  free(sq_grad_pot);
  
  cgrid_free(workspace);
  cgrid_free(workspace2);
  
  free(energy_long);
  free(error_long);
  free(energy_short);
  free(error_short);

  error = erel;
  *rtau = tau;
  *riterations = l;
  
  return error;
}

#endif
