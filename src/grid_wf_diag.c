/*
 * Routines for diagonalizing using ITP.
 *
 */

#include "grid.h"

/*
 * Calculate energy (E) and the error (dE).
 *
 * gwf       = wavefunction for the operation (wf *).
 * potential = grid containing the potential (cgrid *).
 * workspace = additional workspace required for the operation (cgrid *).
 * error     = error estimate for energy (REAL *).
 * 
 * Returns the energy.
 *
 */

EXPORT REAL grid_wf_energy_and_error(wf *gwf, cgrid *potential, cgrid *workspace, REAL *error) { 

  REAL energy;
  
  /* 
   * energy and its error
   * dE^2 = int dE^2 |psi|^2 dtau = ... = int E_local^2 |psi|^2 dtau - E_avg^2
   * dE = E_local - E_avg
   * E_local psi = H psi
   *
   */

  /* T psi */
  if (gwf->boundary == WF_DIRICHLET_BOUNDARY || gwf->boundary == WF_NEUMANN_BOUNDARY) {
    cgrid_fd_laplace(gwf->grid, workspace);
    cgrid_multiply(workspace, -HBAR*HBAR / (2.0 * gwf->mass));
  } else if (gwf->boundary == WF_PERIODIC_BOUNDARY) {
    cgrid_copy(workspace, gwf->grid);
    cgrid_fft(workspace);
    cgrid_fft_laplace(workspace, workspace);
    cgrid_scaled_inverse_fft(workspace, -gwf->grid->fft_norm * HBAR * HBAR / (2.0 * gwf->mass));
  } else {
    fprintf(stderr, "libgrid: Error in grid_wf_energy_and_error(). Invalid boundary condition.\n");
    abort();
  }
  
  /* H psi */
  cgrid_add_scaled_product(workspace, 1.0, potential, gwf->grid);
  
  /* int E_local^2 |psi|^2 dtau */
  *error = cgrid_integral_of_square(workspace);
  
  /* int E_local |psi|^2 dtau */
  cgrid_conjugate_product(workspace, gwf->grid, workspace);
  energy = CREAL(cgrid_integral(workspace));
  
  /* SQRT( int E_local^2 |psi|^2 dtau - ( int E_local |psi|^2 dtau )^2 ) */
  *error = SQRT(*error - energy * energy);
  
  return energy;
}


/*
 * Project "gwfb" out from "gwfa".
 *
 * gwfa = input wavefunction (wf *).
 * gwfb = this will be projected out from gwfa (wf *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_project_out(wf *gwfa, wf *gwfb) {

  REAL complex overlap = grid_wf_overlap(gwfa, gwfb);

  cgrid_add_scaled(gwfa->grid, -overlap, gwfb->grid);
}

/*
 * Linear transform a set of wavefunctions.
 *
 * gwf       = an array of wavefunctions (wf **).
 * transform = transformation matrix (REAL complex *).
 * states    = number of states (int).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_linear_transform(wf **gwf, REAL complex *transform, INT states) {

  INT p, q, offset;
  INT ij, k, nxy, nz, ijnz;
  REAL complex **value, *tmp;
  
#ifdef USE_CUDA
  for(k = 0; k < states; k++)
    if(cuda_status()) cuda_remove_block(gwf[k]->grid->value, 1);
#endif

  nxy = gwf[0]->grid->nx * gwf[0]->grid->ny;
  nz = gwf[0]->grid->nz;
  
  /* + 16 to prevent "write locks" */
  tmp = (REAL complex *) malloc(((size_t) (omp_get_max_threads() * (states + 16))) * sizeof(REAL complex));
  
  value = (REAL complex **) malloc(((size_t) states) * sizeof(REAL complex *));

  for(p = 0; p < states; p++)
    value[p] = gwf[p]->grid->value;
  
#pragma omp parallel for firstprivate(nz,nxy,value,states,transform,tmp) private(ij,k,p,q,ijnz,offset) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij*nz;
    
    offset = (states + 16) * omp_get_thread_num();
    
    for(k = 0; k < nz; k++) {
      for(p = 0; p < states; p++)
        tmp[offset + p] = 0.0;
      
      for(p = 0; p < states; p++)
        for(q = 0; q < states; q++)
          tmp[offset + p] += transform[p * states + q] * value[q][ijnz + k];
      
      for(p = 0; p < states; p++)
        value[p][ijnz + k] = tmp[offset + p];
    }
  }
  
  free(value);
  free(tmp);
}
