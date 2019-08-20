/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * Wave function analysis routines.
 *
 */

#include "grid.h"

/*
 * Calculate velocity x component.
 *
 */

EXPORT char grid_cuda_wf_velocity_x(wf *gwf, rgrid *vx, REAL inv_delta, REAL cutoff) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, vx->value, vx->grid_len, vx->id, 0) < 0) return -1;

  grid_cuda_wf_velocity_xW(cuda_block_address(grid->value), cuda_block_address(vx->value), inv_delta, cutoff, grid->nx, grid->ny, grid->nz, vx->nz2);
  return 0;
}

/*
 * Calculate velocity y component.
 *
 */

EXPORT char grid_cuda_wf_velocity_y(wf *gwf, rgrid *vy, REAL inv_delta, REAL cutoff) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, vy->value, vy->grid_len, vy->id, 0) < 0) return -1;

  grid_cuda_wf_velocity_yW(cuda_block_address(grid->value), cuda_block_address(vy->value), inv_delta, cutoff, grid->nx, grid->ny, grid->nz, vy->nz2);
  return 0;
}

/*
 * Calculate velocity z component.
 *
 */

EXPORT char grid_cuda_wf_velocity_z(wf *gwf, rgrid *vz, REAL inv_delta, REAL cutoff) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, vz->value, vz->grid_len, vz->id, 0) < 0) return -1;

  grid_cuda_wf_velocity_zW(cuda_block_address(grid->value), cuda_block_address(vz->value), inv_delta, cutoff, grid->nx, grid->ny, grid->nz, vz->nz2);
  return 0;
}

/*
 * Setup for FFT-based evaluation of velocity.
 *
 */

EXPORT char grid_cuda_wf_fft_velocity_setup(wf *gwf, rgrid *veloc, REAL c) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, veloc->value, veloc->grid_len, veloc->id, 0) < 0) return -1;

  grid_cuda_wf_fft_velocity_setupW(cuda_block_address(grid->value), cuda_block_address(veloc->value), c, grid->nx, grid->ny, grid->nz, veloc->nz2);

  return 0;
}

/*
 * Calculate the probability flux x component.
 * 
 */

EXPORT char grid_cuda_wf_probability_flux_x(wf *gwf, rgrid *flux_x, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, flux_x->value, flux_x->grid_len, flux_x->id, 0) < 0) return -1;

  grid_cuda_wf_probability_flux_xW(cuda_block_address(grid->value), cuda_block_address(flux_x->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_x->nz2);
  return 0;
}

/*
 * Calculate the probability flux y component.
 * 
 */

EXPORT char grid_cuda_wf_probability_flux_y(wf *gwf, rgrid *flux_y, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, flux_y->value, flux_y->grid_len, flux_y->id, 0) < 0) return -1;

  grid_cuda_wf_probability_flux_yW(cuda_block_address(grid->value), cuda_block_address(flux_y->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_y->nz2);
  return 0;
}

/*
 * Calculate the probability flux z component.
 * 
 */

EXPORT char grid_cuda_wf_probability_flux_z(wf *gwf, rgrid *flux_z, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, flux_z->value, flux_z->grid_len, flux_z->id, 0) < 0) return -1;

  grid_cuda_wf_probability_flux_zW(cuda_block_address(grid->value), cuda_block_address(flux_z->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_z->nz2);
  return 0;
}

/*
 * Calculate angular momentum <L_x>. Integration done at higher level.
 *
 */

EXPORT char grid_cuda_wf_lx(wf *gwf, rgrid *workspace, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, workspace->value, workspace->grid_len, workspace->id, 0) < 0) return -1;

  grid_cuda_wf_lxW(cuda_block_address(grid->value), cuda_block_address(workspace->value), inv_delta, workspace->nx, workspace->ny, workspace->nz, workspace->nz2, grid->y0, grid->z0, grid->step);
  return 0;
}

/*
 * Calculate angular momentum <L_y>. Integration done at higher level.
 *
 */

EXPORT char grid_cuda_wf_ly(wf *gwf, rgrid *workspace, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, workspace->value, workspace->grid_len, workspace->id, 0) < 0) return -1;

  grid_cuda_wf_lyW(cuda_block_address(grid->value), cuda_block_address(workspace->value), inv_delta, workspace->nx, workspace->ny, workspace->nz, workspace->nz2, grid->x0, grid->z0, grid->step);
  return 0;
}

/*
 * Calculate angular momentum <L_z>. Integration done at higher level.
 *
 */

EXPORT char grid_cuda_wf_lz(wf *gwf, rgrid *workspace, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, workspace->value, workspace->grid_len, workspace->id, 0) < 0) return -1;

  grid_cuda_wf_lzW(cuda_block_address(grid->value), cuda_block_address(workspace->value), inv_delta, workspace->nx, workspace->ny, workspace->nz, workspace->nz2, grid->x0, grid->y0, grid->step);
  return 0;
}
