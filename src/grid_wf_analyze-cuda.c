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
 * Calculate the probability flux x component.
 * 
 */

EXPORT char grid_cuda_wf_fd_probability_flux_x(wf *gwf, rgrid *flux_x, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(grid->host_lock || flux_x->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(flux_x->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, flux_x->value, flux_x->grid_len, flux_x->cufft_handle_r2c, flux_x->id, 0) < 0)
    return -1;

  grid_cuda_wf_fd_probability_flux_xW(cuda_block_address(grid->value), cuda_block_address(flux_x->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_x->nz2);

  return 0;
}

/*
 * Calculate the probability flux y component.
 * 
 */

EXPORT char grid_cuda_wf_fd_probability_flux_y(wf *gwf, rgrid *flux_y, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(grid->host_lock || flux_y->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(flux_y->value, 0);
    return -1;
  }

  if(grid->host_lock || flux_y->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(flux_y->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, flux_y->value, flux_y->grid_len, flux_y->cufft_handle_r2c, flux_y->id, 0) < 0)
    return -1;

  grid_cuda_wf_fd_probability_flux_yW(cuda_block_address(grid->value), cuda_block_address(flux_y->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_y->nz2);

  return 0;
}

/*
 * Calculate the probability flux z component.
 * 
 */

EXPORT char grid_cuda_wf_fd_probability_flux_z(wf *gwf, rgrid *flux_z, REAL inv_delta) {

  cgrid *grid = gwf->grid;

  if(grid->host_lock || flux_z->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(flux_z->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, flux_z->value, flux_z->grid_len, flux_z->cufft_handle_r2c, flux_z->id, 0) < 0) 
    return -1;

  grid_cuda_wf_fd_probability_flux_zW(cuda_block_address(grid->value), cuda_block_address(flux_z->value), inv_delta, grid->nx, grid->ny, grid->nz, flux_z->nz2);

  return 0;
}

