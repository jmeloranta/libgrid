/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * Wave function routines.
 *
 */

#include "grid.h"
#include <cufft.h>

/*
 * Propagate potential energy with absorbing boundaries.
 * 
 * Note: device code in grid_wf_cn-cuda2.cu (absorbing boundary function).
 *
 */

EXPORT char grid_cuda_wf_propagate_potential(wf *gwf, REAL complex tstep, cgrid *pot, REAL cons) {

  cgrid *grid = gwf->grid;
  struct grid_abs *ab = &(gwf->abs_data);
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts, amp;
  char add_abs = 0;
  CUREAL rho0;

  if(grid->host_lock || pot->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(pot->value, 1);
    return -1;
  }

  if(!gwf->ts_func || gwf->ts_func != grid_wf_absorb) {
    lx = hx = ly = hy = lz = hz = 0;
    amp.x = amp.y = 0.0;
    rho0 = 0.0;
  } else {
    lx = ab->data[0];
    hx = ab->data[1];
    ly = ab->data[2];
    hy = ab->data[3];
    lz = ab->data[4];
    hz = ab->data[5];
    amp.x = CREAL(ab->amp);
    amp.y = CIMAG(ab->amp);
    rho0 = ab->rho0;
  }

  if(gwf->propagator < WF_2ND_ORDER_CN && gwf->ts_func != NULL) add_abs = 1;   /* abs potential for FFT */

  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, pot->value, pot->grid_len, pot->cufft_handle, pot->id, 1) < 0) return -1;

  grid_cuda_wf_propagate_potentialW(cuda_block_address(grid->value), cuda_block_address(pot->value), ts, add_abs, amp, rho0, cons, lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Density.
 *
 */

EXPORT char grid_cuda_wf_density(wf *gwf, rgrid *density) {

  cgrid *grid = gwf->grid;

  if(grid->host_lock || density->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(density->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, density->value, density->grid_len, density->cufft_handle_r2c, density->id, 0) < 0)
    return -1;

  grid_cuda_wf_densityW(cuda_block_address(grid->value), cuda_block_address(density->value), grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Complex absorbing potential.
 *
 */

EXPORT char grid_cuda_wf_absorb_potential(wf *gwf, cgrid *pot_grid, REAL amp, REAL rho0) {

  cgrid *gwf_grid = gwf->grid;

  if(gwf_grid->host_lock || pot_grid->host_lock) {
    cuda_remove_block(gwf_grid->value, 1);
    cuda_remove_block(pot_grid->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(gwf_grid->value, gwf_grid->grid_len, gwf->grid->cufft_handle, gwf_grid->id, 1, pot_grid->value, pot_grid->grid_len, pot_grid->cufft_handle, pot_grid->id, 1) < 0)
    return -1;

  grid_cuda_wf_absorb_potentialW(cuda_block_address(gwf_grid->value), cuda_block_address(pot_grid->value), amp, rho0, 
    gwf->abs_data.data[0], gwf->abs_data.data[1], gwf->abs_data.data[2], gwf->abs_data.data[3], gwf->abs_data.data[4], gwf->abs_data.data[5],
    pot_grid->nx, pot_grid->ny, pot_grid->nz);

  return 0;
}

