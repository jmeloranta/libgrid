/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * REAL complex (cgrid) versions.
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

EXPORT char grid_cuda_wf_propagate_potential(wf *gwf, REAL complex (*time)(INT, INT, INT, void *, REAL complex), REAL complex tstep, void *privdata, cgrid *pot) {

  cgrid *grid = gwf->grid;
  struct grid_abs *ab = (struct grid_abs *) privdata;
  REAL amp;
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts;

  if(!ab) {
    amp = 0.0;
    lx = hx = ly = hy = lz = hz = 0;
  } else {
    amp = ab->amp;
    lx = ab->data[0];
    hx = ab->data[1];
    ly = ab->data[2];
    hy = ab->data[3];
    lz = ab->data[4];
  }
  if(time && CIMAG(tstep) != 0.0) {
    fprintf(stderr, "libgrid: Imaginary time for absorbing boundary - forcing real time.\n");
    tstep = CREAL(tstep);
  }
  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, pot->value, pot->grid_len, pot->id, 1) < 0) return -1;

  grid_cuda_wf_propagate_potentialW(cuda_block_address(grid->value), cuda_block_address(pot->value), ts, amp, lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Density.
 *
 */

EXPORT char grid_cuda_wf_density(wf *gwf, rgrid *density) {

  cgrid *grid = gwf->grid;

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, density->value, density->grid_len, density->id, 0) < 0) return -1;
  grid_cuda_wf_densityW(cuda_block_address(grid->value), cuda_block_address(density->value), grid->nx, grid->ny, grid->nz);
  return 0;
}

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
