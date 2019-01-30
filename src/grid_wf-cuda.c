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

