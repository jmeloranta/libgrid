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
 */

EXPORT char grid_cuda_wf_propagate_potential(wf *gwf, REAL complex tstep, cgrid *pot, REAL cons) {

  cgrid *grid = gwf->grid;
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts;

  if(grid->host_lock || pot->host_lock) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(pot->value, 1);
    return -1;
  }

  if(!gwf->ts_func || gwf->ts_func != grid_wf_absorb || gwf->propagator < WF_2ND_ORDER_CN) {
    lx = hx = ly = hy = lz = hz = 0;
  } else {
    lx = gwf->lx;
    hx = gwf->hx;
    ly = gwf->ly;
    hy = gwf->hy;
    lz = gwf->lz;
    hz = gwf->hz;
  }

  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, pot->value, pot->grid_len, pot->cufft_handle, pot->id, 1) < 0) return -1;

  grid_cuda_wf_propagate_potentialW(cuda_find_block(grid->value), cuda_find_block(pot->value), ts, cons, lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);

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

  grid_cuda_wf_densityW(cuda_find_block(grid->value), cuda_find_block(density->value), grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Wave function merging.
 *
 */

EXPORT char grid_cuda_wf_merge(wf *dst, cgrid *cwfr, cgrid *cwfi) {

  cgrid *cdst = dst->grid;
  INT lx = dst->lx, hx = dst->hx, ly = dst->ly, hy = dst->hy, lz = dst->lz, hz = dst->hz;

  if(cwfr->host_lock || cwfi->host_lock || cdst->host_lock) {
    cuda_remove_block(cwfr->value, 1);
    cuda_remove_block(cwfi->value, 1);
    cuda_remove_block(cdst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(cwfr->value, cwfr->grid_len, cwfr->cufft_handle, cwfr->id, 1, cwfi->value, cwfi->grid_len, cwfi->cufft_handle, cwfi->id, 1, cdst->value, cdst->grid_len, cdst->cufft_handle, cdst->id, 0) < 0)
    return -1;

  grid_cuda_wf_mergeW(cuda_find_block(cdst->value), cuda_find_block(cwfr->value), cuda_find_block(cwfi->value), lx, hx, ly, hy, lz, hz, cdst->nx, cdst->ny, cdst->nz);

  return 0;
}
