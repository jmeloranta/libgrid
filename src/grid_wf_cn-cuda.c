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
 * Auxiliary routine for propagating kinetic energy using Crank-Nicolson along x.
 *
 */

EXPORT char grid_cuda_wf_propagate_kinetic_cn_x(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  cgrid *grid = gwf->grid;
  struct grid_abs *ab = &(gwf->abs_data);
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts;
  
  if(grid->host_lock || workspace->host_lock || workspace2->host_lock || (workspace3 && workspace3->host_lock)) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(workspace->value, 0);
    cuda_remove_block(workspace2->value, 0);
    cuda_remove_block(workspace3->value, 0);
    return -1;
  }

  if(!gwf->ts_func || gwf->ts_func != grid_wf_absorb) {
    lx = hx = ly = hy = lz = hz = 0;
  } else {
    lx = ab->data[0];
    hx = ab->data[1];
    ly = ab->data[2];
    hy = ab->data[3];
    lz = ab->data[4];
    hz = ab->data[5];
  }
  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);
  
  if(workspace3) {
    if(cuda_four_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0, workspace3->value, workspace3->grid_len, workspace3->cufft_handle, "CN workspace 3", 0) < 0)
       return -1;
  } else {
    if(cuda_three_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0) < 0)
       return -1;
  }

  grid_cuda_wf_propagate_kinetic_cn_xW(grid->nx, grid->ny, grid->nz, ts, cuda_block_address(grid->value), gwf->boundary, gwf->mass, grid->step, grid->kx0, grid->omega, grid->y0, cuda_block_address(workspace->value), cuda_block_address(workspace2->value), workspace3?cuda_block_address(workspace3->value):NULL, lx, hx, ly, hy, lz, hz);

  return 0;
}

/*
 * Auxiliary routine for propagating kinetic energy using Crank-Nicolson along y.
 *
 */

EXPORT char grid_cuda_wf_propagate_kinetic_cn_y(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  cgrid *grid = gwf->grid;
  struct grid_abs *ab = &(gwf->abs_data);
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts;
   
  if(grid->host_lock || workspace->host_lock || workspace2->host_lock || (workspace3 && workspace3->host_lock)) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(workspace->value, 0);
    cuda_remove_block(workspace2->value, 0);
    cuda_remove_block(workspace3->value, 0);
    return -1;
  }

  if(!gwf->ts_func || gwf->ts_func != grid_wf_absorb) {
    lx = hx = ly = hy = lz = hz = 0;
  } else {
    lx = ab->data[0];
    hx = ab->data[1];
    ly = ab->data[2];
    hy = ab->data[3];
    lz = ab->data[4];
    hz = ab->data[5];
  }
  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);
  
  if(workspace3) {
    if(cuda_four_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0, workspace3->value, workspace3->grid_len, workspace3->cufft_handle, "CN workspace 3", 0) < 0)
       return -1;
  } else {
    if(cuda_three_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0) < 0)
       return -1;
  }

  grid_cuda_wf_propagate_kinetic_cn_yW(grid->nx, grid->ny, grid->nz, ts, cuda_block_address(grid->value), gwf->boundary, gwf->mass, grid->step, grid->ky0, grid->omega, grid->x0, cuda_block_address(workspace->value), cuda_block_address(workspace2->value), workspace3?cuda_block_address(workspace3->value):NULL, lx, hx, ly, hy, lz, hz);

  return 0;
}

/*
 * Auxiliary routine for propagating kinetic energy using Crank-Nicolson along z.
 *
 */

EXPORT char grid_cuda_wf_propagate_kinetic_cn_z(wf *gwf, REAL complex tstep, cgrid *workspace, cgrid *workspace2, cgrid *workspace3) {

  cgrid *grid = gwf->grid;
  struct grid_abs *ab = &(gwf->abs_data);
  INT lx, hx, ly, hy, lz, hz;
  CUCOMPLEX ts;
  
  if(grid->host_lock || workspace->host_lock || workspace2->host_lock || (workspace3 && workspace3->host_lock)) {
    cuda_remove_block(grid->value, 1);
    cuda_remove_block(workspace->value, 0);
    cuda_remove_block(workspace2->value, 0);
    cuda_remove_block(workspace3->value, 0);
    return -1;
  }

  if(!gwf->ts_func || gwf->ts_func != grid_wf_absorb) {
    lx = hx = ly = hy = lz = hz = 0;
  } else {
    lx = ab->data[0];
    hx = ab->data[1];
    ly = ab->data[2];
    hy = ab->data[3];
    lz = ab->data[4];
    hz = ab->data[5];
  }
  ts.x = CREAL(tstep);
  ts.y = CIMAG(tstep);
  
  if(workspace3) {
    if(cuda_four_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0, workspace3->value, workspace3->grid_len, workspace3->cufft_handle, "CN workspace 3", 0) < 0)
       return -1;
  } else {
    if(cuda_three_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1, workspace->value, workspace->grid_len, workspace->cufft_handle, "CN workspace 1", 0,
                              workspace2->value, workspace2->grid_len, workspace2->cufft_handle, "CN workspace 2", 0) < 0)
       return -1;
  }

  grid_cuda_wf_propagate_kinetic_cn_zW(grid->nx, grid->ny, grid->nz, ts, cuda_block_address(grid->value), gwf->boundary, gwf->mass, grid->step, grid->kz0, cuda_block_address(workspace->value), cuda_block_address(workspace2->value), workspace3?cuda_block_address(workspace3->value):NULL, lx, hx, ly, hy, lz, hz);

  return 0;
}
