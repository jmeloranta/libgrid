/*
 * cuRAND wrapper functions.
 *
 * We allocate separate states for each thread. There are typically around 8 * 8 * 8 = 512 threads running at the same time.
 * So we reserve that many state variables. Ideally, each block/thread should have their own but this is too much.
 *
 */

#include "grid.h"
#include <time.h>
#include <strings.h>

extern size_t rand_prev_len;

/*
 * Initialize the random seed.
 *
 */

EXPORT INT grid_cuda_random_seed(INT nx, INT ny, INT nz, INT seed) {

  grid_cuda_random_seedW(nx, ny, nz, seed);

  return 0;
}

/*
 * Add random numbers to real grid using cuRAND (uniform distribution).
 *
 * grid  = Grid for the operation (rgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale (REAL; input).
 *
 */

EXPORT INT rgrid_cuda_random_uniform(rgrid *grid, REAL scale) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  rgrid_cuda_random_uniformW(cuda_find_block(grid->value), scale, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to real grid using cuRAND (normal distribution).
 *
 * grid  = Grid for the operation (rgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale (REAL; input).
 *
 */

EXPORT INT rgrid_cuda_random_normal(rgrid *grid, REAL scale) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  rgrid_cuda_random_normalW(cuda_find_block(grid->value), scale, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (uniform distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale for both real and imaginary parts (REAL complex; input).
 *
 */

EXPORT INT cgrid_cuda_random_uniform(cgrid *grid, REAL complex scale) {

  CUCOMPLEX sc;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  sc.x = CREAL(scale);
  sc.y = CIMAG(scale);
  cgrid_cuda_random_uniformW(cuda_find_block(grid->value), sc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (uniform distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor (REAL; input).
 *
 */

EXPORT INT cgrid_cuda_random_uniform_sp(cgrid *grid, REAL scale) {

  CUREAL sc;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  sc = scale;

  cgrid_cuda_random_uniform_spW(cuda_find_block(grid->value), sc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (normal distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale added to both real and imaginary parts (REAL; input).
 *
 */

EXPORT INT cgrid_cuda_random_normal(cgrid *grid, REAL complex scale) {

  CUCOMPLEX sc;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  sc.x = CREAL(scale);
  sc.y = CIMAG(scale);
  cgrid_cuda_random_normalW(cuda_find_block(grid->value), sc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (normal distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor (REAL; input).
 *
 */

EXPORT INT cgrid_cuda_random_normal_sp(cgrid *grid, REAL scale) {

  CUREAL sc;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  if(rand_prev_len < grid->nx * grid->ny)
    grid_cuda_random_seed(grid->nx, grid->ny, grid->nz, time(0));

  sc = scale;

  cgrid_cuda_random_normal_spW(cuda_find_block(grid->value), sc, grid->nx, grid->ny, grid->nz);

  return 0;
}
