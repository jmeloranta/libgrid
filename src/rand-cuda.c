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
//#include <curand_kernel.h>

/* including curand_kernel.h does not work but we need the size of this data type! */
struct curandState {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
typedef struct curandState curandState;
/* End copy from curand_kernel.h */

char grid_gpu_rand_holder;  // Place holder
void *grid_gpu_rand = NULL; // cuRAND states (host)
cudaXtDesc *grid_gpu_rand_addr = NULL; // cuRAND states (GPU)
static size_t prev_len = 0;

#define STATES (CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK)

/*
 * Initialize the random seed.
 *
 */

EXPORT INT grid_cuda_random_seed(INT seed) {

  size_t len;

  /* Every block has its own state */
  len = ((size_t) STATES) * sizeof(curandState);

  if(prev_len < len) {
    if(grid_gpu_rand) {
      cuda_unlock_block(grid_gpu_rand);
      cuda_remove_block(grid_gpu_rand, 0);
    }
    prev_len = len;
    grid_gpu_rand = (void *) &grid_gpu_rand_holder;
    if(!(cuda_add_block(grid_gpu_rand, len, -1, "GPU RAND", 0))) {
      fprintf(stderr, "libgrid(CUDA): Failed to allocate temporary space on GPU.\n");
      abort();
    }
    grid_gpu_rand_addr = (cuda_block_address(grid_gpu_rand))->gpu_info->descriptor;
    cuda_lock_block(grid_gpu_rand);
  }

  grid_cuda_random_seedW(STATES, seed);

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

  if(!prev_len)
    grid_cuda_random_seed(time(0));

  rgrid_cuda_random_uniformW(cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

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

  if(!prev_len)
    grid_cuda_random_seed(time(0));

  rgrid_cuda_random_normalW(cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

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

  if(!prev_len) 
    grid_cuda_random_seed(time(0));

  sc.x = CREAL(scale);
  sc.y = CIMAG(scale);
  cgrid_cuda_random_uniformW(cuda_block_address(grid->value), sc, grid->nx, grid->ny, grid->nz);

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

  if(!prev_len) 
    grid_cuda_random_seed(time(0));

  sc.x = CREAL(scale);
  sc.y = CIMAG(scale);
  cgrid_cuda_random_normalW(cuda_block_address(grid->value), sc, grid->nx, grid->ny, grid->nz);

  return 0;
}
