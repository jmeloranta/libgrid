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

static curandGenerator_t *gen = NULL;
void *grid_gpu_rand = NULL; // cuRAND states (host)
void *grid_gpu_rand_addr = NULL; // cuRAND states (GPU)

EXPORT INT grid_cuda_random_seed(INT states, INT seed) {

  static size_t prev_len = 0;
  size_t len;

  /* Every block has its own state */
  len = ((size_t) states) * sizeof(curandState);

  if(gen && len > prev_len) {
    curandDestroyGenerator(*gen);
    free(gen);
    cuda_unlock_block(grid_gpu_rand);
    cuda_remove_block(grid_gpu_rand, 0);
    cudaFreeHost(grid_gpu_rand);
    gen = NULL;
  }
  if(gen == NULL) {
    if(cudaMallocHost((void **) &grid_gpu_rand, len) != cudaSuccess) {
      fprintf(stderr, "libgrid(CUDA): Not enough memory in grid_cuda_random_seed().\n");
      abort();
    }
    bzero(grid_gpu_rand, len);
    if(!(cuda_add_block(grid_gpu_rand, len, -1, "GPU RAND", 1))) {
      fprintf(stderr, "libgrid(CUDA): Failed to allocate temporary space on GPU.\n");
      abort();
    }
    grid_gpu_rand_addr = cuda_block_address(grid_gpu_rand);
    cuda_lock_block(grid_gpu_rand);
  }
  grid_cuda_random_seedW(states, seed);
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

  INT states;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  states = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK;

  if(gen == NULL) grid_cuda_random_seed(states, time(0));

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

  INT states;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  states = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK;

  if(gen == NULL) grid_cuda_random_seed(states, time(0));

  rgrid_cuda_random_normalW(cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (uniform distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale added to both real and imaginary parts (REAL; input).
 *
 */

EXPORT INT cgrid_cuda_random_uniform(cgrid *grid, REAL scale) {

  INT states;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  states = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK;

  if(gen == NULL) grid_cuda_random_seed(states, time(0));

  cgrid_cuda_random_uniformW(cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/*
 * Add random numbers to complex grid using cuRAND (normal distribution).
 *
 * grid  = Grid for the operation (cgrid *; input/output).
 * scale = Scale factor. Numbers between -scale, scale added to both real and imaginary parts (REAL; input).
 *
 */

EXPORT INT cgrid_cuda_random_normal(cgrid *grid, REAL scale) {

  INT states;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  states = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK;

  if(gen == NULL) grid_cuda_random_seed(states, time(0));

  cgrid_cuda_random_normalW(cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}
