/*
 * cuRAND wrapper functions.
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

EXPORT INT grid_cuda_random_seed(INT states, INT seed, INT nx, INT ny, INT nz) {

  static size_t prev_len = 0;
  size_t len;

  /* Every block has its own state */
  len = ((size_t) states) * sizeof(curandState);

  if(len > prev_len) {
    curandDestroyGenerator(*gen);
    free(gen);
    cuda_unlock_block(grid_gpu_rand);
    cuda_remove_block(grid_gpu_rand, 0);
    gen = NULL;
  }
  if(gen == NULL) {
#ifdef SINGLE_PREC
    if(!(grid_gpu_rand = (void *) fftwf_malloc(len)) {
#else
    if(!(grid_gpu_rand = (void *) fftw_malloc(len))) {
#endif
      fprintf(stderr, "libgrid(CUDA): Not enough memory in cgrid_cuda_init().\n");
      abort();
    }
    bzero(grid_gpu_rand, len);
    if(!(cuda_add_block(grid_gpu_rand, len, "GPU RAND", 0))) {
      fprintf(stderr, "libgrid(CUDA): Failed to allocate temporary space on GPU.\n");
      abort();
    }
    grid_gpu_rand_addr = cuda_block_address(grid_gpu_rand);
    cuda_lock_block(grid_gpu_rand);
  }
  grid_cuda_random_seedW(states, seed, nx, ny, nz);
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
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  states = ((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
         * ((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
         * ((nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_random_seed(states, time(0), nx, ny, nz);

  rgrid_cuda_random_uniformW((CUREAL *) cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

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
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  states = ((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
         * ((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
         * ((nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_random_seed(states, time(0), nx, ny, nz);

  rgrid_cuda_random_normalW((CUREAL *) cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

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
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  states = ((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
         * ((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
         * ((nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(gen == NULL) grid_cuda_random_seed(states, time(0), nx, ny, nz);

  cgrid_cuda_random_uniformW((CUCOMPLEX *) cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

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
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  states = ((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
         * ((ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
         * ((nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(gen == NULL) grid_cuda_random_seed(states, time(0), nx, ny, nz);

  cgrid_cuda_random_normalW((CUCOMPLEX *) cuda_block_address(grid->value), scale, grid->nx, grid->ny, grid->nz);

  return 0;
}
