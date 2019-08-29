/*
 * CUDA device code (for cuRAND).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cufft.h>
#ifdef SINGLE_PREC
#define CUREAL float
#define CUCOMPLEX cufftComplex
#else
#define CUREAL double
#define CUCOMPLEX cufftDoubleComplex
#endif
#include "cuda-math.h"

extern void *grid_gpu_rand_addr;
extern "C" void cuda_error_check();

/*
 *
 * Set random number seeds for cuRAND.
 *
 * Every block has its own random number seed.
 * 
 */

__global__ void grid_cuda_random_seed_gpu(curandState *st, INT states, INT seed) {

  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(cstate >= states) return;

  curand_init(seed + cstate, cstate, 0, &st[cstate]);
}

/*
 * Wrapper for setting up random number seeds.
 *
 * states = Number of states (= number of blocks) (INT; input).
 * seed   = Base random number seed (INT; input).
 *
 */

extern "C" void grid_cuda_random_seedW(INT states, INT seed) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks(1, 1, 1);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  grid_cuda_random_seed_gpu<<<blocks,threads>>>(st, states, seed);
  cuda_error_check();
}

/*
 * Add uniform random numbers to real grid (uniform distribution between +- scale).
 *
 */

__global__ void rgrid_cuda_random_uniform_gpu(CUREAL *grid, curandState *st, CUREAL scale, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

#if CUREAL == float
  grid[idx] = grid[idx] + 2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale;
#else
  grid[idx] = grid[idx] + 2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale;
#endif
}

/*
 * Add uniform random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (CUREAL *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_uniformW(CUREAL *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  rgrid_cuda_random_uniform_gpu<<<blocks,threads>>>(grid, st, scale, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Add normal random numbers to real grid (normal distribution between +- scale).
 *
 */

__global__ void rgrid_cuda_random_normal_gpu(CUREAL *grid, curandState *st, REAL scale, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

#if CUREAL == float
  grid[idx] = grid[idx] + curand_normal(&st[cstate]);
#else
  grid[idx] = grid[idx] + curand_normal_double(&st[cstate]);
#endif
}

/*
 * Add normal random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (CUREAL *; output).
 * scale   = Scaling factor (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_normalW(CUREAL *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  rgrid_cuda_random_normal_gpu<<<blocks,threads>>>(grid, st, scale, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Add uniform random numbers to complex grid (uniform distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_uniform_gpu(CUCOMPLEX *grid, curandState *st, REAL scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#if CUREAL == float
  grid[idx] = grid[idx] + CUMAKE(2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale, 2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale);
#else
  grid[idx] = grid[idx] + CUMAKE(2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale, 2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale);
#endif
}

/*
 * Add uniform random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (CUCOMPLEX *; output).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_uniformW(CUCOMPLEX *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  cgrid_cuda_random_uniform_gpu<<<blocks,threads>>>(grid, st, scale, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add normal random numbers to real grid (normal distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_normal_gpu(CUCOMPLEX *grid, curandState *st, REAL scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#if CUREAL == float
  grid[idx] = grid[idx] + CUMAKE(curand_normal(&st[cstate]), curand_normal(&st[cstate]));
#else
  grid[idx] = grid[idx] + CUMAKE(curand_normal_double(&st[cstate]), curand_normal_double(&st[cstate]));
#endif
}

/*
 * Add normal random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (CUREAL *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_normalW(CUCOMPLEX *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  cgrid_cuda_random_normal_gpu<<<blocks,threads>>>(grid, st, scale, nx, ny, nz);
  cuda_error_check();
}
