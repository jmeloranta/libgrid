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
#include <cuda/cufft.h>
#include <cuda/cufftXt.h>
#include "cuda-vars.h"

extern void *grid_gpu_rand_addr;
extern "C" void cuda_error_check();
extern "C" int cuda_ngpus();
extern "C" int *cuda_gpus();

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
  INT i, *gpus = (INT *) cuda_gpus();

  for(i = 0; i < cuda_ngpus(); i++) {
    cudaSetDevice(gpus[i]);
    grid_cuda_random_seed_gpu<<<blocks,threads>>>(st, states, seed);
  }
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
 * grid    = Destination for operation (cudaXtDesc *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_uniformW(cudaXtDesc *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    rgrid_cuda_random_uniform_gpu<<<blocks1,threads>>>((CUREAL *) grid->data[i], st, scale, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    rgrid_cuda_random_uniform_gpu<<<blocks2,threads>>>((CUREAL *) grid->data[i], st, scale, nnx2, ny, nz, nzz);
  }

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
 * Add normal random numbers between -1 and 1 (scaled by "scale").
 *
 * grid    = Destination for operation (cudaXtDesc *; output).
 * scale   = Scaling factor (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_normalW(cudaXtDesc *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    rgrid_cuda_random_normal_gpu<<<blocks1,threads>>>((CUREAL *) grid->data[i], st, scale, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    rgrid_cuda_random_normal_gpu<<<blocks2,threads>>>((CUREAL *) grid->data[i], st, scale, nnx2, ny, nz, nzz);
  }

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
 * grid    = Destination for operation (cudaXtDesc *; output).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 * space   = which space (real or reciprocal) (char; input).
 *
 */

extern "C" void cgrid_cuda_random_uniformW(cudaXtDesc *grid, CUREAL scale, INT nx, INT ny, INT nz, char space) {

  SETUP_VARIABLES(grid);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_random_uniform_gpu<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], st, scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_random_uniform_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], st, scale, nnx2, nny2, nz);
  }

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
 * grid    = Destination for operation (cudaXtDesc *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 * space   = which space (real or reciprocal) (char; input).
 *
 */

extern "C" void cgrid_cuda_random_normalW(cudaXtDesc *grid, CUREAL scale, INT nx, INT ny, INT nz, char space) {

  SETUP_VARIABLES(grid);
  curandState *st = (curandState *) grid_gpu_rand_addr;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_random_normal_gpu<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], st, scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_random_normal_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], st, scale, nnx2, nny2, nz);
  }

  cuda_error_check();
}
