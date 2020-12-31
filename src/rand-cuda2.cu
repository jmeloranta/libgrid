/*
 * CUDA device code (for cuRAND).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 */

#include <stdio.h>
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
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-vars.h"
#include "cuda.h"

extern cudaXtDesc *grid_gpu_rand_addr;
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
  int *gpus = (int *) cuda_gpus();  // This must be int rather than INT
  INT i;

  for(i = 0; i < cuda_ngpus(); i++) {
    cudaSetDevice(gpus[i]);
    grid_cuda_random_seed_gpu<<<blocks,threads>>>((curandState *) grid_gpu_rand_addr->data[i], states, seed);
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

#ifdef SINGLE_PREC
  grid[idx] = grid[idx] + 2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale;
#else
  grid[idx] = grid[idx] + 2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale;
#endif
}

/*
 * Add uniform random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_uniformW(gpu_mem_block *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgird(cuda): random_uniform wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_random_uniform_gpu<<<blocks1,threads>>>((CUREAL *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_random_uniform_gpu<<<blocks2,threads>>>((CUREAL *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Add normal random numbers to real grid (normal distribution between +- scale).
 *
 */

__global__ void rgrid_cuda_random_normal_gpu(CUREAL *grid, curandState *st, CUREAL scale, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

#ifdef SINGLE_PREC
  grid[idx] = grid[idx] + scale * curand_normal(&st[cstate]);
#else
  grid[idx] = grid[idx] + scale * curand_normal_double(&st[cstate]);
#endif
}

/*
 * Add normal random numbers between -1 and 1 (scaled by "scale").
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Scaling factor (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_random_normalW(gpu_mem_block *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgird(cuda): random_normal wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_random_normal_gpu<<<blocks1,threads>>>((CUREAL *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_random_normal_gpu<<<blocks2,threads>>>((CUREAL *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Add uniform random numbers to complex grid (uniform distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_uniform_gpu(CUCOMPLEX *grid, curandState *st, CUCOMPLEX scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#ifdef SINGLE_PREC
  grid[idx] = grid[idx] + CUMAKE(2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale.x, 2.0 * (curand_uniform(&st[cstate]) - 0.5) * scale.y);
#else
  grid[idx] = grid[idx] + CUMAKE(2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale.x, 2.0 * (curand_uniform_double(&st[cstate]) - 0.5) * scale.y);
#endif
}

/*
 * Add uniform random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Random number scle (-scale, scale) (CUCOMPLEX; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_uniformW(gpu_mem_block *grid, CUCOMPLEX scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_uniform_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_uniform_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add uniform random numbers to complex grid (uniform distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_uniform_sp_gpu(CUCOMPLEX *grid, curandState *st, CUREAL scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  REAL r1, r2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#ifdef SINGLE_PREC
  r1 = scale * curand_uniform(&st[cstate]);
  r2 = curand_uniform(&st[cstate]) * 2.0 * M_PI;
#else
  r1 = scale * curand_uniform_double(&st[cstate]);
  r2 = curand_uniform_double(&st[cstate]) * 2.0 * M_PI;
#endif
  grid[idx].x = grid[idx].x + r1 * COS(r2);
  grid[idx].y = grid[idx].y + r1 * SIN(r2);
}

/*
 * Add uniform random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Random number scle (-scale, scale) (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_uniform_spW(gpu_mem_block *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_uniform_sp_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_uniform_sp_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add normal random numbers to real grid (normal distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_normal_gpu(CUCOMPLEX *grid, curandState *st, CUCOMPLEX scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#ifdef SINGLE_PREC
  grid[idx] = grid[idx] + CUMAKE(scale.x * curand_normal(&st[cstate]), scale.y * curand_normal(&st[cstate]));
#else
  grid[idx] = grid[idx] + CUMAKE(scale.x * curand_normal_double(&st[cstate]), scale.y * curand_normal_double(&st[cstate]));
#endif
}

/*
 * Add normal random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Scale for the random numbers (CUCOMPLEX; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_normalW(gpu_mem_block *grid, CUCOMPLEX scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_normal_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_normal_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add normal random numbers to real grid (normal distribution between +- scale).
 *
 */

__global__ void cgrid_cuda_random_normal_sp_gpu(CUCOMPLEX *grid, curandState *st, CUREAL scale, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  INT cstate = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  CUREAL r1, r2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

#ifdef SINGLE_PREC
  r1 = scale * curand_normal(&st[cstate]);
  r2 = curand_uniform(&st[cstate]) * M_PI;
#else
  r1 = scale * curand_normal_double(&st[cstate]);
  r2 = curand_uniform_double(&st[cstate]) * M_PI;
#endif
  grid[idx].x = grid[idx].x + r1 * COS(r2);
  grid[idx].y = grid[idx].y + r1 * SIN(r2);
}

/*
 * Add normal random numbers between -scale and +scale to real grid.
 *
 * grid    = Destination for operation (gpu_mem_block *; output).
 * scale   = Scale for the random numbers (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_random_normal_spW(gpu_mem_block *grid, CUREAL scale, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_normal_sp_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_random_normal_sp_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (curandState *) grid_gpu_rand_addr->data[i], scale, nnx2, nny2, nz);
  }

  cuda_error_check();
}
