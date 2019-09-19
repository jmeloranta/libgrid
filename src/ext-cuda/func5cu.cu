/*
 * Function #5: Backflow related function for libdft.
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../cuda-math.h"

#include "func5.h"

extern "C" void cuda_error_check();

__global__ void grid_func5_cuda_operate_one_product_gpu(CUREAL *c, CUREAL *b, CUREAL *a, CUREAL beta, CUREAL rhom, REAL C, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = a[idx];
  
  c[idx] = b[idx] * (C * (1.0 + TANH(beta * (rhop - rhom))) * rhop);
}

__global__ void grid_func5_cuda_operate_one_gpu(CUREAL *c, CUREAL *a, CUREAL beta, CUREAL rhom, CUREAL C, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = a[idx];
  
  c[idx] = (C * (1.0 + TANH(beta * (rhop - rhom))) * rhop);
}

extern "C" void grid_func5_cuda_operate_one_productW(CUREAL *gridc, CUREAL *gridb, CUREAL *grida, CUREAL beta, CUREAL rhom, CUREAL C, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func5_cuda_operate_one_product_gpu<<<blocks,threads>>>(gridc, gridb, grida, beta, rhom, C, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

extern "C" void grid_func5_cuda_operate_oneW(CUREAL *gridc, CUREAL *grida, CUREAL beta, CUREAL rhom, CUREAL C, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func5_cuda_operate_one_gpu<<<blocks,threads>>>(gridc, grida, beta, rhom, C, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}
