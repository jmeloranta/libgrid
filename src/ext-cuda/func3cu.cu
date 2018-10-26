/*
 * Function #3: Backflow related function for libdft.
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "../cuda-math.h"

#include "func3.h"

extern "C" void cuda_error_check();

__global__ void grid_func3_cuda_operate_one_product_gpu(CUREAL *c, CUREAL *b, CUREAL *a, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop, tmp, tmp2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = a[idx];

  tmp = 1.0 / (COSH((rhop - rhobf) * xi) + DFT_BF_EPS);
  tmp2 = 0.5 * (1.0 - TANH(xi * (rhop - rhobf)));
  
  c[idx] = b[idx] * (rhop * (-0.5 * xi * tmp * tmp) + tmp2);
}

extern "C" void grid_func3_cuda_operate_one_productW(CUREAL *gridc, CUREAL *gridb, CUREAL *grida, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func3_cuda_operate_one_product_gpu<<<blocks,threads>>>(gridc, gridb, grida, xi, rhobf, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}
