/*
 * Function #1: Backflow related function for libdft.
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../cuda.h"
#include "../cuda-math.h"
#include "../cuda-vars.h"

#include "func1.h"

extern "C" void cuda_error_check();

__global__ void grid_func1_cuda_operate_one_product_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = src2[idx];
  dst[idx] = src1[idx] * FUNCTION;
}

__global__ void grid_func1_cuda_operate_one_gpu(CUREAL *dst, CUREAL *src, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = src[idx];
  dst[idx] = FUNCTION;
}

extern "C" void grid_func1_cuda_operate_one_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func1_cuda_operate_one_product_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], xi, rhobf, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func1_cuda_operate_one_product_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], xi, rhobf, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

extern "C" void grid_func1_cuda_operate_oneW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func1_cuda_operate_one_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], xi, rhobf, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func1_cuda_operate_one_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], xi, rhobf, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}
