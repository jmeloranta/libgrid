/*
 * CUDA device code for rfunction objects.
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "rgrid_bc-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

/*
 * Return precomputed function value at given value (no interpolation). Not to be called directly.
 *
 * pfunc = Precomputed function (rfunction *; input).
 * x     = Value where the function is evaluated (REAL; input).
 * 
 * Returns the function value.
 *
 */

__device__ inline CUREAL rgrid_cuda_function_value(CUREAL *pfunc, CUREAL x, CUREAL begin, INT nsteps, CUREAL step) {

  INT i;

  i = (INT) ((x - begin) / step);

  if(i < 0) i = 0;
  if(i >= nsteps) i = nsteps - 1;

  return pfunc[i];    
}

/*
 *
 * dst = src1 * operator(src2).
 *
 */

__global__ void rgrid_cuda_operate_one_product_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, CUREAL *func, INT nx, INT ny, INT nz, INT nzz, CUREAL begin, INT nsteps, CUREAL step) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = src1[idx] * rgrid_cuda_function_value(func, src2[idx], begin, nsteps, step);
}

/*
 * Operate on a grid by a function and multiply.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src1     = Source for operation (gpu_mem_block *; input).
 * src2     = Source for operation (gpu_mem_block *; input).
 * func     = Precomputed function (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * begin    = Function x range start (CUREAL; input).
 * nsteps   = Number of values (INT; input).
 * step     = Step length (CUREAL; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_function_operate_one_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, gpu_mem_block *func, INT nx, INT ny, INT nz, CUREAL begin, INT nsteps, CUREAL step) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): function_operate_one_product must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_function_operate_one_product_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], (CUREAL *) func, nnx1, ny, nz, nzz, begin, nsteps, step);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_function_operate_one_product_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], (CUREAL *) func, nnx2, ny, nz, nzz, begin, nsteps, step);
  }

  cuda_error_check();
}

/*
 *
 * dst = operator(src).
 *
 */

__global__ void rgrid_cuda_operate_one_gpu(CUREAL *dst, CUREAL *src, CUREAL *func, INT nx, INT ny, INT nz, INT nzz, CUREAL begin, INT nsteps, CUREAL step) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = rgrid_cuda_function_value(func, src[idx], begin, nsteps, step);
}

/*
 * Operate on a grid by a function.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src      = Source for operation (gpu_mem_block *; input).
 * func     = Precomputed function (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * begin    = Function x range start (CUREAL; input).
 * nsteps   = Number of values (INT; input).
 * step     = Step length (CUREAL; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_function_operate_oneW(gpu_mem_block *dst, gpu_mem_block *src, gpu_mem_block *func, INT nx, INT ny, INT nz, CUREAL begin, INT nsteps, CUREAL step) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): Power must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_function_operate_one_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], (CUREAL *) func, nnx1, ny, nz, nzz, begin, nsteps, step);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_function_operate_one_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], (CUREAL *) func, nnx2, ny, nz, nzz, begin, nsteps, step);
  }

  cuda_error_check();
}
