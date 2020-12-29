/*
 * Function #3: Backflow related function for libdft.
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../cuda.h"
#include "../cuda-math.h"
#include "../cuda-vars.h"

#include "func3.h"

extern "C" void cuda_error_check();

__global__ void grid_func3_cuda_operate_one_product_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop, tmp, tmp2, tmp3;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = src2[idx];

  tmp3 = xi * (rhop - rhobf);
  if(tmp3 > DFT_MAX_COSH) tmp3 = DFT_MAX_COSH;
  tmp = 1.0 / (COSH(tmp3) + DFT_BF_EPS);
  tmp2 = 0.5 * (1.0 - TANH(tmp3));
  
  dst[idx] = src1[idx] * (rhop * (-0.5 * xi * tmp * tmp) + tmp2);
}

extern "C" void grid_func3_cuda_operate_one_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL xi, CUREAL rhobf, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): func3 wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func3_cuda_operate_one_product_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], xi, rhobf, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_func3_cuda_operate_one_product_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], xi, rhobf, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}
