/*
 * CUDA device code (REAL complex = CUCOMPLEX; cgrid).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 * Since the ordering of the data on multiple GPUs changes after FFT,
 * we must keep track of this. The variable space in cgrid structure (structs.h)
 * is zero if the grid is currently in real space and one if it is in reciprocal space (i.e., after FFT).
 * The real space data is distributed on GPUs according to the first index (x as in data[x][y][z])
 * where as the resiprocal space data is distributed along the second index (y as in data[x][y][z]).
 *
 * SETUP_VARIABLES and SETUP_VARIABLES_SEG are macros that follow the correct distribution over GPUs
 * based on the value of space variable. The latter also keeps track of the actual index so that
 * routines that need to access proper grid indices can do so.
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "cgrid_bc-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

extern void *grid_gpu_mem;            /* host memory holder for the block used for reductions */
extern cudaXtDesc *grid_gpu_mem_addr; /* the corresponding GPU memory addresses */
extern "C" void cuda_error_check();

/*
 * Fourier space convolution device code.
 *
 * dst = src1 * src2 but with alternating signs for FFT.
 *
 */

__global__ void cgrid_cuda_fft_convolute_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if((i + j + k) & 1) norm *= -1.0;
  dst[idx] = src1[idx] * src2[idx] * norm;
}

/*
 * Convolution in the Fourier space.
 *
 * dst   = convolution output (gpu_mem_block *; output).
 * src1  = 1st grid to be convoluted (gpu_mem_block *; input).
 * src2  = 2nd grid to be convoluted (gpu_mem_block *; input).
 * norm  = FFT norm (REAL complex; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 * In Fourier space.
 *
 */

extern "C" void cgrid_cuda_fft_convoluteW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_convolute_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], norm, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_convolute_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], norm, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Grid abs power device code.
 *
 * dst = POW(|src|,x)
 *
 */

__global__ void cgrid_cuda_abs_power_gpu(CUCOMPLEX *dst, CUCOMPLEX *src, CUREAL x, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx].x = POW(CUCREAL(src[idx]) * CUCREAL(src[idx]) + CUCIMAG(src[idx]) * CUCIMAG(src[idx]), x / 2.0);
  dst[idx].y = 0.0;
}

/*
 * Grid abs power.
 *
 * dst      = Destination for operation (gpu_mem_block *; input).
 * src      = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * In real space.
 *
 */

extern "C" void cgrid_cuda_abs_powerW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_abs_power_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], exponent, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_abs_power_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], exponent, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Grid power device code.
 *
 * dst = POW(src,x)
 *
 */

__global__ void cgrid_cuda_power_gpu(CUCOMPLEX *dst, CUCOMPLEX *src, CUREAL x, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx] = CUCPOW(src[idx], x);
}

/*
 * Grid power.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src      = Source for operation (gpu_mem_block *; input).
 * exponent = Exponent (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_powerW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_power_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], exponent, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_power_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], exponent, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Multiply grid by constant device code.
 *
 * dst = C * dst
 *
 */

__global__ void cgrid_cuda_multiply_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] * c;
}

/*
 * Multiply grid by a constant.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * c        = Multiplying constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiplyW(gpu_mem_block *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_multiply_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_multiply_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Sum of two grids.
 *
 * dst = src1 + src2
 *
 */

__global__ void cgrid_cuda_sum_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] + src2[idx];
}

/*
 * Sum of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Input grid 1 (gpu_mem_block *; input).
 * src2     = Input grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_sumW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_sum_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_sum_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Subtract of two grids.
 *
 * dst = src1 - src2
 *
 */

__global__ void cgrid_cuda_difference_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] - src2[idx];
}

/*
 * Subtract two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Input grid 1 (gpu_mem_block *; input).
 * src2     = Input grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_differenceW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_difference_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_difference_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Product of two grids.
 *
 * dst = src1 * src2.
 *
 */

__global__ void cgrid_cuda_product_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] * src2[idx];
}

/*
 * Product of two grids.
 *
 * grida    = Destination grid (gpu_mem_block *; output).
 * gridb    = Source grid 1 (gpu_mem_block *; input).
 * gridc    = Source grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Conjugate product of two grids.
 *
 * dst = src1^* X src2.
 *
 */

__global__ void cgrid_cuda_conjugate_product_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = CUCONJ(src1[idx]) * src2[idx];
}

/*
 * Conjugate product of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Source grid 1 (gpu_mem_block *; input).
 * src2     = Source grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugate_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_conjugate_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_conjugate_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Division of two grids.
 *
 * dst = src1 / src2.
 *
 * Note: Avoid division as it is slow on GPUs.
 *
 */

__global__ void cgrid_cuda_division_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] / src2[idx];
}

/*
 * Division of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Source grid 1 (gpu_mem_block *; input).
 * src2     = Source grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_divisionW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_division_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_division_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Safe division of two grids.
 *
 * dst = src1 / (src2 + eps).
 *
 * Note: Avoid division as it is slow on GPUs.
 *
 */

__global__ void cgrid_cuda_division_eps_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, CUREAL eps, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  dst[idx] = src1[idx] / (src2[idx] + eps);
}

/*
 * "Safe" division of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Source grid 1 (gpu_mem_block *; input).
 * src2     = Source grid 2 (gpu_mem_block *; input).
 * eps      = Epsilon (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_division_epsW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL eps, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_division_eps_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], eps, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_division_eps_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], eps, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add constant to grid device code.
 *
 * dst = dst + c
 *
 */

__global__ void cgrid_cuda_add_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + c;
}

/*
 * Add constant to grid.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * c        = Constant (CUCOMPLEX).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_addW(gpu_mem_block *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add multiply and add device code.
 *
 * dst = cm * dst + ca
 *
 */

__global__ void cgrid_cuda_multiply_and_add_gpu(CUCOMPLEX *dst, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = (cm * dst[idx]) + ca;
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * cm       = Multiplier (CUCOMPLEX; input).
 * ca       = Additive constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiply_and_addW(gpu_mem_block *dst, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_multiply_and_add_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], cm, ca, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_multiply_and_add_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], cm, ca, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add multiply and add device code.
 *
 * dst = cm * (dst + ca)
 *
 */

__global__ void cgrid_cuda_add_and_multiply_gpu(CUCOMPLEX *dst, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = cm * (dst[idx] + ca);
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * ca       = Additive constant (CUCOMPLEX; input).
 * cm       = Multiplier (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_and_multiplyW(gpu_mem_block *dst, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_and_multiply_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], ca, cm, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_and_multiply_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], ca, cm, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add scaled grid device code.
 *
 * dst = dst + d * src
 *
 */

__global__ void cgrid_cuda_add_scaled_gpu(CUCOMPLEX *dst, CUCOMPLEX d, CUCOMPLEX *src, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + (d * src[idx]);
}

/*
 * Scaled add grid.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * d        = Scaling factor (CUCOMPLEX; input).
 * srd      = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaledW(gpu_mem_block *dst, CUCOMPLEX d, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_scaled_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], d, (CUCOMPLEX *) SRC->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_scaled_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], d, (CUCOMPLEX *) SRC->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Add scaled product grid device code.
 *
 * dst = dst + d * src1 * src2
 *
 */

__global__ void cgrid_cuda_add_scaled_product_gpu(CUCOMPLEX *dst, CUCOMPLEX d, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + (d * src1[idx] * src2[idx]);
}

/*
 * Add scaled product.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * d        = Scaling factor (CUCOMPLEX; input).
 * src1     = Source for operation (gpu_mem_block *; input).
 * src2     = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaled_productW(gpu_mem_block *dst, CUCOMPLEX d, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_scaled_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], d, (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_add_scaled_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], d, (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Set dst to constant: dst = c
 *
 */

__global__ void cgrid_cuda_constant_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = c;
}

/*
 * Set grid to constant.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * c        = Constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_constantW(gpu_mem_block *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_constant_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_constant_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], c, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Block init (zero elements).
 *
 * blocks  = Block table (CUCOMPLEX *; output).
 * nblocks = Number of blocks in table (INT; input).
 *
 */

__global__ void cgrid_cuda_block_init(CUCOMPLEX *blocks, INT nblocks) {

  INT i;

  for(i = 0; i < nblocks; i++) blocks[i].x = blocks[i].y = 0.0;
}

/*
 * Block reduction.
 *
 * blocks  = Block list to reduce (CUCOMPLEX *; input/output). blocks[0] will contain the reduced value.
 * nblocks = Number of blocks (INT; input).
 *
 */

__global__ void cgrid_cuda_block_reduce(CUCOMPLEX *blocks, INT nblocks) {

  INT i;

  for(i = 1; i < nblocks; i++) blocks[0] = blocks[0] + blocks[i];
}

/*
 * Integrate over grid a.
 *
 */

__global__ void cgrid_cuda_integral_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els[idx2] = els[idx2] + a[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] = blocks[idx2] + els[t]; // reduce threads
    }
  }
}

/*
 * Integrate over grid.
 *
 * grid     = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void cgrid_cuda_integralW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, nny1, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);  // reduce over blocks
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx2, nny2, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  // + overloaded to device function - work around!
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * Integrate over A with limits.
 *
 * nx = number of elements for current GPU.
 * nx2 = Position in the overall grid.
 *
 */

__global__ void cgrid_cuda_integral_region_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, INT segx, INT segy) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, ii = i + segx, jj = j + segy;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  if(ii >= il && ii <= iu && jj >= jl && jj <= ju && k >= kl && k <= ku) {
    idx = (i * ny + j) * nz + k;
    idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    els[idx2] = els[idx2] + a[idx];
  }
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] = blocks[idx2] + els[t];  // reduce threads
    }
  }
}

/*
 * Integrate over grid with limits.
 *
 * grid     = Source for operation (gpu_mem_block *; input).
 * il       = Lower index for x (INT; input).
 * iu       = Upper index for x (INT; input).
 * jl       = Lower index for y (INT; input).
 * ju       = Upper index for y (INT; input).
 * kl       = Lower index for z (INT; input).
 * ku       = Upper index for z (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void cgrid_cuda_integral_regionW(gpu_mem_block *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z, segx = 0, segy = 0;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(il < 0) il = 0;  
  if(jl < 0) jl = 0;  
  if(kl < 0) kl = 0;  
  if(iu > nx-1) iu = nx-1;
  if(ju > ny-1) ju = ny-1;
  if(ku > nz-1) ku = nz-1;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], il, iu, jl, ju, kl, ku, nnx1, nny1, nz, segx, segy);
    segx += dsegx1;
    segy += dsegy1;
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_region_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], il, iu, jl, ju, kl, ku, nnx2, nny2, nz, segx, segy);
    segx += dsegx2;
    segy += dsegy2;
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  /// + overloaded to device function - work around!
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * Integrate of |A|^2.
 *
 */

__global__ void cgrid_cuda_integral_of_square_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els[idx2] = els[idx2] + (a[idx].x * a[idx].x + a[idx].y * a[idx].y);
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] = blocks[idx2] + els[t];  // reduce threads
    }
  }
}

/*
 * Integral of square (|grid|^2).
 *
 * grid     = Source for operation (udaXtDesc_t *; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void cgrid_cuda_integral_of_squareW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_square_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, nny1, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_square_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx2, nny2, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);  // get on the real part
    value->x += tmp.x;
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * Integrate A^* X B (overlap).
 *
 */

__global__ void cgrid_cuda_integral_of_conjugate_product_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els[idx2].x += a[idx].x * b[idx].x + a[idx].y * b[idx].y;  // A^* times B
  els[idx2].y += a[idx].x * b[idx].y - a[idx].y * b[idx].x;
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] = blocks[idx2] + els[t];  // reduce threads
    }
  }
}

/*
 * Integral of conjugate product (overlap).
 *
 * grid     = Source 1 for operation (gpu_mem_block *; input).
 * src      = Source 2 for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void cgrid_cuda_integral_of_conjugate_productW(gpu_mem_block *grid, gpu_mem_block *src, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_conjugate_product_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) SRC->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, nny1, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_conjugate_product_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) SRC->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx2, nny2, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  /// + overloaded to device function - work around!
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * Integrate A * |B|^2.
 *
 */

__global__ void cgrid_cuda_grid_expectation_value_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  CUREAL tmp;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  tmp = b[idx].x * b[idx].x + b[idx].y * b[idx].y;
  els[idx2] = els[idx2] + (a[idx] * tmp);
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] = blocks[idx2] + els[t];  // reduce threads
    }
  }
}

/*
 * Integral A * |B|^2.
 *
 * grida    = Source 1 (A) for operation (gpu_mem_block *; input).
 * gridb    = Source 2 (B) for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void cgrid_cuda_grid_expectation_valueW(gpu_mem_block *grida, gpu_mem_block *gridb, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES(grida);
  cudaXtDesc *GRIDA = grida->gpu_info->descriptor, *GRIDB = gridb->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRIDA->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_grid_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRIDA->data[i], (CUCOMPLEX *) GRIDB->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, nny1, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRIDA->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_grid_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) GRIDA->data[i], (CUCOMPLEX *) GRIDB->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx2, nny2, nz);
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  /// + overloaded to device function - work around!
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * dst = FD_X(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * dst = FD_X(src)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;
 
  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = FD_Y(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_y_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Y(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = FD_Z(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_z_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Z(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz)
                         - 6.0 * src[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplacian in dst.
 *
 */

extern "C" void cgrid_cuda_fd_laplaceW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_X(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_X(src)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_Y(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_y_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_Y(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_Z(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_z_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_Z(src)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = FD_X(src)^2 + FD_Y(src)^2 + FD_Z(src)^2.
 *
 */

__global__ void cgrid_cuda_fd_gradient_dot_gradient_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUCOMPLEX tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx] = CUMAKE(0.0, 0.0);

  tmp = inv_delta * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz));
  dst[idx] = dst[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);

  tmp = inv_delta * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz));
  dst[idx] = dst[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);

  tmp = inv_delta * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz));
  dst[idx] = dst[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);
}

/*
 * dst = FD_X(src)^2 + FD_Y(src)^2 + FD_Z(src)^2.
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (4 * step * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_dot_gradientW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_dot_gradient_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * Complex conjugate.
 *
 * dst = src*
 *
 */

__global__ void cgrid_cuda_conjugate_gpu(CUCOMPLEX *dst, CUCOMPLEX *src, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = CUCONJ(src[idx]);
}

/*
 * Grid conjugate.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src      = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugateW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_conjugate_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_conjugate_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * FFT gradient_x
 *
 */

__global__ void cgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL step, INT nx, INT ny, INT nz, INT nx2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = 2.0 * M_PI * ((CUREAL) i) / (((CUREAL) nx) * step) - kx0;
  else 
    kx = 2.0 * M_PI * ((CUREAL) (i - nx)) / (((CUREAL) nx) * step) - kx0;

  b[idx] = b[idx] * kx * norm;     // multiply by I * kx * norm
}

/*
 * FFT gradient_x
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm       = FFT norm (grid->fft_norm) (CUREAL; input).
 * kx0        = Momentum shift of origin along X (CUREAL; input).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_xW(gpu_mem_block *dst, CUREAL norm, CUREAL kx0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nx2 = nx / 2;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], norm, kx0, step, nnx1, nny1, nz, nx2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], norm, kx0, step, nnx2, nny2, nz, nx2);
  }

  cuda_error_check();
}

/*
 * FFT gradient_y
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL ky0, CUREAL step, INT nx, INT ny, INT nz, INT ny2, INT seg, INT nyy) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (jj <= ny2)
    ky = 2.0 * M_PI * ((CUREAL) jj) / (((CUREAL) nyy) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((CUREAL) (jj - nyy)) / (((CUREAL) nyy) * step) - ky0;

  b[idx] = b[idx] * ky * norm;    // multiply by I * ky * norm
}

/*
 * FFT gradient_y
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm       = FFT norm (grid->fft_norm) (CUREAL; input).
 * ky0        = Momentum shift of origin along Y (CUREAL; input).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_yW(gpu_mem_block *dst, CUREAL norm, CUREAL ky0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT ny2 = ny / 2, segx = 0, segy = 0;  // segx unused

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], norm, ky0, step, nnx1, nny1, nz, ny2, segy, ny);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], norm, ky0, step, nnx2, nny2, nz, ny2, segy, ny);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT gradient_z
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (k <= nz2)
    kz = 2.0 * M_PI * ((CUREAL) k) / (((CUREAL) nz) * step) - kz0;
  else 
    kz = 2.0 * M_PI * ((CUREAL) (k - nz)) / (((CUREAL) nz) * step) - kz0;

  b[idx] = b[idx] * kz * norm;   // multiply by I * kz * norm
}

/*
 * FFT gradient_z
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm       = FFT norm (grid->fft_norm) (CUREAL; input).
 * kz0        = Momentum shift of origin along Z (CUREAL; input).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_zW(gpu_mem_block *dst, CUREAL norm, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nz2 = nz / 2;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], norm, kz0, step, nnx1, nny1, nz, nz2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], norm, kz0, step, nnx2, nny2, nz, nz2);
  }

  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2, INT seg, INT nyy) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (i <= nx2)
    kx = 2.0 * M_PI * ((CUREAL) i) / (((CUREAL) nx) * step) - kx0;
  else 
    kx = 2.0 * M_PI * ((CUREAL) (i - nx)) / (((CUREAL) nx) * step) - kx0;
      
  if (jj <= ny2)
    ky = 2.0 * M_PI * ((CUREAL) jj) / (((CUREAL) nyy) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((CUREAL) (jj - nyy)) / (((CUREAL) nyy) * step) - ky0;
      
  if (k <= nz2)
    kz = 2.0 * M_PI * ((CUREAL) k) / (((CUREAL) nz) * step) - kz0;
  else 
    kz = 2.0 * M_PI * ((CUREAL) (k - nz)) / (((CUREAL) nz) * step) - kz0;      

  b[idx] = b[idx] * (-(kx * kx + ky * ky + kz * kz) * norm);
}

/*
 * FFT laplace
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm     = FFT norm (grid->fft_norm) (CUREAL; input).
 * kx0      = Momentum shift of origin along X (CUREAL; input).
 * ky0      = Momentum shift of origin along Y (CUREAL; input).
 * kz0      = Momentum shift of origin along Z (CUREAL; input).
 * step     = Spatial step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_laplaceW(gpu_mem_block *dst, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, segx = 0, segy = 0; // segx not used

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], norm, kx0, ky0, kz0, step, nnx1, nny1, nz, nx2, ny2, nz2, segy, ny);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], norm, kx0, ky0, kz0, step, nnx2, nny2, nz, nx2, ny2, nz2, segy, ny);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT laplace expectation value.
 *
 * B = <B''> in Fourier space.
 *
 * Only periodic version implemented.
 *
 * Normalization done in cgrid-cuda.c
 *
 */

__global__ void cgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUCOMPLEX *blocks, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2, INT seg, INT nyy) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  CUREAL kx, ky, kz;
  extern __shared__ CUREAL els2[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els2[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
  else 
    kx = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step) - kx0;
      
  if (jj <= ny2)
    ky = 2.0 * M_PI * ((REAL) jj) / (((REAL) nyy) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((REAL) (jj - nyy)) / (((REAL) nyy) * step) - ky0;
      
  if (k <= nz2)
    kz = 2.0 * M_PI * ((REAL) k) / (((REAL) nz) * step) - kz0;
  else 
    kz = 2.0 * M_PI * ((REAL) (k - nz)) / (((REAL) nz) * step) - kz0;

  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els2[idx2] -= (kx * kx + ky * ky + kz * kz) * (CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]));
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2].x += els2[t];  // reduce threads
    }
  }
}

/*
 * FFT laplace expectation value
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Expectation value (REAL; output).
 *
 * Normalization done in cgrid-cuda.c
 *
 */

extern "C" void cgrid_cuda_fft_laplace_expectation_valueW(gpu_mem_block *dst, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES_SEG(dst)
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, segx = 0, segy = 0; // segx not used
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], kx0, ky0, kz0, step, nnx1, nny1, nz, nx2, ny2, nz2, segy, ny);
    segx += dsegx1;
    segy += dsegy1;
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
  }

  cuda_error_check();

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUCOMPLEX)>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], kx0, ky0, kz0, step, nnx2, nny2, nz, nx2, ny2, nz2, segy, ny);
    segx += dsegx2;
    segy += dsegy2;
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  /// + overloaded to device function - work around!
    value->y += tmp.y;
  }

  cuda_error_check();
}

/*
 * Zero real part.
 *
 * A.re = 0
 *
 */

__global__ void cgrid_cuda_zero_re_gpu(CUCOMPLEX *a, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx].x = 0.0;
}

/*
 * Zero real part.
 *
 * grid     = Grid to be operated on (gpu_mem_block *; input/output).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_reW(gpu_mem_block *grid, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_re_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_re_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Zero imaginary part.
 *
 * A.im = 0
 *
 */

__global__ void cgrid_cuda_zero_im_gpu(CUCOMPLEX *a, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx].y = 0.0;
}

/*
 * Zero imaginary part.
 *
 * grid     = Grid to be operated on (gpu_mem_block *; input/output).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_imW(gpu_mem_block *grid, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_im_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_im_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Zero part of complex grid.
 *
 * A = 0 in the specified range.
 *
 */

__global__ void cgrid_cuda_zero_index_gpu(CUCOMPLEX *a, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT segx, INT segy) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, ii = i + segx, jj = j + segy, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if(ii >= lx && ii < hx && jj >= ly && jj < hy && k >= lz && k < hz) a[idx] = CUMAKE(0.0, 0.0);
}

/*
 * Zero specified range of complex grid.
 *
 * grid     = Grid to be operated on (gpu_mem_block *; input/output).
 * lx       = Low x index (INT; input).
 * hx       = High x index (INT; input).
 * ly       = Low y index (INT; input).
 * hy       = High y index (INT; input).
 * lz       = Low z index (INT; input).
 * hz       = High z index (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_indexW(gpu_mem_block *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  INT segx = 0, segy = 0;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_index_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], lx, hx, ly, hy, lz, hz, nnx1, nny1, nz, segx, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_zero_index_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], lx, hx, ly, hy, lz, hz, nnx2, nny2, nz, segx, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * Poisson equation.
 *
 */

__global__ void cgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nz, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  kx = COS(ilx * (CUREAL) i);
  ky = COS(ily * (CUREAL) jj);
  kz = COS(ilz * (CUREAL) k);
  if(i || jj || k)
    grid[idx] = grid[idx] * norm * step2 / (2.0 * (kx + ky + kz - 3.0));
  else
    grid[idx] = CUMAKE(0.0, 0.0);
}

/*
 * Solve Poisson.
 *
 * grid    = Grid specifying the RHS (gpu_mem_block *; input/output).
 * norm    = FFT normalization constant (CUREAL; input).
 * step2   = Spatial step ^ 2 (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_poissonW(gpu_mem_block *grid, CUREAL norm, CUREAL step2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  
  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = 2.0 * M_PI / ((CUREAL) nz);
  INT segx = 0, segy = 0; // segx not used

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_poisson_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], norm, step2, ilx, ily, ilz, nnx1, nny1, nz, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_poisson_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], norm, step2, ilx, ily, ilz, nnx2, nny2, nz, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
