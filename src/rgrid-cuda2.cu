/*
 * CUDA device code (REAL; rgrid).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 * nzz: 2 * (nz / 2 + 1) for real space
 *      (nz / 2 + 1) for reciprocal space
 *
 * x, y, z: split along x for GPUs in real space   (uses nnx1, nnx2)
 *          split along y for GPUs in reciprocal space (uses nny1, nny2)
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "rgrid_bc-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

extern void *grid_gpu_mem;
extern cudaXtDesc *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 *
 * dst = src1 * src2 but with alternating signs for FFT.
 *
 * Fourier space.
 *
 */

__global__ void rgrid_cuda_fft_convolute_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if((i + j + k) & 1) norm *= -1.0;
  dst[idx] = norm * src1[idx] * src2[idx];
}

/*
 * Convolution in the Fourier space (data in GPU). Not called directly.
 *
 * Multiplication in GPU memory: grid_gpu_mem[i] = grid_gpu_mem[i] * grid_gpu_mem[j] (with sign variation).
 * Note: this includes the sign variation needed for convolution as well as normalization!
 *
 * dst   = output (gpu_mem_block *; output).
 * src1  = 1st grid to be convoluted (gpu_mem_block *; input).
 * src2  = 2nd grid to be convoluted (gpu_mem_block *; input).
 * norm  = FFT norm (CUREAL; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_fft_convoluteW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): convolution sources must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_fft_convolute_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], norm, nx, nny1, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_fft_convolute_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) SRC1->data[i], (CUCOMPLEX *) SRC2->data[i], norm, nx, nny2, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;

  cuda_error_check();
}

/*
 *
 * dst = src^x
 *
 */

__global__ void rgrid_cuda_power_gpu(CUREAL *dst, CUREAL *src, CUREAL x, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = POW(src[idx], x);
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
 * Real space.
 *
 */

extern "C" void rgrid_cuda_powerW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): Power must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_power_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_power_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = |src|^x
 *
 */

__global__ void rgrid_cuda_abs_power_gpu(CUREAL *dst, CUREAL *src, CUREAL x, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = POW(FABS(src[idx]), x);
}

/*
 * Grid abs power.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src      = Source for operation (gpu_mem_block *; input).
 * exponent = Exponent (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_abs_powerW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): Abs power must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_abs_power_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_abs_power_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = c * dst
 *
 */

__global__ void rgrid_cuda_multiply_gpu(CUREAL *dst, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= nx || j >= ny || k >= nz) return;

  dst[(i * ny + j) * nzz + k] *= c;
}

/*
 * Multiply grid by a constant.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiplyW(gpu_mem_block *grid, CUREAL c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): multiply must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_multiply_gpu<<<blocks1,threads>>>((CUREAL *) GRID->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_multiply_gpu<<<blocks2,threads>>>((CUREAL *) GRID->data[i], c, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = c * dst (in Fourier space)
 *
 */

__global__ void rgrid_cuda_multiply_fft_gpu(CUCOMPLEX *dst, CUREAL c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx] = dst[idx] * c;
}

/*
 * Multiply (complex) grid by a constant (in Fourier space).
 *
 * st       = Grid to be operated on (gpu_mem_block *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_fftW(gpu_mem_block *grid, CUREAL c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): multiply_fft must be in Fourier space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_multiply_fft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], c, nx, nny1, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_multiply_fft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], c, nx, nny2, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = src1 + src2
 *
 */

__global__ void rgrid_cuda_sum_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

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

extern "C" void rgrid_cuda_sumW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): Sum must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_sum_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_sum_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;

  cuda_error_check();
}

/*
 *
 * dst = src1 - src2
 *
 */

__global__ void rgrid_cuda_difference_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

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

extern "C" void rgrid_cuda_differenceW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): difference must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_difference_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_difference_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = src1 * src2.
 *
 */

__global__ void rgrid_cuda_product_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = src1[idx] * src2[idx];
}

/*
 * Product of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Source grid 1 (gpu_mem_block *; input).
 * src2     = Source grid 2 (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_productW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): product must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_product_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_product_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = src1 / src2.
 *
 */

__global__ void rgrid_cuda_division_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

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

extern "C" void rgrid_cuda_divisionW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): division must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_division_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_division_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = src1 / (src2 + eps).
 *
 */

__global__ void rgrid_cuda_division_eps_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, CUREAL eps, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = src1[idx] / (src2[idx] + eps);
}

/*
 * Division of two grids.
 *
 * dst      = Destination grid (gpu_mem_block *; output).
 * src1     = Source grid 1 (gpu_mem_block *; input).
 * src2     = Source grid 2 (gpu_mem_block *; input).
 * eps      = Epsilon (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_division_epsW(gpu_mem_block *dst, gpu_mem_block *src1, gpu_mem_block *src2, CUREAL eps, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): division_eps must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_division_eps_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], eps, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_division_eps_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], eps, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 *
 * dst = dst + c
 *
 */

__global__ void rgrid_cuda_add_gpu(CUREAL *dst, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] += c;
}

/*
 * Add constant to grid.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * c        = Constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_addW(gpu_mem_block *grid, CUREAL c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): add must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_add_gpu<<<blocks1,threads>>>((CUREAL *) GRID->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_add_gpu<<<blocks2,threads>>>((CUREAL *) GRID->data[i], c, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = cm * dst + ca
 *
 */

__global__ void rgrid_cuda_multiply_and_add_gpu(CUREAL *dst, CUREAL cm, CUREAL ca, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = dst[idx] * cm + ca;
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_and_addW(gpu_mem_block *dst, CUREAL cm, REAL ca, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): multiply_and_add must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_and_add_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], cm, ca, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_and_add_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], cm, ca, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = cm * (dst + ca)
 *
 */

__global__ void rgrid_cuda_add_and_multiply_gpu(CUREAL *dst, CUREAL ca, CUREAL cm, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = (dst[idx] + ca) * cm;
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (gpu_mem_block *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_and_multiplyW(gpu_mem_block *dst, CUREAL ca, CUREAL cm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): add_and_multiply must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_and_multiply_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], ca, cm, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_and_multiply_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], ca, cm, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = dst + d * src
 *
 */

__global__ void rgrid_cuda_add_scaled_gpu(CUREAL *dst, CUREAL d, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = dst[idx] + src[idx] * d;
}

/*
 * Scaled add grid.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * d        = Scaling factor (REAL; input).
 * src      = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaledW(gpu_mem_block *dst, CUREAL d, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): add_scaled must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_scaled_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], d, (CUREAL *) SRC->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_scaled_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], d, (CUREAL *) SRC->data[i], nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = dst + d * src1 * src2
 *
 */

__global__ void rgrid_cuda_add_scaled_product_gpu(CUREAL *dst, CUREAL d, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = dst[idx] + d * src1[idx] * src2[idx];
}

/*
 * Add scaled product.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * d        = Scaling factor (REAL; input).
 * src1     = Source for operation (gpu_mem_block *; input).
 * src2     = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaled_productW(gpu_mem_block *dst, CUREAL d, gpu_mem_block *src1, gpu_mem_block *src2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC1 = src1->gpu_info->descriptor, *SRC2 = src2->gpu_info->descriptor;

  if(src1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || src2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): add_scaled_product must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_scaled_product_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], d, (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_add_scaled_product_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], d, (CUREAL *) SRC1->data[i], (CUREAL *) SRC2->data[i], nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = c
 *
 */

__global__ void rgrid_cuda_constant_gpu(CUREAL *dst, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = c;
}

/*
 * Set grid to constant.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * c        = Constant (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_constantW(gpu_mem_block *dst, CUREAL c, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_constant_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_constant_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], c, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * Block init (zero elements).
 *
 * blocks  = Block table (CUREAL *; output).
 * nblocks = Number of blocks in table (INT; input).
 * 
 */

__global__ void rgrid_cuda_block_init(CUREAL *blocks, INT nblocks) {

  INT i;

  for(i = 0; i < nblocks; i++) blocks[i] = 0.0;
}

/*
 * Block reduction.
 *
 * blocks  = Block list to reduce (CUREAL *; input/output). blocks[0] will contain the reduced value.
 * nblocks = Number of blocks (INT; input).
 *
 */

__global__ void rgrid_cuda_block_reduce(CUREAL *blocks, INT nblocks) {

  INT i;

  for(i = 1; i < nblocks; i++)
    blocks[0] += blocks[i];  // reduce blocks
}

/*
 * Integrate over A.
 *
 */

__global__ void rgrid_cuda_integral_gpu(CUREAL *a, CUREAL *blocks, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nzz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

  els[idx2] += a[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
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

extern "C" void rgrid_cuda_integralW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUREAL tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): integral must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  cuda_error_check();
}

/*
 * Integrate over A with limits.
 *
 */

__global__ void rgrid_cuda_integral_region_gpu(CUREAL *a, CUREAL *blocks, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, INT nzz, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, ii = i + seg;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  if(ii >= il && ii <= iu && j >= jl && j <= ju && k >= kl && k <= ku) {
    idx = (i * ny + j) * nzz + k;
    idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    els[idx2] += a[idx];
  }
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
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
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void rgrid_cuda_integral_regionW(gpu_mem_block *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUREAL tmp;
  INT seg = 0, s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z;
  INT b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): integral_region must be in real space (INPLACE).");
    abort();
  }

  if(il < 0) il = 0;  
  if(jl < 0) jl = 0;  
  if(kl < 0) kl = 0;  
  if(iu > nx-1) iu = nx-1;
  if(ju > ny-1) ju = ny-1;
  if(ku > nz-1) ku = nz-1;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                                    il, iu, jl, ju, kl, ku, nnx1, ny, nz, nzz, seg);
    seg += nnx1;
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_region_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                                    il, iu, jl, ju, kl, ku, nnx2, ny, nz, nzz, seg);
    seg += nnx2;
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }
  cuda_error_check();
}

/*
 * Integrate of A^2.
 *
 */

__global__ void rgrid_cuda_integral_of_square_gpu(CUREAL *a, CUREAL *blocks, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nzz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

  els[idx2] += a[idx] * a[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
    }
  }
}

/*
 * Integral of square.
 *
 * grid     = Source for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void rgrid_cuda_integral_of_squareW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUREAL tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): integral_of_square must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_of_square_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_of_square_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  cuda_error_check();
}

/*
 * Integrate A * B.
 *
 */

__global__ void rgrid_cuda_integral_of_product_gpu(CUREAL *a, CUREAL *b, CUREAL *blocks, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nzz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

  els[idx2] += a[idx] * b[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
    }
  }
}

/*
 * Integral of product.
 *
 * grid1    = Source 1 for operation (gpu_mem_block *; input).
 * grid2    = Source 2 for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Return value (CUREAL *; output).
 *
 */

extern "C" void rgrid_cuda_integral_of_productW(gpu_mem_block *grid1, gpu_mem_block *grid2, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(grid1);
  cudaXtDesc *GRID1 = grid1->gpu_info->descriptor, *GRID2 = grid2->gpu_info->descriptor;
  CUREAL tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(grid1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || grid2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): integral_of_product must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID1->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_of_product_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID1->data[i], (CUREAL *) GRID2->data[i], 
                                                                             (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID1->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_of_product_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID1->data[i], (CUREAL *) GRID2->data[i], 
                                                                             (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  cuda_error_check();
}

/*
 * Integrate A * B^2.
 *
 */

__global__ void rgrid_cuda_grid_expectation_value_gpu(CUREAL *a, CUREAL *b, CUREAL *blocks, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nzz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

  els[idx2] += a[idx] * b[idx] * b[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
    }
  }
}

/*
 * Integral a * b^2.
 *
 * grid1    = Source 1 (a) for operation (gpu_mem_block *; input).
 * grid2    = Source 2 (b) for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * *value   = Return value (CUREAL *; output).
 *
 */

extern "C" void rgrid_cuda_grid_expectation_valueW(gpu_mem_block *grid1, gpu_mem_block *grid2, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(grid1);
  cudaXtDesc *GRID1 = grid1->gpu_info->descriptor, *GRID2 = grid2->gpu_info->descriptor;
  CUREAL tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(grid1->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || grid2->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): grid_expectation_value must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID1->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_grid_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID1->data[i], (CUREAL *) GRID2->data[i], 
                                                                                (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID1->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_grid_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) GRID1->data[i], (CUREAL *) GRID2->data[i], 
                                                                                (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  cuda_error_check();
}

/*
 * B = FD_X(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_x_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_X(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_x must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = FD_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_y_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Y(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_y must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = FD_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_z_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Z(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_z must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = LAPLACE(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       - 6.0 * src[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta2= 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * Returns the value of integral.
 *
 */

extern "C" void rgrid_cuda_fd_laplaceW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = LAPLACE_X(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_x_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_X(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_x must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = LAPLACE_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_y_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
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
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_y must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = LAPLACE_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_z_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_Z(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_z must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 */

__global__ void rgrid_cuda_fd_gradient_dot_gradient_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = 0.0;

  tmp = inv_delta * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz));
  dst[idx] = dst[idx] + tmp * tmp;

  tmp = inv_delta * (rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz));
  dst[idx] = dst[idx] + tmp * tmp;

  tmp = inv_delta * (rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz));
  dst[idx] = dst[idx] + tmp * tmp;
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (4 * step * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_dot_gradientW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_dot_gradient must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_dot_gradient_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * Maximum value in a grid.
 *
 */

__global__ void rgrid_cuda_max_gpu(CUREAL *a, CUREAL *val, INT nx, INT ny, INT nz, INT nzz) {

  /* blockIdx.x = i, threadIdx.x = j */
  INT i, j, k, idx;

  *val = a[0];
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for(k = 0; k < nz; k++) {
        idx = (i * ny + j) * nzz + k;
        if(a[idx] > *val) *val = a[idx];
      }
}

/*
 * Maximum value contained in a grid. (not parallel)
 *
 * grid    = Source for operation (gpu_mem_block complex *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 * value   = Return value (CUREAL *; output).
 *
 */

extern "C" void grid_cuda_maxW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUREAL *value) {

  INT i, ngpu2 = grid->gpu_info->descriptor->nGPUs, ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);  CUREAL tmp;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): max must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_max_gpu<<<1,1>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_max_gpu<<<1,1>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
  }

  // Reduce over GPUs
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    if(i == 0) *value = tmp;
    else if(tmp > *value) *value = tmp;    
  }

  cuda_error_check();
}

/*
 * Minimum value in a grid.
 *
 */

__global__ void rgrid_cuda_min_gpu(CUREAL *a, CUREAL *val, INT nx, INT ny, INT nz, INT nzz) {

  /* blockIdx.x = i, threadIdx.x = j */
  INT i, j, k, idx;

  *val = a[0];
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for(k = 0; k < nz; k++) {
        idx = (i * ny + j) * nzz + k;
        if(a[idx] < *val) *val = a[idx];
      }
}

/*
 * Minimum value contained in a grid. (not parallel)
 *
 * grid    = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 * value   = Return value (CUREAL *; output).
 *
 */

extern "C" void grid_cuda_minW(gpu_mem_block *grid, INT nx, INT ny, INT nz, CUREAL *value) {

  INT i, ngpu2 = grid->gpu_info->descriptor->nGPUs, ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  CUREAL tmp;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): min must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_min_gpu<<<1,1>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_min_gpu<<<1,1>>>((CUREAL *) GRID->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz, nzz);
  }

  // Reduce over GPUs
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    if(i == 0) *value = tmp;
    else if(tmp > *value) *value = tmp;    
  }

  cuda_error_check();
}

/*
 * |rot|
 *
 */

__global__ void rgrid_cuda_abs_rot_gpu(CUREAL *rot, CUREAL *fx, CUREAL *fy, CUREAL *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) { 
 
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  /* x: (d/dy) fz - (d/dz) fy */
  tmp = inv_delta * (rgrid_cuda_bc_y_plus(fz, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(fz, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_z_plus(fy, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(fy, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = tmp * tmp;

  /* y: (d/dz) fx - (d/dx) fz */
  tmp = inv_delta * (rgrid_cuda_bc_z_plus(fx, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(fx, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_x_plus(fz, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(fz, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = rot[idx] + tmp * tmp;

  /* z: (d/dx) fy - (d/dy) fx */
  tmp = inv_delta * (rgrid_cuda_bc_x_plus(fy, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(fy, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_y_plus(fx, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(fx, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = rot[idx] + tmp * tmp;
  rot[idx] = SQRT(rot[idx]);
}

/*
 * |rot|
 *
 * rot       = Grid to be operated on (gpu_mem_block *; input/output).
 * fx        = x component of the field (gpu_mem_block *; input).
 * fy        = y component of the field (gpu_mem_block *; input).
 * fz        = z component of the field (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * TODO: For this it probably makes sense to force transferring the blocks to host memory and do the operation there.
 *
 */

extern "C" void rgrid_cuda_abs_rotW(gpu_mem_block *rot, gpu_mem_block *fx, gpu_mem_block *fy, gpu_mem_block *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *ROT = rot->gpu_info->descriptor, *FX = fx->gpu_info->descriptor, *FY = fy->gpu_info->descriptor, *FZ = fz->gpu_info->descriptor;

  if(fx->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || fy->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || fz->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): abs_rot must be in real space (INPLACE).");
    abort();
  }

  if(ROT->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(ROT->GPUs[0]);
  rgrid_cuda_abs_rot_gpu<<<blocks,threads>>>((CUREAL *) ROT->data[0], (CUREAL *) FX->data[0], (CUREAL *) FY->data[0], (CUREAL *) FZ->data[0], 
                                             inv_delta, bc, nx, ny, nz, nzz);

  rot->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;

  cuda_error_check();
}

/*
 * dst = POW(src, n) with n integer.
 *
 */

__global__ void rgrid_cuda_ipower_gpu(CUREAL *dst, CUREAL *src, INT n, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii, sig;
  CUREAL value = 1.0;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  if(n == 0) {
    dst[idx] = 1.0;
    return;
  }
  sig = (n < 0) ? -1:1;
  n = ABS(n);
  switch(n) {
    case 1:      
      dst[idx] = src[idx];
      break;
    case 2:
      dst[idx] = src[idx] * src[idx];
      break;
    case 3:
      dst[idx] = src[idx] * src[idx] * src[idx];
      break;
    default:
      for(ii = 0; ii < n; ii++)
        value *= src[idx];
      dst[idx] = value;
  }
  if(sig == -1) dst[idx] = 1.0 / dst[idx];
}

/*
 * Grid integer power.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * src      = Source for operation (gpu_mem_block *; input).
 * exponent = Integer exponent (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_ipowerW(gpu_mem_block *dst, gpu_mem_block *src, INT exponent, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): ipower must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_ipower_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_ipower_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], exponent, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * Grid threshold clear device code.
 *
 */

__global__ void rgrid_cuda_threshold_clear_gpu(CUREAL *dest, CUREAL *src, CUREAL ul, CUREAL ll, CUREAL uval, CUREAL lval, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  if(src[idx] < ll) dest[idx] = lval;
  if(src[idx] > ul) dest[idx] = uval;
}

/*
 * Grid clear based on threshold.
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * ul      = upper limit threshold for the operation (REAL; input).
 * ll      = lower limit threshold for the operation (REAL; input).
 * uval    = value to set when the upper limit was exceeded (REAL; input).
 * lval    = value to set when the lower limit was exceeded (REAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_threshold_clearW(gpu_mem_block *dst, gpu_mem_block *src, CUREAL ul, CUREAL ll, CUREAL uval, CUREAL lval, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): threshold_clear must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_threshold_clear_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], ul, ll, uval, lval, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_threshold_clear_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUREAL *) SRC->data[i], ul, ll, uval, lval, nnx2, ny, nz, nzz);
  }

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * Zero part of real grid.
 *
 * A = 0 in the specified range.
 *
 */

__global__ void rgrid_cuda_zero_index_gpu(CUREAL *dst, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT nzz, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii = i + seg;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  if(ii >= lx && ii < hx && j >= ly && j < hy && k >= lz && k < hz) dst[idx] = 0.0;
}

/*
 * Zero specified range of complex grid.
 *
 * grid     = Grid to be operated on (gpu_mem_block *; input/output).
 * lx       = Low x index (INT; input). 
 * hx       = Low x index (INT; input). 
 * ly       = Low y index (INT; input). 
 * hy       = Low y index (INT; input). 
 * lz       = Low z index (INT; input). 
 * hx       = Low z index (INT; input). 
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_zero_indexW(gpu_mem_block *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  INT seg = 0;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): clear_index must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_zero_index_gpu<<<blocks1,threads>>>((CUREAL *) GRID->data[i], lx, hx, ly, hy, lz, hz, nnx1, ny, nz, nzz, seg);
    seg += nnx1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_zero_index_gpu<<<blocks2,threads>>>((CUREAL *) GRID->data[i], lx, hx, ly, hy, lz, hz, nnx2, ny, nz, nzz, seg);
    seg += nnx2;
  }

  cuda_error_check();
}

/*
 * Poisson equation.
 *
 */

__global__ void rgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nzz, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nzz) return;

  idx = (i * ny + j) * nzz + k;
  kx = COS(ilx * (CUREAL) i);
  ky = COS(ily * (CUREAL) jj);
  kz = COS(ilz * (CUREAL) k);
  if(i || jj || k)
    grid[idx] = grid[idx] * norm * step2 / (2.0 * (kx + ky + kz - 3.0));
  else
    grid[idx] = CUMAKE(0.0,0.0);
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
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_poissonW(gpu_mem_block *grid, CUREAL norm, CUREAL step2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  INT seg = 0;
  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = M_PI / ((CUREAL) nzz);

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): poisson must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_poisson_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], norm, step2, ilx, ily, ilz, nx, nny1, nzz, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_poisson_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], norm, step2, ilx, ily, ilz, nx, nny2, nzz, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT gradient (x).
 *
 */

__global__ void rgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *gradient, REAL kx0, REAL step, REAL norm, REAL lx, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kx * norm);
}

/*
 * Gradient of grid in Fourier space (X).
 *
 * gradient_x = Source & destination for operation (gpu_mem_block *; input/output).
 * kx0        = Baseline momentum (grid->kx0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_xW(gpu_mem_block *gradient_x, REAL kx0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_x);
  cudaXtDesc *GRADIENT_X = gradient_x->gpu_info->descriptor;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step);

  if(gradient_x->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_x must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], kx0, step, norm, lx, nx, nny1, nzz, nx / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], kx0, step, norm, lx, nx, nny2, nzz, nx / 2);
  }

  cuda_error_check();
}

/*
 * FFT gradient (y).
 *
 */

__global__ void rgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *gradient, REAL ky0, REAL step, REAL norm, REAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  REAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, ky * norm);
}

/*
 * Gradient of grid in Fourier space (Y).
 *
 * gradient_y = Source & destination for operation (gpu_mem_block *; input/output).
 * ky0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_yW(gpu_mem_block *gradient_y, REAL ky0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_y);
  cudaXtDesc *GRADIENT_Y = gradient_y->gpu_info->descriptor;
  INT seg = 0;
  REAL ly = 2.0 * M_PI / (((REAL) ny) * step);

  if(gradient_y->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_y must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_Y->GPUs[i]);
    rgrid_cuda_fft_gradient_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], ky0, step, norm, ly, nx, nny1, nzz, ny, ny / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Y->GPUs[i]);
    rgrid_cuda_fft_gradient_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], ky0, step, norm, ly, nx, nny2, nzz, ny, ny / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT gradient (z).
 *
 */

__global__ void rgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *gradient, REAL kz0, REAL step, REAL norm, REAL lz, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kz * norm);
}

/*
 * Gradient of grid in Fourier space (Z).
 *
 * gradient_z = Source & destination for operation (gpu_mem_block *; input/output).
 * kz0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_zW(gpu_mem_block *gradient_z, REAL kz0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_z);
  cudaXtDesc *GRADIENT_Z = gradient_z->gpu_info->descriptor;
  REAL lz = M_PI / (((REAL) nzz - 1) * step);

  if(gradient_z->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_z must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], kz0, step, norm, lz, nx, nny1, nzz, nzz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], kz0, step, norm, lz, nx, nny2, nzz, nzz / 2);
  }

  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void rgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  b[idx] = b[idx] * (-(kx * kx + ky * ky + kz * kz) * norm);
}

/*
 * FFT laplace
 *
 * laplace  = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Only periodic boundaries!
 *
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_fft_laplaceW(gpu_mem_block *laplace, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace);
  cudaXtDesc *LAPLACE = laplace->gpu_info->descriptor;
  INT seg = 0;

  if(laplace->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE->data[i], norm, kx0, ky0, kz0, 
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE->data[i], norm, kx0, ky0, kz0,
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny2;
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
 * Normalization done in rgrid-cuda.c
 *
 */

__global__ void rgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUREAL *blocks, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

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

  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els2[idx2] -= (kx * kx + ky * ky + kz * kz) * (CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]));
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els2[t];  // reduce threads
    }
  }
}

/*
 * FFT laplace expectation value
 *
 * laplace  = Source/destination grid for operation (gpu_mem_block *; input/output).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * sum      = Expectation value (REAL; output).
 *
 * Only periodic boundaries!  In Fourier space.
 *
 * Normalization done in rgrid-cuda.c
 *
 */

extern "C" void rgrid_cuda_fft_laplace_expectation_valueW(gpu_mem_block *laplace, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_RECIPROCAL(laplace);
  cudaXtDesc *LAPLACE = laplace->gpu_info->descriptor;
  CUREAL tmp;
  INT seg = 0, s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(laplace->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_expectation_value must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) LAPLACE->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                               kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
                               step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) LAPLACE->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                               kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
                               step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    seg += nny2;
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value += tmp;
  }

  cuda_error_check();
}
