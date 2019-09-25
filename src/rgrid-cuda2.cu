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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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
 * Integrate opgrid * dgrid^2.
 *
 */

__global__ void rgrid_cuda_grid_expectation_value_gpu(CUREAL *dgrid, CUREAL *opgrid, CUREAL *blocks, INT nx, INT ny, INT nz, INT nzz) {

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

  els[idx2] += opgrid[idx] * dgrid[idx] * dgrid[idx];
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els[t];  // reduce threads
    }
  }
}

/*
 * Integral opgrid * dgrid^2
 *
 * dgrid    = Source 1 for operation (gpu_mem_block *; input).
 * opgrid   = Source 2 for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * *value   = Return value (CUREAL *; output).
 *
 */

extern "C" void rgrid_cuda_grid_expectation_valueW(gpu_mem_block *dgrid, gpu_mem_block *opgrid, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_REAL(dgrid);
  cudaXtDesc *DGRID = dgrid->gpu_info->descriptor, *OPGRID = opgrid->gpu_info->descriptor;
  CUREAL tmp;
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(dgrid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || opgrid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): grid_expectation_value grids must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DGRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_grid_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUREAL *) DGRID->data[i], (CUREAL *) OPGRID->data[i], 
                                                                                (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz, nzz);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DGRID->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_grid_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUREAL *) DGRID->data[i], (CUREAL *) OPGRID->data[i], 
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

  rot->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
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
 *
 * dst = dst * x
 *
 */

__global__ void rgrid_cuda_multiply_by_x_gpu(CUREAL *dst, CUREAL x0, INT nx, INT ny, INT nz, INT nzz, CUREAL step, INT nx2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii = i + seg;
  CUREAL x;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  x = ((CUREAL) (ii - nx2)) * step - x0;
  dst[idx] = dst[idx] * x;
}

/*
 * Multiply grid by x.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * x0       = Origin x (CUREAL; input).
 * step     = Step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_multiply_by_xW(gpu_mem_block *dst, CUREAL x0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT seg = 0;

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): multiply_by_x must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_x_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], x0, nnx1, ny, nz, nzz, step, nx / 2, seg);
    seg += nnx1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_x_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], x0, nnx2, ny, nz, nzz, step, nx / 2, seg);
    seg += nnx2;
  }

  cuda_error_check();
}

/*
 *
 * dst = dst * y
 *
 */

__global__ void rgrid_cuda_multiply_by_y_gpu(CUREAL *dst, CUREAL y0, INT nx, INT ny, INT nz, INT nzz, CUREAL step, INT ny2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL y;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  y = ((CUREAL) (j - ny2)) * step - y0;
  dst[idx] = dst[idx] * y;
}

/*
 * Multiply grid by y.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * y0       = Origin y (CUREAL; input).
 * step     = Step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_multiply_by_yW(gpu_mem_block *dst, CUREAL y0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): multiply_by_y must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_y_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], y0, nnx1, ny, nz, nzz, step, ny / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_y_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], y0, nnx2, ny, nz, nzz, step, ny / 2);
  }

  cuda_error_check();
}

/*
 *
 * dst = dst * z
 *
 */

__global__ void rgrid_cuda_multiply_by_z_gpu(CUREAL *dst, CUREAL z0, INT nx, INT ny, INT nz, INT nzz, CUREAL step, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  z = ((CUREAL) (k - nz2)) * step - z0;
  dst[idx] = dst[idx] * z;
}

/*
 * Multiply grid by z.
 *
 * dst      = Destination for operation (gpu_mem_block *; output).
 * z0       = Origin z (CUREAL; input).
 * step     = Step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_multiply_by_zW(gpu_mem_block *dst, CUREAL z0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): multiply_by_z must be in real space (INPLACE).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_z_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], z0, nnx1, ny, nz, nzz, step, nz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    rgrid_cuda_multiply_by_z_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], z0, nnx2, ny, nz, nzz, step, nz / 2);
  }

  cuda_error_check();
}
