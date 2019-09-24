/*
 * CUDA device code (mixed cgrid/rgrid).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
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

extern void *grid_gpu_mem;
extern cudaXtDesc *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Real to complex_re.
 *
 * dst.re = src(real). (zeroes the imag part)
 *
 */

__global__ void grid_cuda_real_to_complex_re_gpu(CUCOMPLEX *dst, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;    // Index for complex grid
  idx2 = (i * ny + j) * nzz + k;  // Index for real grid

  dst[idx] = CUMAKE(src[idx2], 0.0);
}

/*
 * Real to complex_re
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_real_to_complex_reW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_real_to_complex_re_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_real_to_complex_re_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Real to complex_im.
 *
 * dst.im = src(real). (zeroes the real part)
 *
 */

__global__ void grid_cuda_real_to_complex_im_gpu(CUCOMPLEX *dst, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nzz + k;

  dst[idx] = CUMAKE(0.0, src[idx2]);
}

/*
 * Real to complex_im
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_real_to_complex_imW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_real_to_complex_im_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_real_to_complex_im_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Add real to complex_re.
 *
 * dst.re = dst.re + src(real).
 *
 */

__global__ void grid_cuda_add_real_to_complex_re_gpu(CUCOMPLEX *dst, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  dst[idx].x = CUCREAL(dst[idx]) + src[idx2];
}

/*
 * Add real to complex.re
 *
 * grida   = Destination for operation (gpu_mem_block *; output).
 * gridb   = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_add_real_to_complex_reW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  if(src->gpu_info->subFormat != dst->gpu_info->subFormat) {
    fprintf(stderr, "libgrid(cuda): add_real_to_complex_re source/destination must have the same subformat.");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_add_real_to_complex_re_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_add_real_to_complex_re_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Add real to complex_im.
 *
 * dst.im = dst.im + src(real).
 *
 */

__global__ void grid_cuda_add_real_to_complex_im_gpu(CUCOMPLEX *dst, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  dst[idx].y = CUCIMAG(dst[idx]) + src[idx2];
}

/*
 * Add real to complex_im
 *
 * grida   = Destination for operation (gpu_mem_block *; output).
 * gridb   = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_add_real_to_complex_imW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  if(src->gpu_info->subFormat != dst->gpu_info->subFormat) {
    fprintf(stderr, "libgrid(cuda): add_real_to_complex_im source/destination must have the same subformat.");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_add_real_to_complex_im_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_add_real_to_complex_im_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Product dst(complex) and src(real).
 *
 * dst = dst * src(real).
 *
 */

__global__ void grid_cuda_product_complex_with_real_gpu(CUCOMPLEX *dst, CUREAL *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  dst[idx] = dst[idx] * src[idx2];
}

/*
 * Product dst(complex) with src(real).
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_product_complex_with_realW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  if(src->gpu_info->subFormat != dst->gpu_info->subFormat) {
    fprintf(stderr, "libgrid(cuda): product_complex_with_real source/destination must have the same subformat.");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_product_complex_with_real_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_product_complex_with_real_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUREAL *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Imag. part to real grid.
 *
 * dst(real) = src.im;
 *
 */

__global__ void grid_cuda_complex_im_to_real_gpu(CUREAL *dst, CUCOMPLEX *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  dst[idx2] = CUCIMAG(src[idx]);
}

/*
 * Imag. part of src to real dst.
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_complex_im_to_realW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_complex_im_to_real_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_complex_im_to_real_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}

/*
 * Real part to real grid.
 *
 * dst(real) = src.re;
 *
 */

__global__ void grid_cuda_complex_re_to_real_gpu(CUREAL *dst, CUCOMPLEX *src, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  dst[idx2] = CUCREAL(src[idx]);
}

/*
 * Real part of B to real A.
 *
 * dst     = Destination for operation (gpu_mem_block *; output).
 * src     = Source for operation (gpu_mem_block *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_complex_re_to_realW(gpu_mem_block *dst, gpu_mem_block *src, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;
  INT nzz = 2 * (nz / 2 + 1);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_complex_re_to_real_gpu<<<blocks1,threads>>>((CUREAL *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx1, nny1, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_complex_re_to_real_gpu<<<blocks2,threads>>>((CUREAL *) DST->data[i], (CUCOMPLEX *) SRC->data[i], nnx2, nny2, nz, nzz);
  }

  cuda_error_check();
}
