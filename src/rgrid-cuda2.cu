/*
 * CUDA device code (REAL; rgrid).
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cufft.h>
#include "cuda-math.h"
#include "rgrid_bc-cuda.h"

extern cudaXtState *grid_gpu_mem_addr;
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
 * dst   = output (cudaXtState *; output).
 * src1  = 1st grid to be convoluted (cudaXtState *; input).
 * src2  = 2nd grid to be convoluted (cudaXtState *; input).
 * norm  = FFT norm (CUREAL; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_fft_convoluteW(cudaXtState *dst, cudaXtState *src1, cudaXtState *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = ny % gpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1, nzz = nz / 2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets 
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_fft_convolute_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], norm, nx, nny1, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_fft_convolute_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLES *) src2->data[i], norm, nx, nny2, nzz);
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
 * dst      = Destination for operation (cudaXtState *; output).
 * src      = Source for operation (cudaXtState *; input).
 * exponent = Exponent (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Real space.
 *
 */

extern "C" void rgrid_cuda_powerW(cudaXtState *dst, cudaXtState *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_power_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src->data[i], exponent, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_power_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src->data[i], exponent, nnx2, ny, nz, nzz);
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
 * dst      = Destination for operation (cudaXtState *; output).
 * src      = Source for operation (cudaXtState *; input).
 * exponent = Exponent (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_abs_powerW(cudaXtState *dst, cudaXtState *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_abs_power_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src->data[i], exponent, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_abs_power_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src->data[i], exponent, nnx2, ny, nz, nzz);
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
 * dst      = Grid to be operated on (cudaXtState *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiplyW(cudaXtState *grid, CUREAL c, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_multiply_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], c, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = c * dst (in Fourier space)
 *
 */

__global__ void rgrid_cuda_multiply_fft_gpu(CUCOMPLEX *dst, CUREAL c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx] = dst[idx] * c;
}

/*
 * Multiply (complex) grid by a constant (in FFT space).
 *
 * st       = Grid to be operated on (cudaXtState *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_fftW(CUCOMPLEX *grid, CUREAL c, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = ny % gpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1, nzz = nz / 2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets 
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_multiply_fft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], c, nx, nny1, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_multiply_fft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], c, nx, nny2, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = src1 + src2
 *
 */

__global__ void rgrid_cuda_sum_gpu(CUREAL *dst, CUREAL *src1, CUREAL *src2, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = src1[idx] + src2[idx];
}

/*
 * Sum of two grids.
 *
 * dst      = Destination grid (cudaXtState *; output).
 * src1     = Input grid 1 (cudaXtState *; input).
 * src2     = Input grid 2 (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_sumW(cudaXtState *grida, cudaXtState *gridb, cudaXtState *gridc, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_sum_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_sum_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx2, ny, nz, nzz);
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
 * dst      = Destination grid (cudaXtState *; output).
 * src1     = Input grid 1 (cudaXtState *; input).
 * src2     = Input grid 2 (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_differenceW(cudaXtState *dst, cudaXtState *src1, cudaXtState *src2, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_difference_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_difference_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx2, ny, nz, nzz);
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
 * dst      = Destination grid (cudaXtState *; output).
 * src1     = Source grid 1 (cudaXtState *; input).
 * src2     = Source grid 2 (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_productW(cudaXtState *dst, cudaXtState *src1, cudaXtState *src2, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_product_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_product_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx2, ny, nz, nzz);
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
 * dst      = Destination grid (cudaXtState *; output).
 * src1     = Source grid 1 (cudaXtState *; input).
 * src2     = Source grid 2 (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_divisionW(cudaXtState *dst, cudaXtState *src1, cudaXtState *src2, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_division_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx2, ny, nz, nzz);
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
 * dst      = Destination grid (cudaXtState *; output).
 * src1     = Source grid 1 (cudaXtState *; input).
 * src2     = Source grid 2 (cudaXtState *; input).
 * eps      = Epsilon (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_division_epsW(cudaXtState *dst, cudaXtState *src1, cudaXtState *src2, CUREAL eps, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_division_eps_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], eps, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_eps_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], eps, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = dst + c
 *
 */

__global__ void rgrid_cuda_add_gpu(CUREAL *dst, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] += c;
}

/*
 * Add constant to grid.
 *
 * dst      = Grid to be operated on (cudaXtState *; input/output).
 * c        = Constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_addW(cudaXtState *grid, CUREAL c, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_add_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], c, nnx2, ny, nz, nzz);
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
 * dst      = Grid to be operated on (cudaXtState *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_and_addW(cudaXtState *grid, CUREAL cm, REAL ca, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_multiply_and_add_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], cm, ca, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_and_add_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], cm, ca, nnx2, ny, nz, nzz);
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
 * dst      = Grid to be operated on (cudaXtState *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_and_multiplyW(cudaXtState *dst, CUREAL ca, CUREAL cm, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_add_and_multiply_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], ca, cm, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_and_multiply_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], ca, cm, nnx2, ny, nz, nzz);
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
 * dst      = Destination for operation (cudaXtState *; output).
 * d        = Scaling factor (REAL; input).
 * src      = Source for operation (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaledW(cudaXtState *dst, CUREAL d, cudaXtState *src, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_add_scaled_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], d, (CUREAL *) src->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], d, (CUREAL *) src->data[i], nnx2, ny, nz, nzz);
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
 * dst      = Destination for operation (cudaXtState *; output).
 * d        = Scaling factor (REAL; input).
 * src1     = Source for operation (cudaXtState *; input).
 * src2     = Source for operation (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaled_productW(cudaXtState *dst, CUREAL d, cudaXtState *src1, cudaXtState *src2, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_add_scaled_product_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], d, (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_product_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], d, (CUREAL *) src1->data[i], (CUREAL *) src2->data[i], nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/*
 *
 * dst = c
 *
 */

__global__ void rgrid_cuda_constant_gpu(CUREAL *dst, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = c;
}

/*
 * Set grid to constant.
 *
 * dst      = Destination for operation (cudaXtState *; output).
 * c        = Constant (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_constantW(cudaXtState *dst, CUREAL c, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_constant_gpu<<<blocks1,threads>>>((CUREAL *) dst->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_constant_gpu<<<blocks2,threads>>>((CUREAL *) dst->data[i], c, nnx2, ny, nz, nzz);
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

/// LEFT HERE - problems with with block reduce (1st overwritten)

/*
 * Integrate over grid.
 *
 * grid     = Source for operation (cudaXtState *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Result (CUCOMPLEX *; output).
 *
 */

extern "C" void rgrid_cuda_integralW(cudaXtState *grid, INT nx, INT ny, INT nz, CUREAL *value) {

  CUREAL tmp;
  INT i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>(CUREAL *) grid->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu1; i++) {
    cuda_get_element(grid_gpu_mem, grid->GPUs[i], 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>(CUREAL *) grid->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], nnx2, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    cuda_error_check();
  }

  // Reduce over GPUs
  for(i = ngpu1; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, grid->GPUs[i], 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }
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
 * grid     = Source for operation (cudaXtState *; input).
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

extern "C" void rgrid_cuda_integral_regionW(cudaXtState *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, CUREAL *value) {

  CUREAL tmp;
  INT i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, seg = 0;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  INT s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  if(il < 0) il = 0;  
  if(jl < 0) jl = 0;  
  if(kl < 0) kl = 0;  
  if(iu > nx-1) iu = nx-1;
  if(ju > ny-1) ju = ny-1;
  if(ku > nz-1) ku = nz-1;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>(CUREAL *) grid->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                                    il, iu, jl, ju, kl, ku, nnx1, ny, nz, seg);
    seg -= nnx1;
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
  }


  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu1; i++) {
    cuda_get_element(grid_gpu_mem, grid->GPUs[i], 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>(CUREAL *) grid->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                                    il, iu, jl, ju, kl, ku, nnx2, ny, nz, seg);
    seg -= nnx2;
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    cuda_error_check();
  }

  // Reduce over GPUs
  for(i = ngpu1; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, grid->GPUs[i], 0, sizeof(CUREAL), &tmp);
    *value = *value + tmp;
  }
}

//// LEFT HERE

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
 * grid     = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void rgrid_cuda_integral_of_squareW(CUREAL *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  rgrid_cuda_integral_of_square_gpu<<<blocks,threads,s*sizeof(REAL)>>>(grid, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
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
 * grid1    = Source 1 for operation (REAL *; input).
 * grid2    = Source 2 for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void rgrid_cuda_integral_of_productW(CUREAL *grid1, CUREAL *grid2, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  // Blocks, Threads, dynamic memory size
  rgrid_cuda_integral_of_product_gpu<<<blocks,threads,s*sizeof(REAL)>>>(grid1, grid2, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
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
 * grid1    = Source 1 (a) for operation (REAL *; input).
 * grid2    = Source 2 (b) for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void rgrid_cuda_grid_expectation_valueW(CUREAL *grid1, CUREAL *grid2, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  rgrid_cuda_grid_expectation_value_gpu<<<blocks,threads,s*sizeof(REAL)>>>(grid1, grid2, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
}

/*
 * B = FD_X(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_x_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta * (rgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_X(A)
 *
 * grida    = Source for operation (REAL *; input).
 * gridb    = Destination for operation (REAL *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_xW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = FD_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_y_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta * (rgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Y(A)
 *
 * grida    = Source for operation (REAL *; input).
 * gridb    = Destination for operation (REAL *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_yW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = FD_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_z_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta * (rgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Z(A)
 *
 * grida    = Source for operation (REAL *; input).
 * gridb    = Destination for operation (REAL *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_zW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = LAPLACE(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                       - 6.0 * a[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * grid1      = Source 1 (a) for operation (REAL *; input).
 * grid2      = Source 2 (b) for operation (REAL *; input).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns the value of integral.
 *
 */

extern "C" void rgrid_cuda_fd_laplaceW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = LAPLACE_X(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_x_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_X(A)
 *
 * grida      = Source for operation (REAL *; input).
 * gridb      = Destination for operation (REAL *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_xW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = LAPLACE_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_y_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta2 * (rgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_Y(A)
 *
 * grida      = Source for operation (REAL *; input).
 * gridb      = Destination for operation (REAL *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_yW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = LAPLACE_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_z_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = inv_delta2 * (rgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_Z(A)
 *
 * grida      = Source for operation (REAL *; input).
 * gridb      = Destination for operation (REAL *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_zW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 */

__global__ void rgrid_cuda_fd_gradient_dot_gradient_gpu(CUREAL *a, CUREAL *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  b[idx] = 0.0;

  tmp = inv_delta * (rgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz, nzz));
  b[idx] = b[idx] + tmp * tmp;

  tmp = inv_delta * (rgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz, nzz));
  b[idx] = b[idx] + tmp * tmp;

  tmp = inv_delta * (rgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz, nzz));
  b[idx] = b[idx] + tmp * tmp;
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 * grida      = Source for operation (REAL *; input).
 * gridb      = Destination for operation (REAL *; output).
 * inv_delta2 = 1 / (4 * step * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_dot_gradientW(CUREAL *grida, CUREAL *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fd_gradient_dot_gradient_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Maximum value in a grid.
 *
 */

__global__ void grid_cuda_max_gpu(CUREAL *a, CUREAL *val, INT nx, INT ny, INT nz, INT nzz) {

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
 * grid    = Source for operation (REAL complex *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 * Returns maximum value in grid_gpu_mem[0].
 *
 */

extern "C" void grid_cuda_maxW(CUREAL *grid, INT nx, INT ny, INT nz) {

  grid_cuda_max_gpu<<<1,1>>>(grid, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Minimum value in a grid.
 *
 */

__global__ void grid_cuda_min_gpu(CUREAL *a, CUREAL *val, INT nx, INT ny, INT nz, INT nzz) {

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
 * grid    = Source for operation (REAL complex *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 * Returns minimum value in grid_gpu_mem[0].
 *
 */

extern "C" void grid_cuda_minW(CUREAL *grid, INT nx, INT ny, INT nz) {

  grid_cuda_min_gpu<<<1,1>>>(grid, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * |rot|
 *
 */

__global__ void rgrid_cuda_abs_rot_gpu(CUREAL *rot, CUREAL *fx, CUREAL *fy, CUREAL *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) { 
 
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
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
 * rot       = Grid to be operated on (CUREAL *; input/output).
 * fx        = x component of the field (rgrid *; input).
 * fy        = y component of the field (rgrid *; input).
 * fz        = z component of the field (rgrid *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_abs_rotW(CUREAL *rot, CUREAL *fx, CUREAL *fy, CUREAL *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_abs_rot_gpu<<<blocks,threads>>>(rot, fx, fy, fz, inv_delta, bc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * A = POW(B,n) with n integer.
 *
 */

__global__ void rgrid_cuda_ipower_gpu(CUREAL *a, CUREAL *b, INT n, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, ii, sig;
  CUREAL value = 1.0;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  if(n == 0) {
    a[idx] = 1.0;
    return;
  }
  sig = (n < 0) ? -1:1;
  n = ABS(n);
  switch(n) {
    case 1:      
      a[idx] = b[idx];
      break;
    case 2:
      a[idx] = b[idx] * b[idx];
      break;
    case 3:
      a[idx] = b[idx] * b[idx] * b[idx];
      break;
    default:
      for(ii = 0; ii < n; ii++)
        value *= b[idx];
      a[idx] = value;
  }
  if(sig == -1) a[idx] = 1.0 / a[idx];
}

/*
 * Grid integer power.
 *
 * grida    = Destination for operation (REAL *; output).
 * gridb    = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_ipowerW(CUREAL *grida, CUREAL *gridb, INT exponent, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_ipower_gpu<<<blocks,threads>>>(grida, gridb, exponent, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Grid threshold clear device code.
 *
 */

__global__ void rgrid_cuda_threshold_clear_gpu(CUREAL *dest, CUREAL *src, CUREAL ul, CUREAL ll, CUREAL uval, CUREAL lval, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  if(src[idx] < ll) dest[idx] = lval;
  if(src[idx] > ul) dest[idx] = uval;
}

/*
 * Grid clear based on threshold.
 *
 * dest    = Destination for operation (REAL *; output).
 * src     = Source for operation (REAL *; input).
 * ul      = upper limit threshold for the operation (REAL; input).
 * ll      = lower limit threshold for the operation (REAL; input).
 * uval    = value to set when the upper limit was exceeded (REAL; input).
 * lval    = value to set when the lower limit was exceeded (REAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_threshold_clearW(CUREAL *dest, CUREAL *src, CUREAL ul, CUREAL ll, CUREAL uval, CUREAL lval, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_threshold_clear_gpu<<<blocks,threads>>>(dest, src, ul, ll, uval, lval, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Zero part of real grid.
 *
 * A = 0 in the specified range.
 *
 */

__global__ void rgrid_cuda_zero_index_gpu(CUREAL *a, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if(i >= lx && i < hx && j >= ly && j < hy && k >= lz && k < hz)
    a[idx] = 0.0;
}

/*
 * Zero specified range of complex grid.
 *
 * grid     = Grid to be operated on (CUREAL *; input/output).
 * lx, hx, ly, hy, lz, hz = limiting indices (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_zero_indexW(CUREAL *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_zero_index_gpu<<<blocks,threads>>>(grid, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  cuda_error_check();
}

/*
 * Poisson equation.
 *
 */

__global__ void rgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nzz) return;

  idx = (i * ny + j) * nzz + k;
  kx = COS(ilx * (CUREAL) i);
  ky = COS(ily * (CUREAL) j);
  kz = COS(ilz * (CUREAL) k);
  if(i || j || k)
    grid[idx] = grid[idx] * norm * step2 / (2.0 * (kx + ky + kz - 3.0));
  else
    grid[idx] = CUMAKE(0.0, 0.0);
}

/*
 * Solve Poisson.
 *
 * grid    = Grid specifying the RHS (CUREAL *; input/output).
 * norm    = FFT normalization constant (CUREAL; input).
 * step2   = Spatial step ^ 2 (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_poissonW(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, INT nx, INT ny, INT nz) {

  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = 2.0 * M_PI / ((CUREAL) nz);
  INT nzz = nz / 2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_poisson_gpu<<<blocks,threads>>>(grid, norm, step2, ilx, ily, ilz, nx, ny, nzz);
  cuda_error_check();
}

/*
 * FFT gradient (x).
 *
 */

__global__ void rgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *gradient, REAL kx0, REAL step, REAL norm, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  REAL lx, kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  lx = 2.0 * M_PI / (((REAL) nx) * step);
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kx * norm);
}

/*
 * Gradient of grid in Fourier space (X).
 *
 * gradient_x = Source & destination for operation (CUCOMPLEX *; input/output).
 * kx0        = Baseline momentum (grid->kx0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input). This is grid->nz2 / 2
 *
 */

extern "C" void rgrid_cuda_fft_gradient_xW(CUCOMPLEX *gradient_x, REAL kx0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fft_gradient_x_gpu<<<blocks,threads>>>(gradient_x, kx0, step, norm, nx, ny, nz, nx / 2);
  cuda_error_check();
}

/*
 * FFT gradient (y).
 *
 */

__global__ void rgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *gradient, REAL ky0, REAL step, REAL norm, INT nx, INT ny, INT nz, INT ny2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  REAL ly, ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  if(j < ny2) 
    ky = ((REAL) j) * ly - ky0;
  else
    ky = -((REAL) (ny - j)) * ly - ky0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, ky * norm);
}

/*
 * Gradient of grid in Fourier space (Y).
 *
 * gradient_y = Source & destination for operation (CUCOMPLEX *; input/output).
 * ky0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input). This is grid->nz2 / 2
 *
 */

extern "C" void rgrid_cuda_fft_gradient_yW(CUCOMPLEX *gradient_y, REAL ky0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fft_gradient_y_gpu<<<blocks,threads>>>(gradient_y, ky0, step, norm, nx, ny, nz, ny / 2);
  cuda_error_check();
}

/*
 * FFT gradient (z).
 *
 */

__global__ void rgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *gradient, REAL kz0, REAL step, REAL norm, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  REAL lz, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  lz = M_PI / (((REAL) nz - 1) * step);
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kz * norm);
}

/*
 * Gradient of grid in Fourier space (Z).
 *
 * gradient_z = Source & destination for operation (CUCOMPLEX *; input/output).
 * kz0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input). This is grid->nz2 / 2
 *
 */

extern "C" void rgrid_cuda_fft_gradient_zW(CUCOMPLEX *gradient_z, REAL kz0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fft_gradient_z_gpu<<<blocks,threads>>>(gradient_z, kz0, step, norm, nx, ny, nz, nz / 2);
  cuda_error_check();
}


/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void rgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(j < ny2) 
    ky = ((REAL) j) * ly - ky0;
  else
    ky = -((REAL) (ny - j)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  b[idx] = b[idx] * (-(kx * kx + ky * ky + kz * kz) * norm);
}

/*
 * FFT laplace
 *
 * laplace  = Source/destination grid for operation (REAL complex *; input/output).
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
 */

extern "C" void rgrid_cuda_fft_laplaceW(CUCOMPLEX *laplace, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = ny / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fft_laplace_gpu<<<blocks,threads>>>(laplace, norm, kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nz - 1) * step), step, nx, ny, nz, nx2, ny2, nz2);
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

__global__ void rgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUCOMPLEX *blocks, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
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
  if(j < ny2) 
    ky = ((REAL) j) * ly - ky0;
  else
    ky = -((REAL) (ny - j)) * ly - ky0;
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
      blocks[idx2].x += els2[t];  // reduce threads
    }
  }
}

/*
 * FFT laplace expectation value
 *
 * laplace  = Source/destination grid for operation (REAL complex *; input/output).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * sum      = Expectation value (REAL; output).
 *
 * Only periodic boundaries!
 *
 * Normalization done in cgrid-cuda.c
 *
 */

extern "C" void rgrid_cuda_fft_laplace_expectation_valueW(CUCOMPLEX *laplace, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks,threads,s*sizeof(CUREAL)>>>(laplace, (CUCOMPLEX *) grid_gpu_mem_addr, kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nz - 1) * step), step, nx, ny, nz, nx2, ny2, nz2);
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
}
