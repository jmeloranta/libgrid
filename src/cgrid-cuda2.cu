/*
 * CUDA device code (REAL complex; cgrid).
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
#include "cgrid_bc-cuda.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Fourier space convolution device code.
 *
 * C = A * B but with alternating signs for FFT.
 *
 */

__global__ void cgrid_cuda_fft_convolute_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, CUREAL norm, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z,
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if((i + j + k) & 1) norm *= -1.0;
  dst[idx] = src1[idx] * src2[idx] * norm;
}

/*
 * Convolution in the Fourier space (data in GPU). Not called directly.
 *
 * dst   = convolution output (cudaXtDesc_t *; output).
 * src1  = 1st grid to be convoluted (cudaXtDesc_t *; input).
 * src2  = 2nd grid to be convoluted (cudaXtDesc_t *; input).
 * norm  = FFT norm (REAL complex; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 * NOTE: This is in Fourier space -> partitioned along y (where the grid is data[x][y][z])
 *
 */

extern "C" void cgrid_cuda_fft_convoluteW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, CUREAL norm, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = ny % gpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets 
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_fft_convolute_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], norm, nx, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_fft_convolute_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLES *) src2->data[i], norm, nx, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Grid abs power device code. This cannot not be called directly.
 *
 * A = POW(|B|,x)
 *
 */

__global__ void cgrid_cuda_abs_power_gpu(CUCOMPLEX *dst, CUCOMPLEX *src, CUREAL x, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx].x = POW(CUCREAL(src[idx]) * CUCREAL(src[idx]) + CUCIMAG(src[idx]) * CUCIMAG(src[idx]), x / 2.0);
  dst[idx].y = 0.0;
}

/*
 * Grid abs power.
 *
 * dst      = Destination for operation (cudaXtDesc_t *; input).
 * src      = Source for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_abs_powerW(cudaXtDesc_t *dst, cudaXtDesc_t *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_abs_power_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src->data[i], exponent, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_abs_power_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src->data[i], exponent, nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Grid power device code. This cannot not be called directly.
 *
 * A = POW(B,x)
 *
 */

__global__ void cgrid_cuda_power_gpu(CUCOMPLEX *dst, CUCOMPLEX *src, CUREAL x, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  dst[idx] = CUCPOW(src[idx], x);
}

/*
 * Grid power.
 *
 * dst      = Destination for operation (cudaXtDesc_t *; output).
 * src      = Source for operation (cudaXtDesc_t *; input).
 * exponent = Exponent (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_powerW(cudaXtDesc_t *dst, cudaXtDesc_t *src, CUREAL exponent, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_power_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src->data[i], exponent, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_power_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLES *) src->data[i], exponent, nnx2, ny, nz);
  }
  cuda_error_check();
}

/*
 * Multiply grid by constant device code. This cannot not be called directly.
 *
 * A = C * A
 *
 */

__global__ void cgrid_cuda_multiply_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] * c;
}

/*
 * Multiply grid by a constant.
 *
 * dst      = Grid to be operated on (cudaXtDesc_t *; input/output).
 * c        = Multiplying constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiplyW(cudaXtDesc_t *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx2, ny, nz);
  }
  cuda_error_check();
}

/*
 * Sum of two grids.
 *
 * A = B + C
 *
 */

__global__ void cgrid_cuda_sum_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] + src2[idx];
}

/*
 * Sum of two grids.
 *
 * dst      = Destination grid (cudaXtDesc_t *; output).
 * src1     = Input grid 1 (cudaXtDesc_t *; input).
 * src2     = Input grid 2 (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_sumW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_sum_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_sum_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx2, ny, nz);
  }
  cuda_error_check();
}

/*
 * Subtract of two grids.
 *
 * dst = src1 - src2
 *
 */

__global__ void cgrid_cuda_difference_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] - src2[idx];
}

/*
 * Subtract two grids.
 *
 * dst      = Destination grid (cudaXtDesc_t *; output).
 * src1     = Input grid 1 (cudaXtDesc_t *; input).
 * src2     = Input grid 2 (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_differenceW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_difference_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_difference_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx2, ny, nz);
  }
  cuda_error_check();
}

/*
 * Product of two grids.
 *
 * dst = src1 * src2.
 *
 */

__global__ void cgrid_cuda_product_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] * src2[idx];
}

/*
 * Product of two grids.
 *
 * grida    = Destination grid (cudaXtDesc_t *; output).
 * gridb    = Source grid 1 (cudaXtDesc_t *; input).
 * gridc    = Source grid 2 (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_productW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grida->GPUs[i]);
    cgrid_cuda_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grida->data[i], (CUCOMPLEX *) gridb->data[i], (CUCOMPLEX *) gridc->data[i], nnx2, ny, nz);
  }
  cuda_error_check();
}

/*
 * Conjugate product of two grids.
 *
 * dst = src1^* X src2.
 *
 */

__global__ void cgrid_cuda_conjugate_product_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = CUCONJ(src1[idx]) * src2[idx];
}

/*
 * Conjugate product of two grids.
 *
 * dst      = Destination grid (cudaXtDesc_t *; output).
 * src1     = Source grid 1 (cudaXtDesc_t *; input).
 * src2     = Source grid 2 (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugate_productW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_conjugate_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_conjugate_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Division of two grids.
 *
 * dst = src1 / src2.
 *
 * Note: One should avoid division as it is slow on GPUs.
 *
 */

__global__ void cgrid_cuda_division_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = src1[idx] / src2[idx];
}

/*
 * Division of two grids.
 *
 * grida    = Destination grid (cudaXtDesc_t *; output).
 * gridb    = Source grid 1 (cudaXtDesc_t *; input).
 * gridc    = Source grid 2 (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_divisionW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Safe division of two grids.
 *
 * dst = src1 / (src2 + eps).
 *
 * Note: One should avoid division as it is slow on GPUs.
 *
 */

__global__ void cgrid_cuda_division_eps_gpu(CUCOMPLEX *dst, CUCOMPLEX *src1, CUCOMPLEX *src2, CUREAL eps, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  dst[idx] = src1[idx] / (src2[idx] + eps);
}

/*
 * "Safe" division of two grids.
 *
 * dst      = Destination grid (cudaXtDesc_t *; output).
 * src1     = Source grid 1 (cudaXtDesc_t *; input).
 * src2     = Source grid 2 (cudaXtDesc_t *; input).
 * eps      = Epsilon (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_division_epsW(cudaXtDesc_t *dst, cudaXtDesc_t *src1, cudaXtDesc_t *src2, CUREAL eps, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_eps_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], eps, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_division_eps_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], eps, nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Add constant to grid device code. This cannot not be called directly.
 *
 * dst = dst + c
 *
 */

__global__ void cgrid_cuda_add_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + c;
}

/*
 * Add constant to grid.
 *
 * dst      = Grid to be operated on (cudaXtDesc_t *; input/output).
 * c        = Constant (CUCOMPLEX).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_addW(cudaXtDesc_t *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Add multiply and add device code. This cannot not be called directly.
 *
 * dst = cm * dst + ca
 *
 */

__global__ void cgrid_cuda_multiply_and_add_gpu(CUCOMPLEX *dst, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = (cm * dst[idx]) + ca;
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (cudaXtDesc_t *; input/output).
 * cm       = Multiplier (CUCOMPLEX; input).
 * ca       = Additive constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiply_and_addW(cudaXtDesc_t *dst, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_and_add_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], cm, ca, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_multiply_and_add_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], cm, ca, nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Add multiply and add device code.
 *
 * dst = cm * (dst + ca)
 *
 */

__global__ void cgrid_cuda_add_and_multiply_gpu(CUCOMPLEX *dst, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = cm * (dst[idx] + ca);
}

/*
 * Grid multiply and add.
 *
 * dst      = Grid to be operated on (cudaXtDesc_t *; input/output).
 * ca       = Additive constant (CUCOMPLEX; input).
 * cm       = Multiplier (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_and_multiplyW(cudaXtDesc_t *dst, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_and_multiply_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], ca, cm, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_and_multiply_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], ca, cm, nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Add scaled grid device code.
 *
 * dst = dst + d * src
 *
 */

__global__ void cgrid_cuda_add_scaled_gpu(CUCOMPLEX *dst, CUCOMPLEX d, CUCOMPLEX *src, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + (d * src[idx]);
}

/*
 * Scaled add grid.
 *
 * dst      = Destination for operation (cudaXtDesc_t *; output).
 * d        = Scaling factor (CUCOMPLEX; input).
 * srd      = Source for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaledW(cudaXtDesc_t *dst, CUCOMPLEX d, cudaXtDesc_t *src, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], d, (CUCOMPLEX *) src->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], d, (CUCOMPLEX *) src->data[i], nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Add scaled product grid device code.
 *
 * dst = dst + d * src1 * src2
 *
 */

__global__ void cgrid_cuda_add_scaled_product_gpu(CUCOMPLEX *dst, CUCOMPLEX d, CUCOMPLEX *src1, CUCOMPLEX *src2, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] + (d * src1[idx] * src2[idx]);
}

/*
 * Add scaled product.
 *
 * dst      = Destination for operation (cudaXtDesc_t *; output).
 * d        = Scaling factor (CUCOMPLEX; input).
 * src1     = Source for operation (cudaXtDesc_t *; input).
 * src2     = Source for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaled_productW(cudaXtDesc_t *dst, CUCOMPLEX d, cudaXtDesc_t *src1, cudaXtDesc_t *src2, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_product_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], d, (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_add_scaled_product_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], d, (CUCOMPLEX *) src1->data[i], (CUCOMPLEX *) src2->data[i], nnx2, ny, nz);
  }

  cuda_error_check();
}

/*
 * Set dst to constant.
 *
 * dst = c
 *
 */

__global__ void cgrid_cuda_constant_gpu(CUCOMPLEX *dst, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = c;
}

/*
 * Set grid to constant.
 *
 * dst      = Destination for operation (cudaXtDesc_t *; output).
 * c        = Constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_constantW(cudaXtDesc_t *grid, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_constant_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_constant_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], c, nnx2, ny, nz);
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

__global__ void cgrid_cuda_integral_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

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
 * grid     = Source for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" CUCOMPLEX cgrid_cuda_integralW(CUCOMPLEX *grid, INT nx, INT ny, INT nz) {

  CUCOMPLEX value = 0.0, tmp;
  int i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i],nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }
}

/*
 * Integrate over A with limits.
 *
 */

__global__ void cgrid_cuda_integral_region_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUCOMPLEX els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t].x = els[t].y = 0.0;
  }
  __syncthreads();

  if(i >= il && i <= iu && j >= jl && j <= ju && k >= kl && k <= ku) {
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
 * grid     = Source for operation (REAL complex *; input).
 * il       = Lower index for x (INT; input).
 * iu       = Upper index for x (INT; input).
 * jl       = Lower index for y (INT; input).
 * ju       = Upper index for y (INT; input).
 * kl       = Lower index for z (INT; input).
 * ku       = Upper index for z (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0]. Integral in real space.
 *
 */

extern "C" void cgrid_cuda_integral_regionW(CUCOMPLEX *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz) {

  CUCOMPLEX value = 0.0, tmp;
  int i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, idx;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  if(il < 0) il = 0;  
  if(jl < 0) jl = 0;  
  if(kl < 0) kl = 0;  
  if(iu > nx-1) iu = nx-1;
  if(ju > ny-1) ju = ny-1;
  if(ku > nz-1) ku = nz-1;

  idx = 0;
  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], 
                                    il - idx, iu - idx, jl, ju, kl, ku, nnx1, ny, nz);  // shift x limit as the x index has been shifted
    idx -= nnx1;
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_region_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], 
                                    il - idx, iu - idx, jl, ju, kl, ku, nnx1, ny, nz);
    idx -= nnx2;
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }
}

/*
 * Integrate of |A|^2.
 *
 */

__global__ void cgrid_cuda_integral_of_square_gpu(CUCOMPLEX *a, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els2[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els2[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;
  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els2[idx2] += a[idx].x * a[idx].x + a[idx].y * a[idx].y;
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2].x += els2[t];  // reduce threads
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
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integral_of_squareW(cudaXtDesc_t *grid, INT nx, INT ny, INT nz) {

  CUCOMPLEX value = 0.0, tmp;
  int i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_square_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i],nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_square_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }
}

/*
 * Integrate A^* X B (overlap).
 *
 */

__global__ void cgrid_cuda_integral_of_conjugate_product_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

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
 * grida    = Source 1 for operation (cudaXtDesc_t *; input).
 * gridn    = Source 2 for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integral_of_conjugate_productW(cudaXtDesc_t *grida, cudaXtDesc_t *gridb, INT nx, INT ny, INT nz) {

  CUCOMPLEX value = 0.0, tmp;
  int i, ngpu2 = grida->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grida->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_conjugate_product_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grida->data[i], (CUCOMPLEX *) gridb->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grida->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_integral_of_square_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grida->data[i], (CUCOMPLEX *) gridb->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }
}

/*
 * Integrate A * |B|^2.
 *
 */

__global__ void cgrid_cuda_grid_expectation_value_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *blocks, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
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
 * grida    = Source 1 (A) for operation (cudaXtDesc_t *; input).
 * gridb    = Source 2 (B) for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_grid_expectation_valueW(cudaXtDesc_t *grida, cudaXtDesc_t *gridb, INT nx, INT ny, INT nz) {

  CUCOMPLEX value = 0.0, tmp;
  int i, ngpu2 = grida->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grida->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_grid_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grida->data[i], (CUCOMPLEX *) gridb->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grida->GPUs[i]);
    cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31);
    cuda_error_check();
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_grid_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUCOMPLEX)>>>(CUCOMPLEX *) grida->data[i], (CUCOMPLEX *) gridb->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], nnx1, ny, nz);
    cuda_error_check();
    cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b3);
    cuda_error_check();
  }
}

/*
 * dst = FD_X(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * dst = FD_X(src)
 *
 * src       = Source for operation (cudaXtDesc_t *; input).
 * dst       = Destination for operation (cudaXtDesc_t *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_xW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst>nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst>GPUs[0]);
  cgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * dst = FD_Y(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_y_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Y(A)
 *
 * src       = Source for operation (cudaXtDesc_t *; input).
 * dst       = Destination for operation (cudaXtDesc_t *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_yW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * dst = FD_Z(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_z_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Z(A)
 *
 * src       = Source for operation (cudaXtDesc_t *; input).
 * dst       = Destination for operation (cudaXtDesc_t *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_zW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst>GPUs[0]);
  cgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * dst = LAPLACE(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_gpu(CUCOMPLEX *arc, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

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
 * src        = Source for operation (cudaXtDesc_t *; input).
 * dst        = Destination for operation (cudaXtDesc_t *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplacian in dst.
 *
 */

extern "C" void cgrid_cuda_fd_laplaceW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * dst = LAPLACE_X(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz)
                         - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_X(src)
 *
 * src        = Source for operation (cudaXtDesc_t *; input).
 * dst        = Destination for operation (cudaXtDesc_t *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_xW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta2, bc, nx, ny, nz);
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
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz)
                         - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_Y(A)
 *
 * src        = Source for operation (cudaXtDesc_t *; input).
 * dst        = Destination for operation (cudaXtDesc_t *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_yW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta2, bc, nx, ny, nz);
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
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz)
                         - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_Z(src)
 *
 * src        = Source for operation (cudaXtDesc_t *; input).
 * dst        = Destination for operation (cudaXtDesc_t *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_zW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta2, bc, nx, ny, nz);
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
 * src        = Source for operation (cudaXtDesc_t *; input).
 * dst        = Destination for operation (cudaXtDesc_t *; output).
 * inv_delta2 = 1 / (4 * step * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_dot_gradientW(cudaXtDesc_t *src, cudaXtDesc_t *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  CudaSetDevice(dst->GPUs[0]);
  cgrid_cuda_fd_gradient_dot_gradient_gpu<<<blocks,threads>>>((CUCOMPLEX *) src->data[0], (CUCOMPLEX *) dst->data[0], inv_delta2, bc, nx, ny, nz);
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
 * dst      = Destination for operation (cudaXtDesc_t *; output).
 * src      = Source for operation (cudaXtDesc_t *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugateW(cudaXtDesc_t *dst, cudaXtDesc_t *src, INT nx, INT ny, INT nz) {

  int i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_conjugate_gpu<<<blocks1,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src->data[i], nnx1, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_conjugate_gpu<<<blocks2,threads>>>((CUCOMPLEX *) dst->data[i], (CUCOMPLEX *) src->data[i], nnx2, ny, nz);
  }

  cuda_error_check();
}

///////////// LEFT HERE //////////////

/*
 * FFT gradient_x
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL step, INT nx, INT ny, INT nz, INT nx2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
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
 * gradient_x= Source/destination grid for operation (REAL complex *; input/output).
 * norm      = FFT norm (grid->fft_norm) (REAL; input).
 * kx0       = Momentum shift of origin along X (REAL; input).
 * step      = Spatial step length (REAL; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * Only periodic boundaries (FFT)!
 *
 */

extern "C" void cgrid_cuda_fft_gradient_xW(CUCOMPLEX *gradient_x, CUREAL norm, CUREAL kx0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fft_gradient_x_gpu<<<blocks,threads>>>(gradient_x, norm, kx0, step, nx, ny, nz, nx2);
  cuda_error_check();
}

/*
 * FFT gradient_y
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL ky0, CUREAL step, INT nx, INT ny, INT nz, INT ny2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (j <= ny2)
    ky = 2.0 * M_PI * ((CUREAL) j) / (((CUREAL) ny) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((CUREAL) (j - ny)) / (((CUREAL) ny) * step) - ky0;

  b[idx] = b[idx] * ky * norm;    // multiply by I * ky * norm
}

/*
 * FFT gradient_y
 *
 * gradient_y= Source/destination grid for operation (REAL complex *; input/output).
 * norm      = FFT norm (grid->fft_norm) (REAL; input).
 * kx0       = Momentum shift of origin along X (REAL; input).
 * step      = Spatial step length (REAL; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void cgrid_cuda_fft_gradient_yW(CUCOMPLEX *gradient_y, CUREAL norm, CUREAL ky0, CUREAL step, INT nx, INT ny, INT nz) {

  INT ny2 = ny / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fft_gradient_y_gpu<<<blocks,threads>>>(gradient_y, norm, ky0, step, nx, ny, nz, ny2);
  cuda_error_check();
}

/*
 * FFT gradient_z
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
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
 * gradient_z= Source/destination grid for operation (REAL complex *; input/output).
 * norm      = FFT norm (grid->fft_norm) (REAL; input).
 * kx0       = Momentum shift of origin along X (REAL; input).
 * step      = Spatial step length (REAL; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void cgrid_cuda_fft_gradient_zW(CUCOMPLEX *gradient_z, CUREAL norm, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nz2 = nz / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fft_gradient_z_gpu<<<blocks,threads>>>(gradient_z, norm, kz0, step, nx, ny, nz, nz2);
  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (i <= nx2)
    kx = 2.0 * M_PI * ((CUREAL) i) / (((CUREAL) nx) * step) - kx0;
  else 
    kx = 2.0 * M_PI * ((CUREAL) (i - nx)) / (((CUREAL) nx) * step) - kx0;
      
  if (j <= ny2)
    ky = 2.0 * M_PI * ((CUREAL) j) / (((CUREAL) ny) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((CUREAL) (j - ny)) / (((CUREAL) ny) * step) - ky0;
      
  if (k <= nz2)
    kz = 2.0 * M_PI * ((CUREAL) k) / (((CUREAL) nz) * step) - kz0;
  else 
    kz = 2.0 * M_PI * ((CUREAL) (k - nz)) / (((CUREAL) nz) * step) - kz0;      

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

extern "C" void cgrid_cuda_fft_laplaceW(CUCOMPLEX *laplace, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = ny / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fft_laplace_gpu<<<blocks,threads>>>(laplace, norm, kx0, ky0, kz0, step, nx, ny, nz, nx2, ny2, nz2);
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

__global__ void cgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUCOMPLEX *blocks, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {

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

  if (i <= nx2)
    kx = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
  else 
    kx = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step) - kx0;
      
  if (j <= ny2)
    ky = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
  else 
    ky = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - ky0;
      
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

extern "C" void cgrid_cuda_fft_laplace_expectation_valueW(CUCOMPLEX *laplace, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks,threads,s*sizeof(CUREAL)>>>(laplace, (CUCOMPLEX *) grid_gpu_mem_addr, kx0, ky0, kz0, step, nx, ny, nz, nx2, ny2, nz2);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
}

/*
 * Zero real part.
 *
 * A.re = 0
 *
 */

__global__ void cgrid_cuda_zero_re_gpu(CUCOMPLEX *a, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx].x = 0.0;
}

/*
 * Zero real part.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_reW(CUCOMPLEX *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_zero_re_gpu<<<blocks,threads>>>(grid, nx, ny, nz);
  cuda_error_check();
}

/*
 * Zero imaginary part.
 *
 * A.im = 0
 *
 */

__global__ void cgrid_cuda_zero_im_gpu(CUCOMPLEX *a, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx].y = 0.0;
}

/*
 * Zero imaginary part.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_imW(CUCOMPLEX *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_zero_im_gpu<<<blocks,threads>>>(grid, nx, ny, nz);
  cuda_error_check();
}

/*
 * Zero part of complex grid.
 *
 * A = 0 in the specified range.
 *
 */

__global__ void cgrid_cuda_zero_index_gpu(CUCOMPLEX *a, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if(i >= lx && i < hx && j >= ly && j < hy && k >= lz && k < hz)
    a[idx] = CUMAKE(0.0, 0.0);
}

/*
 * Zero specified range of complex grid.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * lx, hx, ly, hy, lz, hz = limiting indices (INT; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_zero_indexW(CUCOMPLEX *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_zero_index_gpu<<<blocks,threads>>>(grid, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  cuda_error_check();
}

/*
 * Poisson equation.
 *
 */

__global__ void cgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
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
 * grid    = Grid specifying the RHS (CUCOMPLEX *; input/output).
 * norm    = FFT normalization constant (CUREAL; input).
 * step2   = Spatial step ^ 2 (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_poissonW(CUCOMPLEX *grid, CUREAL norm, CUREAL step2, INT nx, INT ny, INT nz) {

  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = 2.0 * M_PI / ((CUREAL) nz);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_poisson_gpu<<<blocks,threads>>>(grid, norm, step2, ilx, ily, ilz, nx, ny, nz);
  cuda_error_check();
}
