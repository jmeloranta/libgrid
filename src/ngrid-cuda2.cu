/*
 * CUDA device code (mixed cgrid/rgrid).
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

extern "C" void cuda_error_check();

/*
 * Real to complex_re.
 *
 * A.re = B(real). (zeroes the imag part)
 *
 */

__global__ void grid_cuda_real_to_complex_re_gpu(CUCOMPLEX *a, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;    // Index for complex grid
  idx2 = (i * ny + j) * nzz + k;  // Index for real grid

  a[idx] = CUMAKE(b[idx2], 0.0);
}

/*
 * Real to complex_re
 *
 * grida   = Destination for operation (REAL complex *; output).
 * gridb   = Source for operation (REAL *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_real_to_complex_reW(CUCOMPLEX *grida, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_real_to_complex_re_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Real to complex_im.
 *
 * A.im = B(real). (zeroes the real part)
 *
 */

__global__ void grid_cuda_real_to_complex_im_gpu(CUCOMPLEX *a, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nzz + k;

  a[idx] = CUMAKE(0.0, b[idx2]);
}

/*
 * Real to complex_im
 *
 * grida   = Destination for operation (REAL complex *; output).
 * gridb   = Source for operation (REAL *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_real_to_complex_imW(CUCOMPLEX *grida, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_real_to_complex_im_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Add real to complex_re.
 *
 * A.re = A.re + B(real).
 *
 */

__global__ void grid_cuda_add_real_to_complex_re_gpu(CUCOMPLEX *a, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  a[idx].x = CUCREAL(a[idx]) + b[idx2];
}

/*
 * Add real to complex.re
 *
 * grida   = Destination for operation (REAL complex *; output).
 * gridb   = Source for operation (REAL *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_add_real_to_complex_reW(CUCOMPLEX *grida, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_add_real_to_complex_re_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Add real to complex_im.
 *
 * A.im = A.im + B(real).
 *
 */

__global__ void grid_cuda_add_real_to_complex_im_gpu(CUCOMPLEX *a, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  a[idx].y = CUCIMAG(a[idx]) + b[idx2];
}

/*
 * Add real to complex_im
 *
 * grida   = Destination for operation (REAL complex *; output).
 * gridb   = Source for operation (REAL *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_add_real_to_complex_imW(CUCOMPLEX *grida, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_add_real_to_complex_im_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Product A(complex) and B(real).
 *
 * A = A * B(real).
 *
 */

__global__ void grid_cuda_product_complex_with_real_gpu(CUCOMPLEX *a, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  a[idx] = a[idx] * b[idx2];
}

/*
 * Product A(complex) with B(real).
 *
 * grida   = Destination for operation (REAL complex *; output).
 * gridb   = Source for operation (REAL *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_product_complex_with_realW(CUCOMPLEX *grida, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_product_complex_with_real_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Imag. part to real grid.
 *
 * A(real) = B.im;
 *
 */

__global__ void grid_cuda_complex_im_to_real_gpu(CUREAL *a, CUCOMPLEX *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  a[idx2] = CUCIMAG(b[idx]);
}

/*
 * Imag. part of B to real A.
 *
 * grida   = Destination for operation (REAL *; output).
 * gridb   = Source for operation (REAL complex *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_complex_im_to_realW(CUREAL *grida, CUCOMPLEX *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_complex_im_to_real_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Real part to real grid.
 *
 * A(real) = B.re;
 *
 */

__global__ void grid_cuda_complex_re_to_real_gpu(CUREAL *a, CUCOMPLEX *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx, idx2, tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  tmp = i * ny + j;
  idx = tmp * nz + k;
  idx2 = tmp * nzz + k;

  a[idx2] = CUCREAL(b[idx]);
}

/*
 * Real part of B to real A.
 *
 * grida   = Destination for operation (REAL *; output).
 * gridb   = Source for operation (REAL complex *; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_complex_re_to_realW(CUREAL *grida, CUCOMPLEX *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_complex_re_to_real_gpu<<<blocks,threads>>>(grida, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}
