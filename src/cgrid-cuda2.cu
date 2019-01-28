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

__global__ void cgrid_cuda_fft_convolute_gpu(CUCOMPLEX *c, CUCOMPLEX *a, CUCOMPLEX *b, CUREAL norm, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z,
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if((i + j + k) & 1) norm *= -1.0;
  c[idx] = a[idx] * b[idx] * norm;
}

/*
 * Convolution in the Fourier space (data in GPU). Not called directly.
 *
 * gridc = convolution output (CUCOMPLEX *; output).
 * grida = 1st grid to be convoluted (CUCOMPLEX *; input).
 * gridb = 2nd grid to be convoluted (CUCOMPLEX *; input).
 * norm  = FFT norm (REAL complex; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_convoluteW(CUCOMPLEX *gridc, CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL norm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK, 
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fft_convolute_gpu<<<blocks,threads>>>(gridc, grida, gridb, norm, nx, ny, nz);
  cuda_error_check();
}

/*
 * Grid abs power device code. This cannot not be called directly.
 *
 * A = POW(|B|,x)
 *
 */

__global__ void cgrid_cuda_abs_power_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL x, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z,
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx].x = POW(CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]), x / 2.0);
  a[idx].y = 0.0;
}

/*
 * Grid abs power.
 *
 * gridb    = Destination for operation (REAL complex *; output).
 * grida    = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_abs_powerW(CUCOMPLEX *gridb, CUCOMPLEX *grida, CUREAL exponent, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_abs_power_gpu<<<blocks,threads>>>(gridb, grida, exponent, nx, ny, nz);
  cuda_error_check();
}

/*
 * Grid power device code. This cannot not be called directly.
 *
 * A = POW(B,x)
 *
 */

__global__ void cgrid_cuda_power_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL x, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx] = CUCPOW(b[idx], x);
}

/*
 * Grid power.
 *
 * gridb    = Destination for operation (REAL complex *; output).
 * grida    = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_powerW(CUCOMPLEX *gridb, CUCOMPLEX *grida, CUREAL exponent, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_power_gpu<<<blocks,threads>>>(gridb, grida, exponent, nx, ny, nz);
  cuda_error_check();
}

/*
 * Multiply grid by constant device code. This cannot not be called directly.
 *
 * A = C * A
 *
 */

__global__ void cgrid_cuda_multiply_gpu(CUCOMPLEX *a, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = a[idx] * c;
}

/*
 * Multiply grid by a constant.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * c        = Multiplying constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiplyW(CUCOMPLEX *grid, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_multiply_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz);
  cuda_error_check();
}

/*
 * Sum of two grids.
 *
 * A = B + C
 *
 */

__global__ void cgrid_cuda_sum_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = b[idx] + c[idx];
}

/*
 * Sum of two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Input grid 1 (CUCOMPLEX *; input).
 * gridc    = Input grid 2 (CUCOMPLEX *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_sumW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_sum_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Subtract of two grids.
 *
 * A = B - C
 *
 */

__global__ void cgrid_cuda_difference_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = b[idx] - c[idx];
}

/*
 * Subtract two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Input grid 1 (CUCOMPLEX *; input).
 * gridc    = Input grid 2 (CUCOMPLEX *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_differenceW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_difference_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Product of two grids.
 *
 * A = B * C.
 *
 */

__global__ void cgrid_cuda_product_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = b[idx] * c[idx];
}

/*
 * Product of two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Source grid 1 (CUCOMPLEX *; input).
 * gridc    = Source grid 2 (CUCOMPLEX *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_productW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_product_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Conjugate product of two grids.
 *
 * A = B^* X C.
 *
 */

__global__ void cgrid_cuda_conjugate_product_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = CUCONJ(b[idx]) * c[idx];
}

/*
 * Conjugate product of two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Source grid 1 (complex conjugated) (CUCOMPLEX *; input).
 * gridc    = Source grid 2 (CUCOMPLEX *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugate_productW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_conjugate_product_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Division of two grids.
 *
 * A = B / C.
 *
 * Note: One should avoid division as it is slow.
 */

__global__ void cgrid_cuda_division_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = b[idx] / c[idx];
}

/*
 * Division of two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Source grid 1 (CUCOMPLEX *; input).
 * gridc    = Source grid 2 (CUCOMPLEX *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_divisionW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_division_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Safe division of two grids.
 *
 * A = B / (C + eps).
 *
 * Note: One should avoid division as it is slow.
 *
 */

__global__ void cgrid_cuda_division_eps_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUCOMPLEX *c, CUREAL eps, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  a[idx] = b[idx] / (c[idx] + eps);
}

/*
 * "Safe" division of two grids.
 *
 * grida    = Destination grid (CUCOMPLEX *; output).
 * gridb    = Source grid 1 (CUCOMPLEX *; input).
 * gridc    = Source grid 2 (CUCOMPLEX *; input).
 * eps      = Epsilon (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_division_epsW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUCOMPLEX *gridc, CUREAL eps, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_division_eps_gpu<<<blocks,threads>>>(grida, gridb, gridc, eps, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add constant to grid device code. This cannot not be called directly.
 *
 * A = A + c
 *
 */

__global__ void cgrid_cuda_add_gpu(CUCOMPLEX *a, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = a[idx] + c;
}

/*
 * Add constant to grid.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * c        = Constant (CUCOMPLEX).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_addW(CUCOMPLEX *grid, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_add_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add multiply and add device code. This cannot not be called directly.
 *
 * A = cm * A + ca
 *
 */

__global__ void cgrid_cuda_multiply_and_add_gpu(CUCOMPLEX *a, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = (cm * a[idx]) + ca;
}

/*
 * Grid multiply and add.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * cm       = Multiplier (CUCOMPLEX; input).
 * ca       = Additive constant (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_multiply_and_addW(CUCOMPLEX *grid, CUCOMPLEX cm, CUCOMPLEX ca, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_multiply_and_add_gpu<<<blocks,threads>>>(grid, cm, ca, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add multiply and add device code.
 *
 * A = cm * (A + ca)
 *
 */

__global__ void cgrid_cuda_add_and_multiply_gpu(CUCOMPLEX *a, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = cm * (a[idx] + ca);
}

/*
 * Grid multiply and add.
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * ca       = Additive constant (CUCOMPLEX; input).
 * cm       = Multiplier (CUCOMPLEX; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_and_multiplyW(CUCOMPLEX *grid, CUCOMPLEX ca, CUCOMPLEX cm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_add_and_multiply_gpu<<<blocks,threads>>>(grid, ca, cm, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add scaled grid device code.
 *
 * A = A + d * B
 *
 */

__global__ void cgrid_cuda_add_scaled_gpu(CUCOMPLEX *a, CUCOMPLEX d, CUCOMPLEX *b, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = a[idx] + (d * b[idx]);
}

/*
 * Scaled add grid.
 *
 * grida    = Destination for operation (REAL complex *; output).
 * d        = Scaling factor (REAL complex; input).
 * gridb    = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaledW(CUCOMPLEX *grida, CUCOMPLEX d, CUCOMPLEX *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_add_scaled_gpu<<<blocks,threads>>>(grida, d, gridb, nx, ny, nz);
  cuda_error_check();
}

/*
 * Add scaled product grid device code.
 *
 * A = A + d * B * C
 *
 */

__global__ void cgrid_cuda_add_scaled_product_gpu(CUCOMPLEX *a, CUCOMPLEX d, CUCOMPLEX *b, CUCOMPLEX *c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = a[idx] + (d * b[idx] * c[idx]);
}

/*
 * Add scaled product.
 *
 * grida    = Destination for operation (REAL complex *; output).
 * d        = Scaling factor (REAL complex; input).
 * gridb    = Source for operation (REAL complex *; input).
 * gridc    = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_add_scaled_productW(CUCOMPLEX *grida, CUCOMPLEX d, CUCOMPLEX *gridb, CUCOMPLEX *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_add_scaled_product_gpu<<<blocks,threads>>>(grida, d, gridb, gridc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Set A to constant.
 *
 * A = c
 *
 */

__global__ void cgrid_cuda_constant_gpu(CUCOMPLEX *a, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, 
      idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = c;
}

/*
 * Set grid to constant.
 *
 * grid     = Destination for operation (REAL complex *; output).
 * c        = Constant (REAL complex; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_constantW(CUCOMPLEX *grid, CUCOMPLEX c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_constant_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz);
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
 * Integrate over grid A.
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
 * grid     = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integralW(CUCOMPLEX *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  // Blocks, Threads, dynamic memory size
  cgrid_cuda_integral_gpu<<<blocks,threads,s*sizeof(CUCOMPLEX)>>>(grid, (CUCOMPLEX *) grid_gpu_mem_addr, nx, ny, nz);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
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
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integral_regionW(CUCOMPLEX *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  if(il < 0) il = 0;  
  if(jl < 0) jl = 0;  
  if(kl < 0) kl = 0;  
  if(iu > nx-1) iu = nx-1;
  if(ju > ny-1) ju = ny-1;
  if(ku > nz-1) ku = nz-1;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  cgrid_cuda_integral_region_gpu<<<blocks,threads,s*sizeof(CUCOMPLEX)>>>(grid, (CUCOMPLEX *) grid_gpu_mem_addr, il, iu, jl, ju, kl, ku, nx, ny, nz);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
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
 * grid     = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integral_of_squareW(CUCOMPLEX *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  cgrid_cuda_integral_of_square_gpu<<<blocks,threads,s*sizeof(CUREAL)>>>(grid, (CUCOMPLEX *) grid_gpu_mem_addr, nx, ny, nz);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
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
 * grid1    = Source 1 for operation (REAL complex *; input).
 * grid2    = Source 2 for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_integral_of_conjugate_productW(CUCOMPLEX *grid1, CUCOMPLEX *grid2, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  // Blocks, Threads, dynamic memory size
  cgrid_cuda_integral_of_conjugate_product_gpu<<<blocks,threads,s*sizeof(CUCOMPLEX)>>>(grid1, grid2, (CUCOMPLEX *) grid_gpu_mem_addr, nx, ny, nz);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
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
 * grid1    = Source 1 (A) for operation (REAL complex *; input).
 * grid2    = Source 2 (B) for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void cgrid_cuda_grid_expectation_valueW(CUCOMPLEX *grid1, CUCOMPLEX *grid2, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  cgrid_cuda_grid_expectation_value_gpu<<<blocks,threads,s*sizeof(CUCOMPLEX)>>>(grid1, grid2, (CUCOMPLEX *) grid_gpu_mem_addr, nx, ny, nz);
  cuda_error_check();
  cgrid_cuda_block_reduce<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
}

/*
 * B = FD_X(A).
 *
 */

__global__ void cgrid_cuda_fd_gradient_x_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta * (cgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_X(A)
 *
 * grida     = Source for operation (REAL complex *; input).
 * gridb     = Destination for operation (REAL complex *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_xW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = FD_Y(A).
 *
 */

__global__ void cgrid_cuda_fd_gradient_y_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta * (cgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Y(A)
 *
 * grida     = Source for operation (REAL complex *; input).
 * gridb     = Destination for operation (REAL complex *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_yW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = FD_Z(A).
 *
 */

__global__ void cgrid_cuda_fd_gradient_z_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta * (cgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Z(A)
 *
 * grida     = Source for operation (REAL complex *; input).
 * gridb     = Destination for operation (REAL complex *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_zW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>(grida, gridb, inv_delta, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = LAPLACE(A).
 *
 */

__global__ void cgrid_cuda_fd_laplace_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz)
                         - 6.0 * a[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * grida      = Source for operation (REAL complex *; input).
 * gridb      = Destination for operation (REAL complex *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplacian in grid.
 *
 */

extern "C" void cgrid_cuda_fd_laplaceW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = LAPLACE_X(A).
 *
 */

__global__ void cgrid_cuda_fd_laplace_x_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_X(A)
 *
 * grida      = Source for operation (REAL complex *; input).
 * gridb      = Destination for operation (REAL complex *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_xW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = LAPLACE_Y(A).
 *
 */

__global__ void cgrid_cuda_fd_laplace_y_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta2 * (cgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_Y(A)
 *
 * grida      = Source for operation (REAL complex *; input).
 * gridb      = Destination for operation (REAL complex *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_yW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = LAPLACE_Z(A).
 *
 */

__global__ void cgrid_cuda_fd_laplace_z_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = inv_delta2 * (cgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz)
                         - 2.0 * a[idx]);
}

/*
 * B = LAPLACE_Z(A)
 *
 * grida      = Source for operation (REAL complex *; input).
 * gridb      = Destination for operation (REAL complex *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in gridb.
 *
 */

extern "C" void cgrid_cuda_fd_laplace_zW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 */

__global__ void cgrid_cuda_fd_gradient_dot_gradient_gpu(CUCOMPLEX *a, CUCOMPLEX *b, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;
  CUCOMPLEX tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  b[idx] = CUMAKE(0.0, 0.0);

  tmp = inv_delta * (cgrid_cuda_bc_x_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(a, bc, i, j, k, nx, ny, nz));
  b[idx] = b[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);

  tmp = inv_delta * (cgrid_cuda_bc_y_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(a, bc, i, j, k, nx, ny, nz));
  b[idx] = b[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);

  tmp = inv_delta * (cgrid_cuda_bc_z_plus(a, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(a, bc, i, j, k, nx, ny, nz));
  b[idx] = b[idx] + CUCREAL(tmp) * CUCREAL(tmp) + CUCIMAG(tmp) * CUCIMAG(tmp);
}

/*
 * B = FD_X(A)^2 + FD_Y(A)^2 + FD_Z(A)^2.
 *
 * grida      = Source for operation (REAL complex *; input).
 * gridb      = Destination for operation (REAL complex *; output).
 * inv_delta2 = 1 / (4 * step * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_dot_gradientW(CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_fd_gradient_dot_gradient_gpu<<<blocks,threads>>>(grida, gridb, inv_delta2, bc, nx, ny, nz);
  cuda_error_check();
}

/*
 * Complex conjugate.
 *
 * A = B*
 *
 */

__global__ void cgrid_cuda_conjugate_gpu(CUCOMPLEX *a, CUCOMPLEX *b, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  a[idx] = CUCONJ(b[idx]);
}

/*
 * Grid conjugate.
 *
 * gridb    = Destination for operation (REAL complex *; output).
 * grida    = Source for operation (REAL complex *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_conjugateW(CUCOMPLEX *gridb, CUCOMPLEX *grida, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cgrid_cuda_conjugate_gpu<<<blocks,threads>>>(gridb, grida, nx, ny, nz);
  cuda_error_check();
}

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
 */

__global__ void cgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUCOMPLEX *blocks, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {

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
  els2[idx2] += -(kx * kx + ky * ky + kz * kz) * (CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]));
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
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
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
 */

extern "C" void cgrid_cuda_fft_laplace_expectation_valueW(CUCOMPLEX *laplace, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  cgrid_cuda_block_init<<<1,1>>>((CUCOMPLEX *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks,threads,s*sizeof(CUREAL)>>>(laplace, (CUCOMPLEX *) grid_gpu_mem_addr, norm, kx0, ky0, kz0, step, nx, ny, nz, nx2, ny2, nz2);
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
