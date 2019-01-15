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

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 *
 * C = A * B but with alternating signs for FFT.
 *
 */

__global__ void rgrid_cuda_fft_convolute_gpu(CUCOMPLEX *c, CUCOMPLEX *a, CUCOMPLEX *b, CUREAL norm, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if((i + j + k) & 1) norm *= -1.0;
  c[idx] = norm * a[idx] * b[idx];
}

/*
 * Convolution in the Fourier space (data in GPU). Not called directly.
 *
 * Multiplication in GPU memory: grid_gpu_mem[i] = grid_gpu_mem[i] * grid_gpu_mem[j] (with sign variation).
 * Note: this includes the sign variation needed for convolution as well as normalization!
 *
 * grida = 1st grid to be convoluted (CUCOMPLEX *; input).
 * gridb = 2nd grid to be convoluted (CUCOMPLEX *; input).
 * gridc = output (CUCOMPLEX *; output).
 * norm  = FFT norm (REAL; input).
 * nx    = Grid dim x (INT; input).
 * ny    = Grid dim y (INT; input).
 * nz    = Grid dim z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_convoluteW(CUCOMPLEX *gridc, CUCOMPLEX *grida, CUCOMPLEX *gridb, CUREAL norm, INT nx, INT ny, INT nz) {

  INT nzz = nz / 2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_fft_convolute_gpu<<<blocks,threads>>>(gridc, grida, gridb, norm, nx, ny, nzz);
  cuda_error_check();
}

/*
 *
 * A = POW(B,x) = B^x
 *
 */

__global__ void rgrid_cuda_power_gpu(CUREAL *a, CUREAL *b, CUREAL x, INT nx, INT ny, INT nz, INT nzz) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = POW(b[idx], x);
}

/*
 * Grid power.
 *
 * grida    = Destination for operation (REAL *; output).
 * gridb    = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_powerW(CUREAL *grida, CUREAL *gridb, CUREAL exponent, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_power_gpu<<<blocks,threads>>>(grida, gridb, exponent, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = POW(|B|,x) = |B|^x
 *
 */

__global__ void rgrid_cuda_abs_power_gpu(CUREAL *a, CUREAL *b, CUREAL x, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = POW(FABS(b[idx]), x);
}

/*
 * Grid abs power.
 *
 * grida    = Destination for operation (REAL *; output).
 * gridb    = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_abs_powerW(CUREAL *grida, CUREAL *gridb, CUREAL exponent, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_abs_power_gpu<<<blocks,threads>>>(grida, gridb, exponent, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = C * A
 *
 */

__global__ void rgrid_cuda_multiply_gpu(CUREAL *a, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;

  if(i >= nx || j >= ny || k >= nz) return;

  a[(i * ny + j) * nzz + k] *= c;
}

/*
 * Multiply grid by a constant.
 *
 * grid     = Grid to be operated on (CUREAL *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiplyW(CUREAL *grid, CUREAL c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_multiply_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = C * A (in FFT space)
 *
 */

__global__ void rgrid_cuda_multiply_fft_gpu(CUCOMPLEX *a, CUREAL c, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  a[idx] = a[idx] * c;
}

/*
 * Multiply (complex) grid by a constant (in FFT space).
 *
 * grid     = Grid to be operated on (CUCOMPLEX *; input/output).
 * c        = Multiplying constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_fftW(CUCOMPLEX *grid, CUREAL c, INT nx, INT ny, INT nz) {

  INT nzz = nz / 2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nzz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_multiply_fft_gpu<<<blocks,threads>>>(grid, c, nx, ny, nzz);
  cuda_error_check();
}

/*
 *
 * A = B + C
 *
 */

__global__ void rgrid_cuda_sum_gpu(CUREAL *a, CUREAL *b, CUREAL *c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = b[idx] + c[idx];
}

/*
 * Sum of two grids.
 *
 * grida    = Destination grid (CUREAL *; output).
 * gridb    = Input grid 1 (CUREAL *; input).
 * gridc    = Input grid 2 (CUREAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_sumW(CUREAL *grida, CUREAL *gridb, CUREAL *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_sum_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = B - C
 *
 */

__global__ void rgrid_cuda_difference_gpu(CUREAL *a, CUREAL *b, CUREAL *c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = b[idx] - c[idx];
}

/*
 * Subtract two grids.
 *
 * grida    = Destination grid (CUREAL *; output).
 * gridb    = Input grid 1 (CUREAL *; input).
 * gridc    = Input grid 2 (CUREAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_differenceW(CUREAL *grida, CUREAL *gridb, CUREAL *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_difference_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = B * C.
 *
 */

__global__ void rgrid_cuda_product_gpu(CUREAL *a, CUREAL *b, CUREAL *c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = b[idx] * c[idx];
}

/*
 * Product of two grids.
 *
 * grida    = Destination grid (CUREAL *; output).
 * gridb    = Source grid 1 (CUREAL *; input).
 * gridc    = Source grid 2 (CUREAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_productW(CUREAL *grida, CUREAL *gridb, CUREAL *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_product_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = B / C.
 *
 */

__global__ void rgrid_cuda_division_gpu(CUREAL *a, CUREAL *b, CUREAL *c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = b[idx] / c[idx];
}

/*
 * Division of two grids.
 *
 * grida    = Destination grid (CUREAL *; output).
 * gridb    = Source grid 1 (CUREAL *; input).
 * gridc    = Source grid 2 (CUREAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_divisionW(CUREAL *grida, CUREAL *gridb, CUREAL *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_division_gpu<<<blocks,threads>>>(grida, gridb, gridc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = B / (C + eps).
 *
 */

__global__ void rgrid_cuda_division_eps_gpu(CUREAL *a, CUREAL *b, CUREAL *c, CUREAL eps, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = b[idx] / (c[idx] + eps);
}

/*
 * Division of two grids.
 *
 * grida    = Destination grid (CUREAL *; output).
 * gridb    = Source grid 1 (CUREAL *; input).
 * gridc    = Source grid 2 (CUREAL *; input).
 * eps      = Epsilon (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_division_epsW(CUREAL *grida, CUREAL *gridb, CUREAL *gridc, CUREAL eps, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_division_eps_gpu<<<blocks,threads>>>(grida, gridb, gridc, eps, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = A + c
 *
 */

__global__ void rgrid_cuda_add_gpu(CUREAL *a, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] += c;
}

/*
 * Add constant to grid.
 *
 * grid     = Grid to be operated on (CUREAL *; input/output).
 * c        = Constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_addW(CUREAL *grid, CUREAL c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_add_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = cm * A + ca
 *
 */

__global__ void rgrid_cuda_multiply_and_add_gpu(CUREAL *a, CUREAL cm, CUREAL ca, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = a[idx] * cm + ca;
}

/*
 * Grid multiply and add.
 *
 * grid     = Grid to be operated on (CUREAL *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_multiply_and_addW(CUREAL *grid, CUREAL cm, REAL ca, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_multiply_and_add_gpu<<<blocks,threads>>>(grid, cm, ca, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = cm * (A + ca)
 *
 */

__global__ void rgrid_cuda_add_and_multiply_gpu(CUREAL *a, CUREAL ca, CUREAL cm, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = (a[idx] + ca) * cm;
}

/*
 * Grid multiply and add.
 *
 * grid     = Grid to be operated on (CUREAL *; input/output).
 * cm       = Multiplier (CUREAL; input).
 * ca       = Additive constant (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_and_multiplyW(CUREAL *grid, CUREAL ca, CUREAL cm, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_add_and_multiply_gpu<<<blocks,threads>>>(grid, ca, cm, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = A + d * B
 *
 */

__global__ void rgrid_cuda_add_scaled_gpu(CUREAL *a, CUREAL d, CUREAL *b, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = a[idx] + b[idx] * d;
}

/*
 * Scaled add grid.
 *
 * grida    = Destination for operation (REAL *; output).
 * d        = Scaling factor (REAL; input).
 * gridb    = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaledW(CUREAL *grida, CUREAL d, CUREAL *gridb, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_add_scaled_gpu<<<blocks,threads>>>(grida, d, gridb, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = A + d * B * C
 *
 */

__global__ void rgrid_cuda_add_scaled_product_gpu(CUREAL *a, CUREAL d, CUREAL *b, CUREAL *c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = a[idx] + d * b[idx] * c[idx];
}

/*
 * Add scaled product.
 *
 * grida    = Destination for operation (REAL *; output).
 * d        = Scaling factor (REAL; input).
 * gridb    = Source for operation (REAL *; input).
 * gridc    = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_add_scaled_productW(CUREAL *grida, CUREAL d, CUREAL *gridb, CUREAL *gridc, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_add_scaled_product_gpu<<<blocks,threads>>>(grida, d, gridb, gridc, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 *
 * A = c
 *
 */

__global__ void rgrid_cuda_constant_gpu(CUREAL *a, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  a[idx] = c;
}

/*
 * Set grid to constant.
 *
 * grid     = Destination for operation (REAL *; output).
 * c        = Constant (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_constantW(CUREAL *grid, CUREAL c, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_constant_gpu<<<blocks,threads>>>(grid, c, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/*
 * Block init (zero elements).
 *
 * blocks  = Block table (CUCOMPLEX *; output).
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
 * blocks  = Block list to reduce (CUCOMPLEX *; input/output). blocks[0] will contain the reduced value.
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
 * grid     = Source for operation (REAL *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Returns the value of integral in grid_gpu_mem[0].
 *
 */

extern "C" void rgrid_cuda_integralW(CUREAL *grid, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b3 = blocks.x * blocks.y * blocks.z;

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  // Blocks, Threads, dynamic memory size
  rgrid_cuda_integral_gpu<<<blocks,threads,s*sizeof(REAL)>>>(grid, (CUREAL *) grid_gpu_mem_addr, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
}

/*
 * Integrate over A with limits.
 *
 */

__global__ void rgrid_cuda_integral_region_gpu(CUREAL *a, CUREAL *blocks, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  extern __shared__ CUREAL els[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els[t] = 0.0;
  }
  __syncthreads();

  if(i >= il && i <= iu && j >= jl && j <= ju && k >= kl && k <= ku) {
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
 * grid     = Source for operation (REAL *; input).
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

extern "C" void rgrid_cuda_integral_regionW(CUREAL *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, INT nx, INT ny, INT nz) {

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

  rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
  cuda_error_check();
  rgrid_cuda_integral_region_gpu<<<blocks,threads,s*sizeof(REAL)>>>(grid, (CUREAL *) grid_gpu_mem_addr, il, iu, jl, ju, kl, ku, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
  rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr, b3);
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
 * Poisson equation.
 *
 */

__global__ void rgrid_cuda_poisson_gpu(CUREAL *grid, CUREAL norm, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nz) {
  
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
    grid[idx] = 0.0;
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

extern "C" void rgrid_cuda_poissonW(CUREAL *grid, CUREAL norm, CUREAL step2, INT nx, INT ny, INT nz) {

  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = 2.0 * M_PI / ((CUREAL) nz);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  rgrid_cuda_poisson_gpu<<<blocks,threads>>>(grid, norm, step2, ilx, ily, ilz, nx, ny, nz);
  cuda_error_check();
}
