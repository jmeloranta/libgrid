/*
 * Function #7 a & b & c & d: Potential function and its dervatives.
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "../cuda-math.h"

#include "func7.h"

extern "C" void cuda_error_check();

/****************** Potential function **********************/

__device__ CUREAL pot_func(CUREAL r, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5) {

  CUREAL ri, r2, r4, r6, r8, r10;

  r -= radd;
  if(r < rmin) r = rmin;

  ri = 1.0 / r;  // minimize divisions (slow)
  r2 = ri * ri;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return a0 * EXP(-a1 * r) - a2 * r4 - a3 * r6 - a4 * r8 - a5 * r10;
}

__global__ void grid_func7a_cuda_operate_one_gpu(CUCOMPLEX *a, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2, CUREAL x0, CUREAL y0, CUREAL z0, CUREAL step) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL x, y, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  x = ((CUREAL) (i - nx2)) * step - x0;
  y = ((CUREAL) (j - ny2)) * step - y0;    
  z = ((CUREAL) (k - nz2)) * step - z0;        
  a[idx].x += pot_func(SQRT(x*x + y*y + z*z), rmin, radd, a0, a1, a2, a3, a4, a5);
}

extern "C" void grid_func7a_cuda_operate_oneW(CUCOMPLEX *grid, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, CUREAL x0, CUREAL y0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func7a_cuda_operate_one_gpu<<<blocks,threads>>>(grid, rmin, radd, a0, a1, a2, a3, a4, a5, nx, ny, nz, nx/2, ny/2, nz/2, x0, y0, z0, step);
  cuda_error_check();
}

/****************** d/dx of the potential function **********************/

__device__ CUREAL pot_funcd(CUREAL rp, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5) {

  CUREAL r, ri, r2, r3, r5, r7, r9, r11;

  r = rp - radd;
  if(r < rmin) return 0.0;

  ri = 1.0 / r;
  r2 = ri * ri;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (-a0 * a1 * EXP(-a1 * r) + 4.0 * a2 * r5 + 6.0 * a3 * r7 + 8.0 * a4 * r9 + 10.0 * a5 * r11) / rp;
}

__global__ void grid_func7b_cuda_operate_one_gpu(CUREAL *a, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, INT nzz, INT nx2, INT ny2, INT nz2, REAL x0, REAL y0, REAL z0, REAL step) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL x, y, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;
  x = ((CUREAL) (i - nx2)) * step - x0;
  y = ((CUREAL) (j - ny2)) * step - y0;    
  z = ((CUREAL) (k - nz2)) * step - z0;        
  a[idx] = x * pot_funcd(SQRT(x*x + y*y + z*z), rmin, radd, a0, a1, a2, a3, a4, a5);
}

extern "C" void grid_func7b_cuda_operate_oneW(CUREAL *grid, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, CUREAL x0, CUREAL y0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func7b_cuda_operate_one_gpu<<<blocks,threads>>>(grid, rmin, radd, a0, a1, a2, a3, a4, a5, nx, ny, nz, 2 * (nz / 2 + 1), nx/2, ny/2, nz/2, x0, y0, z0, step);
  cuda_error_check();
}

/****************** d/dy of the potential function **********************/

__global__ void grid_func7c_cuda_operate_one_gpu(CUREAL *a, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, INT nzz, INT nx2, INT ny2, INT nz2, REAL x0, REAL y0, REAL z0, REAL step) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL x, y, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;
  x = ((CUREAL) (i - nx2)) * step - x0;
  y = ((CUREAL) (j - ny2)) * step - y0;    
  z = ((CUREAL) (k - nz2)) * step - z0;        
  a[idx] = y * pot_funcd(SQRT(x*x + y*y + z*z), rmin, radd, a0, a1, a2, a3, a4, a5);
}

extern "C" void grid_func7c_cuda_operate_oneW(CUREAL *grid, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, CUREAL x0, CUREAL y0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func7c_cuda_operate_one_gpu<<<blocks,threads>>>(grid, rmin, radd, a0, a1, a2, a3, a4, a5, nx, ny, nz, 2 * (nz / 2 + 1), nx/2, ny/2, nz/2, x0, y0, z0, step);
  cuda_error_check();
}

/****************** d/dz of the potential function **********************/

__global__ void grid_func7d_cuda_operate_one_gpu(CUREAL *a, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, INT nzz, INT nx2, INT ny2, INT nz2, REAL x0, REAL y0, REAL z0, REAL step) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL x, y, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;
  x = ((CUREAL) (i - nx2)) * step - x0;
  y = ((CUREAL) (j - ny2)) * step - y0;    
  z = ((CUREAL) (k - nz2)) * step - z0;        
  a[idx] = z * pot_funcd(SQRT(x*x + y*y + z*z), rmin, radd, a0, a1, a2, a3, a4, a5);
}

extern "C" void grid_func7d_cuda_operate_oneW(CUREAL *grid, CUREAL rmin, CUREAL radd, CUREAL a0, CUREAL a1, CUREAL a2, CUREAL a3, CUREAL a4, CUREAL a5, INT nx, INT ny, INT nz, CUREAL x0, CUREAL y0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func7d_cuda_operate_one_gpu<<<blocks,threads>>>(grid, rmin, radd, a0, a1, a2, a3, a4, a5, nx, ny, nz, 2 * (nz / 2 + 1), nx/2, ny/2, nz/2, x0, y0, z0, step);
  cuda_error_check();
}
