/*
 * Function #6 a & b: Thermal correction function for libdft.
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../cuda-math.h"

#include "func6.h"

extern "C" void cuda_error_check();

__device__ CUREAL lwl3(CUREAL mass, CUREAL temp) {

  CUREAL lwl;

  /* hbar = 1 */
  lwl = SQRT(2.0 * M_PI * HBAR * HBAR / (mass * GRID_AUKB * temp));
  return lwl * lwl * lwl;
}

/* # of terms to evaluate in the series */
#define NTERMS 256

__device__ CUREAL gf(CUREAL z, CUREAL s) {

  CUREAL val = 0.0, zk = 1.0;
  INT k;

  for (k = 1; k <= NTERMS; k++) {
    zk *= z;
    val += zk / POW((CUREAL) k, s);
  }
  return val;
}

__device__ CUREAL find_z0(CUREAL val) {

  /* Golden sectioning */
  CUREAL a, b, c, d, fc, fd, tmp;

  if(val >= gf(1.0, 3.0 / 2.0)) return 1.0; /* g_{3/2}(1) */

  a = 0.0;
  b = 1.0;

  c = b - (b - a) / GOLDEN;
  d = a + (b - a) / GOLDEN;

  while (FABS(c - d) > STOP) {

    tmp = val - gf(c, 3.0 / 2.0);
    fc = tmp * tmp;
    tmp = val - gf(d, 3.0 / 2.0);
    fd = tmp * tmp;

    if(fc < fd) b = d; else a = c;

    c = b - (b - a) / GOLDEN;
    d = a + (b - a) / GOLDEN;
  }
  return (b + a) / 2.0;       
}

__global__ void grid_func6a_cuda_operate_one_gpu(CUREAL *c, CUREAL *a, CUREAL mass, CUREAL temp, CUREAL c4, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop, tmp, m;
  CUREAL l3, z0, rl3, g12, g32;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = a[idx];

  l3 = lwl3(mass, temp);
  rl3 = rhop * l3;
  tmp = gf(1.0, 3.0 / 2.0);
  if(rl3 >= tmp) m = -c4 * GRID_AUKB * (temp / l3) * tmp;
  else {
    z0 = find_z0(rl3);
    g12 = gf(z0, 1.0/2.0);
    g32 = gf(z0, 3.0/2.0);
    m = c4 * GRID_AUKB * temp * (LOG(z0) + rl3 / g12 - g32 / g12);
  }
  c[idx] = m;
}

extern "C" void grid_func6a_cuda_operate_oneW(CUREAL *gridc, CUREAL *grida, CUREAL mass, CUREAL temp, CUREAL c4, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func6a_cuda_operate_one_gpu<<<blocks,threads>>>(gridc, grida, mass, temp, c4, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

__global__ void grid_func6b_cuda_operate_one_gpu(CUREAL *c, CUREAL *a, CUREAL mass, CUREAL temp, CUREAL c4, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL rhop;
  CUREAL z, l3;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  rhop = a[idx];

  l3 = lwl3(mass, temp);
  z = find_z0(rhop * l3);
  
  c[idx] = (c4 * GRID_AUKB * temp * (rhop * LOG(z) - gf(z, 5.0 / 2.0) / l3));;
}

extern "C" void grid_func6b_cuda_operate_oneW(CUREAL *gridc, CUREAL *grida, CUREAL mass, CUREAL temp, CUREAL c4, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_func6b_cuda_operate_one_gpu<<<blocks,threads>>>(gridc, grida, mass, temp, c4, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}
