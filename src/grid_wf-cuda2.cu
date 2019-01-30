/*
 * CUDA device code for wf.
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "defs.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/********************************************************************************************************************/

/*
 * Density
 *
 */

__global__ void grid_cuda_wf_density_gpu(CUCOMPLEX *b, CUREAL *dens, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nz2 + k;

  dens[idx2] = CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]);
}

/*
 * Density
 *
 * wf       = Source/destination grid for operation (REAL complex *; input).
 * dens     = Density grid (CUCOMPLEX *; output).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_densityW(CUCOMPLEX *grid, CUREAL *dens, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_density_gpu<<<blocks,threads>>>(grid, dens, nx, ny, nz, 2 * (nz / 2 + 1));
  cuda_error_check();
}

/********************************************************************************************************************/
