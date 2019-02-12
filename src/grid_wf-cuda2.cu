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
#include "grid_wf-cuda.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/********************************************************************************************************************/

/*
 * Potential energy propagation in real space (possibly with absorbing boundaries).
 *
 */

/* amp = 0 */
__global__ void grid_cuda_wf_propagate_potential_gpu1(CUCOMPLEX *b, CUCOMPLEX *pot, CUCOMPLEX c, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  b[idx] = b[idx] * CUCEXP(c * pot[idx]);
}

/* amp != 0 */
__global__ void grid_cuda_wf_propagate_potential_gpu2(CUCOMPLEX *b, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  c = c * grid_cuda_wf_absorb(i, j, k, amp, lx, hx, ly, hy, lz, hz);
  b[idx] = b[idx] * CUCEXP(c * pot[idx]);
}

/*
 * Propagate potential energy in real space with absorbing boundaries.
 *
 * wf       = Source/destination grid for operation (REAL complex *; input/output).
 * pot      = Potential grid (CUCOMPLEX *; input).
 * time_step= Time step length (CUCOMPLEX; input).
 * amp      = Max amplitude for imag. part (CUREAL; input).
 * lx       = Lower bound for absorbing bc (INT; input).
 * hx       = Upper bound for absorbing bc (INT; input).
 * ly       = Lower bound for absorbing bc (INT; input).
 * hy       = Upper bound for absorbing bc (INT; input).
 * lz       = Lower bound for absorbing bc (INT; input).
 * hz       = Upper bound for absorbing bc (INT; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void grid_cuda_wf_propagate_potentialW(CUCOMPLEX *grid, CUCOMPLEX *pot, CUCOMPLEX time_step, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  CUCOMPLEX c;

  c.x =  (1.0 / HBAR) * time_step.y;
  c.y = -(1.0 / HBAR) * time_step.x;
  if(amp != 0.0) 
    grid_cuda_wf_propagate_potential_gpu2<<<blocks,threads>>>(grid, pot, c, amp, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  else
    grid_cuda_wf_propagate_potential_gpu1<<<blocks,threads>>>(grid, pot, c, nx, ny, nz);
  cuda_error_check();
}

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
