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

/* regular */
__global__ void grid_cuda_wf_propagate_potential_gpu1(CUCOMPLEX *new, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL cons, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  new[idx] = new[idx] * CUCEXP(c * (cons + pot[idx]));
}

/* abs using imag time */
__global__ void grid_cuda_wf_propagate_potential_gpu2(CUCOMPLEX *new, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
  c.x = c.y * tmp;
  c.y *= 1.0 - tmp;
  new[idx] = new[idx] * CUCEXP(c * (cons + pot[idx]));
}

/* abs using complex potential */
__global__ void grid_cuda_wf_propagate_potential_gpu3(CUCOMPLEX *new, CUCOMPLEX *pot, CUCOMPLEX c, CUCOMPLEX amp, CUREAL rho0, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  new[idx] = new[idx] * CUCEXP(c * (cons + pot[idx] - CUMAKE(0.0, 1.0) * amp * grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz)
    * (CUCSQNORM(b[idx]) - rho0)));
}

/*
 * Propagate potential energy in real space with absorbing boundaries.
 *
 * wf       = Source/destination grid for operation (cudaXtDesc_t *; input/output).
 * pot      = Potential grid (cudaXtDesc_t *; input).
 * time_step= Time step length (CUCOMPLEX; input).
 * add_abs  = Add complex abs potential? (char; input).
 * amp      = Amplitude for complex boundary (CUCOMPLEX; input).
 * rho0     = Target value for |psi|^2 (REAL; input).
 * cons     = Constant to add to potential (REAL; input).
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

extern "C" void grid_cuda_wf_propagate_potentialW(cudaXtDesc_t *grid, cudaXtDesc_t *pot, CUCOMPLEX time_step, char add_abs, CUCOMPLEX amp, CUREAL rho0, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  CUCOMPLEX c;
  INT i, ngpu2 = grid->nGPUs, ngpu1 = nx % gpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (nny1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (nny2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  c.x =  time_step.y / HBAR;
  c.y = -time_step.x / HBAR;

  for(i = 0; i < ngpu1; i++) { // Full sets 
    CudaSetDevice(grid->GPUs[i]);
    if(lz) {
      if(add_abs)
        grid_cuda_wf_propagate_potential_gpu3<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, amp, rho0, cons, 
           lx, hx, ly, hy, lz, hz, nnx1, ny, nz);
      else
        grid_cuda_wf_propagate_potential_gpu2<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, cons, 
           lx, hx, ly, hy, lz, hz, nnx1, ny, nz);
    } else
        grid_cuda_wf_propagate_potential_gpu3<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, cons, nnx1, ny, nz);
  }
  cuda_error_check();

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
      if(add_abs)
        grid_cuda_wf_propagate_potential_gpu3<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, amp, rho0, cons, 
           lx, hx, ly, hy, lz, hz, nnx2, ny, nz);
      else
        grid_cuda_wf_propagate_potential_gpu2<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, cons, 
           lx, hx, ly, hy, lz, hz, nnx2, ny, nz);
    } else
        grid_cuda_wf_propagate_potential_gpu3<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], (CUCOMPLEX *) pot->data[i], c, cons, nnx2, ny, nz);
  }
  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * Density
 *
 */

__global__ void grid_cuda_wf_density_gpu(CUCOMPLEX *grid, CUREAL *dens, INT nx, INT ny, INT nz, INT nzz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;  // complex
  idx2 = (i * ny + j) * nzz + k; // real

  dens[idx2] = CUCREAL(grid[idx]) * CUCREAL(grid[idx]) + CUCIMAG(grid[idx]) * CUCIMAG(grid[idx]);
}

/*
 * Density
 *
 * wf       = Source/destination grid for operation (cudaXtDesc_t *; input).
 * dens     = Density grid (cudaXtDesc_t *; output).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_densityW(cudaXtDesc_t *grid, cudaXtDesc_t *dens, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = grid->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_density_gpu<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], (CUREAL *) dens->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_density_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], (CUREAL *) dens->data[i], nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * Add complex absorbing potential.
 *
 */

__global__ void grid_cuda_wf_absorb_potential_gpu(CUCOMPLEX *gwf, CUCOMPLEX *pot, REAL amp, REAL rho0, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT seg) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii = i + seg;
  REAL g, sq;

  if(i >= nx || j >= ny || k >= nz) return;
  if((g = grid_cuda_wf_absorb(ii, j, k, lx, hx, ly, hy, lz, hz)) == 0.0) return;

  idx = (i * ny + j) * nz + k;

  sq = gwf[idx].x * gwf[idx].x + gwf[idx].y * gwf[idx].y - rho0;
  pot[idx].y -= g * amp * sq;
}

/*
 * Complex absorbing potential.
 *
 * gwf      = wavefunction grid (cudaXtDesc_t *; input).
 * pot      = potential (cudaXtDesc_t *; output).
 * amp      = amplitude of the potential (CUREAL; input).
 * rho0     = rho0 background (CUREAL; input).
 * lx       = lower index for abs boundary (INT; input).
 * hx       = upper index for abs boundary (INT; input).
 * ly       = lower index for abs boundary (INT; input).
 * hy       = upper index for abs boundary (INT; input).
 * lz       = lower index for abs boundary (INT; input).
 * hz       = upper index for abs boundary (INT; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_absorb_potentialW(cudaXtDesc_t *gwf, cudaXtDesc_t *pot, REAL amp, REAL rho0, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(gwf->GPUs[i]);
    cgrid_cuda_wf_absorb_potential_gpu<<<blocks1,threads>>>((CUCOMPLEX *) gwf->data[i], (CUCOMPLEX *) pot->data[i], amp, rho0, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(gwf->GPUs[i]);
    cgrid_cuda_wf_absorb_potential_gpu<<<blocks2,threads>>>((CUCOMPLEX *) gwf->data[i], (CUCOMPLEX *) pot->data[i], amp, rho0, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  }

  cuda_error_check();
}
