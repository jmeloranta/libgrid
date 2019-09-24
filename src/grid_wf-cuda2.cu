/*
 * CUDA device code for wf.
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "grid_wf-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

extern void *grid_gpu_mem;
extern cudaXtDesc *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Potential energy propagation in real space (possibly with absorbing boundaries).
 *
 */

__global__ void grid_cuda_wf_propagate_potential_gpu1(CUCOMPLEX *dst, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL c2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] * CUCEXP(c * (c2 + pot[idx]));
}

/* abs using imag time */
__global__ void grid_cuda_wf_propagate_potential_gpu2(CUCOMPLEX *dst, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
  c.x = c.y * tmp;
  c.y *= 1.0 - tmp;
  dst[idx] = dst[idx] * CUCEXP(c * (cons + pot[idx]));
}

/* abs using complex potential */
__global__ void grid_cuda_wf_propagate_potential_gpu3(CUCOMPLEX *dst, CUCOMPLEX *pot, CUCOMPLEX c, CUCOMPLEX amp, CUREAL rho0, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] * CUCEXP(c * (cons + pot[idx] - CUMAKE(0.0, 1.0) * amp * grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz) * (CUCSQNORM(dst[idx]) - rho0)));
}

/*
 * Propagate potential energy in real space with absorbing boundaries.
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input/output).
 * pot      = Potential grid (gpu_mem_block *; input).
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

extern "C" void grid_cuda_wf_propagate_potentialW(gpu_mem_block *grid, gpu_mem_block *pot, CUCOMPLEX time_step, char add_abs, CUCOMPLEX amp, CUREAL rho0, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  CUCOMPLEX c;
  SETUP_VARIABLES(grid);  
  cudaXtDesc *GRID = grid->gpu_info->descriptor, *POT = pot->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || pot->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): propagate_potential wrong subformat.\n");
    abort();
  }

  c.x =  time_step.y / HBAR;
  c.y = -time_step.x / HBAR;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    if(lz) {
      if(add_abs)
        grid_cuda_wf_propagate_potential_gpu3<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, amp, rho0, cons, 
           lx, hx, ly, hy, lz, hz, nnx1, nny1, nz);
      else
        grid_cuda_wf_propagate_potential_gpu2<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, 
           lx, hx, ly, hy, lz, hz, nnx1, nny1, nz);
    } else
        grid_cuda_wf_propagate_potential_gpu1<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    if(lz) {
      if(add_abs) 
        grid_cuda_wf_propagate_potential_gpu3<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, amp, rho0, cons, 
           lx, hx, ly, hy, lz, hz, nnx2, nny1, nz);
      else
        grid_cuda_wf_propagate_potential_gpu2<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, 
           lx, hx, ly, hy, lz, hz, nnx2, nny2, nz);
    } else
        grid_cuda_wf_propagate_potential_gpu1<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, nnx2, nny2, nz);
  }

  cuda_error_check();
}

/*
 * Density
 *
 */

__global__ void grid_cuda_wf_density_gpu(CUCOMPLEX *grid, CUREAL *dens, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;  // complex
  idx2 = (i * ny + j) * nzz + k; // real

  dens[idx2] = CUCREAL(grid[idx]) * CUCREAL(grid[idx]) + CUCIMAG(grid[idx]) * CUCIMAG(grid[idx]);
}

/*
 * Density
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input).
 * dens     = Density grid (gpu_mem_block *; output).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_densityW(gpu_mem_block *grid, gpu_mem_block *dens, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_REAL(grid);  
  cudaXtDesc *GRID = grid->gpu_info->descriptor, *DENS = dens->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_density wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_density_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUREAL *) DENS->data[i], nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_density_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUREAL *) DENS->data[i], nnx2, ny, nz, nzz);
  }

  dens->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();
}

/*
 * Add complex absorbing potential.
 *
 */

__global__ void grid_cuda_wf_absorb_potential_gpu(CUCOMPLEX *gwf, CUCOMPLEX *pot, REAL amp, REAL rho0, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT seg) {

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
 * gwf      = wavefunction grid (gpu_mem_block *; input).
 * pot      = potential (gpu_mem_block *; output).
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

extern "C" void grid_cuda_wf_absorb_potentialW(gpu_mem_block *gwf, gpu_mem_block *pot, REAL amp, REAL rho0, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  INT segx = 0, segy = 0;
  SETUP_VARIABLES_SEG(gwf);
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *POT = pot->gpu_info->descriptor;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || pot->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): absorb_potential wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GWF->GPUs[i]);
    grid_cuda_wf_absorb_potential_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GWF->data[i], (CUCOMPLEX *) POT->data[i], amp, rho0, lx, hx, ly, hy, lz, hz, nnx1, nny1, nz, segx);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GWF->GPUs[i]);
    grid_cuda_wf_absorb_potential_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GWF->data[i], (CUCOMPLEX *) POT->data[i], amp, rho0, lx, hx, ly, hy, lz, hz, nnx2, nny2, nz, segx);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
