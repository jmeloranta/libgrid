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
 * Potential energy propagation in real space.
 *
 */

__global__ void grid_cuda_wf_propagate_potential_gpu1(CUCOMPLEX *dst, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL c2, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = dst[idx] * CUCEXP(c * (c2 + pot[idx]));
}

/* abs using imag time */
__global__ void grid_cuda_wf_propagate_potential_gpu2(CUCOMPLEX *dst, CUCOMPLEX *pot, CUCOMPLEX c, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii = i + seg;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  tmp = grid_cuda_wf_absorb(ii, j, k, lx, hx, ly, hy, lz, hz);
  c.x = c.y * tmp;
  c.y *= 1.0 - tmp;
  dst[idx] = dst[idx] * CUCEXP(c * (cons + pot[idx]));
}

/*
 * Propagate potential energy in real space with absorbing boundaries.
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input/output).
 * pot      = Potential grid (gpu_mem_block *; input).
 * time_step= Time step length (CUCOMPLEX; input).
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

extern "C" void grid_cuda_wf_propagate_potentialW(gpu_mem_block *grid, gpu_mem_block *pot, CUCOMPLEX time_step, CUREAL cons, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  CUCOMPLEX c;
  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor, *POT = pot->gpu_info->descriptor;
  INT segx = 0, segy = 0;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || pot->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): propagate_potential wrong subformat.\n");
    abort();
  }

  c.x =  time_step.y / HBAR;
  c.y = -time_step.x / HBAR;

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    if(lz) grid_cuda_wf_propagate_potential_gpu2<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, lx, hx, ly, hy, lz, hz, nnx1, nny1, nz, segx);
    else grid_cuda_wf_propagate_potential_gpu1<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, nnx1, nny1, nz);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    if(lz) grid_cuda_wf_propagate_potential_gpu2<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, lx, hx, ly, hy, lz, hz, nnx2, nny2, nz, segx);
    else grid_cuda_wf_propagate_potential_gpu1<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], (CUCOMPLEX *) POT->data[i], c, cons, nnx2, nny2, nz);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * Density
 *
 */

__global__ void grid_cuda_wf_density_gpu(CUCOMPLEX *grid, CUREAL *dens, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;
  CUCOMPLEX d;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;  // complex
  idx2 = (i * ny + j) * nzz + k; // real

  d = grid[idx];
  dens[idx2] = d.x * d.x + d.y * d.y;
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

  dens->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  SETUP_VARIABLES_REAL(dens);  
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

  cuda_error_check();
}

/*
 * wf merging
 *
 */

__global__ void grid_cuda_wf_merge_gpu(CUCOMPLEX *dst, CUCOMPLEX *wfr, CUCOMPLEX *wfi, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, ii = seg + i;
  CUREAL alpha;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;  // complex

  alpha = grid_cuda_wf_absorb(ii, j, k, lx, hx, ly, hy, lz, hz);
  dst[idx] = CUMAKE(1.0 - alpha, 0.0) * wfr[idx] + CUMAKE(alpha, 0.0) * wfi[idx];
}

/*
 * wf merging
 *
 * wf    = Resulting wave functon (wf *; output).
 * wfr   = Wave function from propagating in real time (wf *; input).
 * wfi   = Wave function from propagating in imaginary time (wf *; input).
 * lx    = lower bound for X (INT; input).
 * hx    = lower bound for X (INT; input).
 * ly    = lower bound for Y (INT; input).
 * hy    = lower bound for Y (INT; input).
 * lz    = lower bound for Z (INT; input).
 * hz    = lower bound for Z (INT; input).
 * nx    = Grid dimension X (INT; input).
 * ny    = Grid dimension Y (INT; input).
 * nz    = Grid dimension Z (INT; input).
 *
 */

extern "C" void grid_cuda_wf_mergeW(gpu_mem_block *dst, gpu_mem_block *wfr, gpu_mem_block *wfi, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  SETUP_VARIABLES_SEG(dst);  
  cudaXtDesc *DST = dst->gpu_info->descriptor, *WFR = wfr->gpu_info->descriptor, *WFI = wfi->gpu_info->descriptor;
  INT segx = 0, segy = 0;

  if(wfr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || wfr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_merge wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_wf_merge_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) WFR->data[i], (CUCOMPLEX *) WFI->data[i], lx, hx, ly, hy, lz, hz, nnx1, nny1, nz, segx);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    grid_cuda_wf_merge_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) WFR->data[i], (CUCOMPLEX *) WFI->data[i], lx, hx, ly, hy, lz, hz, nnx2, nny2, nz, segx);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
