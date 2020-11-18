/*
 * CUDA device code for wf analyze.
 *
 * TODO: Many routines have periodic BCs hardcoded...
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "defs.h"
#include "cuda-vars.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Flux X (finite difference).
 *
 */

__global__ void grid_cuda_wf_fd_probability_flux_x_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nz2 + k;
 
  if(i == 0) wm = wf[((nx-1)*ny + j)*nz + k];
  else wm = wf[((i-1)*ny + j)*nz + k];
  if(i == nx-1) wp = wf[j*nz + k]; // i = 0
  else wp = wf[((i+1)*ny + j)*nz + k];

  flux[idx2] = inv_delta * CUCIMAG(CUCONJ(wf[idx]) * (wp - wm));
}

/*
 * Flux x (finite difference).
 *
 * wf       = Source for operation (gpu_mem_block *; input).
 * flux     = Destination grid (gpu_mem_block *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_fd_probability_flux_xW(gpu_mem_block *gwf, gpu_mem_block *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  flux->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *FLUX = flux->gpu_info->descriptor;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_probability_flux_x wrong subformat.\n");
    abort();
  }

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);
  grid_cuda_wf_fd_probability_flux_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) GWF->data[0], (CUREAL *) FLUX->data[0], inv_delta, nx, ny, nz, nz2);

  cuda_error_check();
}

/*
 * Flux Y (finite difference).
 *
 */

__global__ void grid_cuda_wf_fd_probability_flux_y_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nz2 + k;

  if(j == 0) wm = wf[(i*ny + ny-1)*nz + k];
  else wm = wf[(i*ny + j - 1)*nz + k];
  if(j == ny-1) wp = wf[i*ny*nz + k];
  else wp = wf[(i*ny + j + 1)*nz + k];

  flux[idx2] = inv_delta * CUCIMAG(CUCONJ(wf[idx]) * (wp - wm));
}

/*
 * Flux y (finite difference).
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input).
 * flux     = Flux grid (gpu_mem_block *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_fd_probability_flux_yW(gpu_mem_block *gwf, gpu_mem_block *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  flux->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *FLUX = flux->gpu_info->descriptor;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_probability_flux_y wrong subformat.\n");
    abort();
  }

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);
  grid_cuda_wf_fd_probability_flux_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) GWF->data[0], (CUREAL *) FLUX->data[0], inv_delta, nx, ny, nz, nz2);

  cuda_error_check();
}

/*
 * Flux Z (finite difference).
 *
 */

__global__ void grid_cuda_wf_fd_probability_flux_z_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nz2 + k;
 
  if(k == 0) wm = wf[(i*ny + j)*nz + nz-1];
  else wm = wf[(i*ny + j)*nz + k-1];
  if(k == nz-1) wp = wf[(i*ny + j)*nz];
  else wp = wf[(i*ny + j)*nz + k+1];

  flux[idx2] = inv_delta * CUCIMAG(CUCONJ(wf[idx]) * (wp - wm));
}

/*
 * Flux z (finite difference).
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input).
 * flux     = Flux grid (gpu_mem_block *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_fd_probability_flux_zW(gpu_mem_block *gwf, gpu_mem_block *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  flux->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *FLUX = flux->gpu_info->descriptor;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_probability_flux_z wrong subformat.\n");
    abort();
  }

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);
  grid_cuda_wf_fd_probability_flux_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) GWF->data[0], (CUREAL *) FLUX->data[0], inv_delta, nx, ny, nz, nz2);

  cuda_error_check();
}

/*
 * Entropy.
 *
 */

__global__ void grid_cuda_wf_entropy_gpu(CUCOMPLEX *grd, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUCOMPLEX s;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  s = grd[idx];
  grd[idx].x = s.x * LOG(GRID_EPS + s.x);
  grd[idx].y = 0.0;
}

/*
 * Entropy.
 *
 * grid     = Source/destination for operation (gpu_mem_block *; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void grid_cuda_wf_entropyW(gpu_mem_block *grid, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): wf_entropy wrong subformat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_entropy_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], nnx1, nny1, nz);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_entropy_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], nnx2, nny2, nz);
  }

  cuda_error_check();
}
