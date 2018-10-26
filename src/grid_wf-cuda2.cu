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

/*
 * Velocity X.
 *
 */

__global__ void grid_cuda_wf_velocity_x_gpu(CUCOMPLEX *wf, CUREAL *vx, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nz2 + k;
 
  if(i == 0) wm = wf[((nx-1)*ny + j)*nz + k];
  else wm = wf[((i-1)*ny + j)*nz + k];
  if(i == nx-1) wp = wf[j*nz + k]; // i = 0
  else wp = wf[((i+1)*ny + j)*nz + k];

  vx[idx2] = inv_delta * CUCIMAG(CUCLOG(wp * CUCONJ(wm) / (CUCONJ(wp) * wm)));
  if(vx[idx2] > cutoff) vx[idx2] = cutoff;
  if(vx[idx2] < -cutoff) vx[idx2] = -cutoff;
}

/*
 * Velocity x
 *
 * wf       = Source for operation (REAL complex *; input).
 * vx       = Destination grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * cutoff   = Velocity cutoff limi (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_velocity_xW(CUCOMPLEX *gwf, CUREAL *vx, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_velocity_x_gpu<<<blocks,threads>>>(gwf, vx, inv_delta, cutoff, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/********************************************************************************************************************/

/*
 * Velocity Y.
 *
 */

__global__ void grid_cuda_wf_velocity_y_gpu(CUCOMPLEX *wf, CUREAL *vy, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nz2 + k;
 
  if(j == 0) wm = wf[(i*ny + ny-1)*nz + k];
  else wm = wf[(i*ny + j - 1)*nz + k];
  if(j == ny-1) wp = wf[i*ny*nz + k];
  else wp = wf[(i*ny + j + 1)*nz + k];

  vy[idx2] = inv_delta * CUCIMAG(CUCLOG(wp * CUCONJ(wm) / (CUCONJ(wp) * wm)));
  if(vy[idx2] > cutoff) vy[idx2] = cutoff;
  if(vy[idx2] < -cutoff) vy[idx2] = -cutoff;
}

/*
 * Velocity y
 *
 * wf       = Source for operation (REAL complex *; input).
 * vy       = Destination grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * cutoff   = Velocity cutoff limi (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_velocity_yW(CUCOMPLEX *gwf, CUREAL *vy, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_velocity_y_gpu<<<blocks,threads>>>(gwf, vy, inv_delta, cutoff, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/********************************************************************************************************************/

/*
 * Velocity Z.
 *
 */

__global__ void grid_cuda_wf_velocity_z_gpu(CUCOMPLEX *wf, CUREAL *vz, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp, wm;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nz2 + k;
 
  if(k == 0) wm = wf[(i*ny + j)*nz + nz-1];
  else wm = wf[(i*ny + j)*nz + k-1];
  if(k == nz-1) wp = wf[(i*ny + j)*nz];
  else wp = wf[(i*ny + j)*nz + k+1];

  vz[idx2] =  inv_delta * CUCIMAG(CUCLOG(wp * CUCONJ(wm) / (CUCONJ(wp) * wm)));
  if(vz[idx2] > cutoff) vz[idx2] = cutoff;
  if(vz[idx2] < -cutoff) vz[idx2] = -cutoff;
}

/*
 * Velocity z
 *
 * wf       = Source for operation (REAL complex *; input).
 * vz       = Destination grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * cutoff   = Velocity cutoff limi (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_velocity_zW(CUCOMPLEX *gwf, CUREAL *vz, CUREAL inv_delta, CUREAL cutoff, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_velocity_z_gpu<<<blocks,threads>>>(gwf, vz, inv_delta, cutoff, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * Flux X.
 *
 */

__global__ void grid_cuda_wf_probability_flux_x_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

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
 * Flux x
 *
 * wf       = Source for operation (REAL complex *; input).
 * flux     = Destination grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_probability_flux_xW(CUCOMPLEX *gwf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_probability_flux_x_gpu<<<blocks,threads>>>(gwf, flux, inv_delta, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/********************************************************************************************************************/

/*
 * Flux Y.
 *
 */

__global__ void grid_cuda_wf_probability_flux_y_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

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
 * Flux y
 *
 * wf       = Source/destination grid for operation (REAL complex *; input).
 * flux     = Flux grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_probability_flux_yW(CUCOMPLEX *gwf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_probability_flux_y_gpu<<<blocks,threads>>>(gwf, flux, inv_delta, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/********************************************************************************************************************/

/*
 * Flux Z.
 *
 */

__global__ void grid_cuda_wf_probability_flux_z_gpu(CUCOMPLEX *wf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

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
 * Flux z
 *
 * wf       = Source/destination grid for operation (REAL complex *; input).
 * flux     = Flux grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nz2      = # of points along z for real grid (INT).
 *
 */

extern "C" void grid_cuda_wf_probability_flux_zW(CUCOMPLEX *gwf, CUREAL *flux, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nz2) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_probability_flux_z_gpu<<<blocks,threads>>>(gwf, flux, inv_delta, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/
