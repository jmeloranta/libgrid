/*
 * CUDA device code for wf analyze.
 *
 * TODO: Many routines have periodic BCs hardcoded...
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
 * Velocity X.
 *
 * USES FINITE DIFFERENCE!!
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

  wp = wp * CUCONJ(wm) / (GRID_EPS + CUCONJ(wp) * wm);
  if(CUCABS(wp) < GRID_EPS) vx[idx2] = 0.0;
  else vx[idx2] = inv_delta * CUCARG(wp);
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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_velocity_x_gpu<<<blocks,threads>>>(gwf, vx, inv_delta, cutoff, nx, ny, nz, nz2);
  cuda_error_check();
}

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

  wp = wp * CUCONJ(wm) / (GRID_EPS + CUCONJ(wp) * wm);
  if(CUCABS(wp) < GRID_EPS) vy[idx2] = 0.0;
  else vy[idx2] = inv_delta * CUCARG(wp);
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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

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

  wp = wp * CUCONJ(wm) / (GRID_EPS + CUCONJ(wp) * wm);
  if(CUCABS(wp) < GRID_EPS) vz[idx2] = 0.0;
  else vz[idx2] =  inv_delta * CUCARG(wp);
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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_velocity_z_gpu<<<blocks,threads>>>(gwf, vz, inv_delta, cutoff, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * Set up LOG(wf / wf*) for differentiation in the Fourier space (FFT based velocity).
 *
 */

__global__ void grid_cuda_wf_fft_velocity_setup_gpu(CUCOMPLEX *wf, CUREAL *veloc, CUREAL c, INT nx, INT ny, INT nz, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, idx2;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  idx2 = (i * ny + j) * nz2 + k;
  veloc[idx2] = c * CUCARG(wf[idx] / CUCONJ(wf[idx]));
}

/*
 * Velocity grid setup (for FFT)
 *
 * wf       = Source for operation (REAL complex *; input).
 * veloc    = Destination grid (CUREAL *; output).
 * c        = hbar / (2 * mass) (CUREAL; input).
 * cutoff   = Velocity cutoff limit (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nzz      = # of points along z for real grid (INT).
 *
 * In real space.
 *
 */

extern "C" void grid_cuda_wf_fft_velocity_setupW(CUCOMPLEX *gwf, CUREAL *veloc, CUREAL c, INT nx, INT ny, INT nz, INT nzz) {

  INT i, ngpu2 = dst->nGPUs, ngpu1 = nx % ngpus, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks1((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Full set of indices
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx1 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  dim3 blocks2((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,   // Partial set
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nnx2 + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  for(i = 0; i < ngpu1; i++) { // Full sets
    CudaSetDevice(dst->GPUs[i]);
    rgrid_cuda_wf_fft_velocity_setup_gpu<<<blocks1,threads>>>((CUCOMPLEX *) gwf->data[i], (CUREAL *) veloc->data[i], c, nnx1, ny, nz, nzz);
  }

  for(i = ngpu1; i < ngpu2; i++) { // Partial sets
    CudaSetDevice(dst->GPUs[i]);
    cgrid_cuda_wf_fft_velocity_setup_gpu<<<blocks2,threads>>>((CUCOMPLEX *) gwf->data[i], (CUREAL *) veloc->data[i], c, nnx2, ny, nz, nzz);
  }

  cuda_error_check();
}

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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

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

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_probability_flux_z_gpu<<<blocks,threads>>>(gwf, flux, inv_delta, nx, ny, nz, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/


/*
 * LX.
 *
 */

__global__ void grid_cuda_wf_lx_gpu(CUCOMPLEX *wf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, INT ny2, INT nz2, CUREAL y0, CUREAL z0, CUREAL step) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp0, wp1, wm1, wp2, wm2;
  REAL y, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nzz + k;
 
  y = ((REAL) (j - ny2)) * step - y0;
  z = ((REAL) (k - nz2)) * step - z0;    

  wp0 = wf[(i * ny + j) * nz + k];

  // y
  if(j == 0) wm1 = wf[(i * ny + ny - 1) * nz + k];
  else wm1 = wf[(i * ny + j - 1) * nz + k];
  if(j == ny - 1) wp1 = wf[i * ny * nz + k];
  else wp1 = wf[(i * ny + j + 1) * nz + k];

  // z
  if(k == 0) wm2 = wf[(i * ny + j) * nz + nz - 1];
  else wm2 = wf[(i * ny + j) * nz + k - 1];
  if(k == nz - 1) wp2 = wf[(i * ny + j) * nz];
  else wp2 = wf[(i * ny + j) * nz + k + 1];

  workspace[idx2] = (y * CUCIMAG(CUCONJ(wp0) * (wp2 - wm2)) /* y * p_z */
                    -z * CUCIMAG(CUCONJ(wp0) * (wp1 - wm1)) /* -z * p_y */
                    ) * inv_delta;
}

/*
 * L_x
 *
 * wf       = Source for operation (REAL complex *; input).
 * workspace= Workspace grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nzz      = # of points along z for real grid (INT).
 * y0       = origin y0 (CUREAL).
 * z0       = origin z0 (CUREAL).
 * step     = step length (CUREAL).
 *
 */

extern "C" void grid_cuda_wf_lxW(CUCOMPLEX *gwf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, CUREAL y0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_lx_gpu<<<blocks,threads>>>(gwf, workspace, inv_delta, nx, ny, nz, nzz, ny/2, nz/2, y0, z0, step);
  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * LY.
 *
 */

__global__ void grid_cuda_wf_ly_gpu(CUCOMPLEX *wf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, INT nx2, INT nz2, CUREAL x0, CUREAL z0, CUREAL step) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp0, wp1, wm1, wp2, wm2;
  REAL x, z;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nzz + k;
 
  x = ((REAL) (i - nx2)) * step - x0;
  z = ((REAL) (k - nz2)) * step - z0;    

  wp0 = wf[(i * ny + j) * nz + k];

  // x
  if(i == 0) wm1 = wf[((nx-1)*ny + j)*nz + k];
  else wm1 = wf[((i-1)*ny + j)*nz + k];
  if(i == nx-1) wp1 = wf[j*nz + k]; // i = 0
  else wp1 = wf[((i+1)*ny + j)*nz + k];

  // z
  if(k == 0) wm2 = wf[(i*ny + j)*nz + nz-1];
  else wm2 = wf[(i*ny + j)*nz + k-1];
  if(k == nz-1) wp2 = wf[(i*ny + j)*nz];
  else wp2 = wf[(i*ny + j)*nz + k+1];

  workspace[idx2] = (z * CUCIMAG(CUCONJ(wp0) * (wp1 - wm1)) /* z * p_x */
                    -x * CUCIMAG(CUCONJ(wp0) * (wp2 - wm2)) /* -x * p_z */
                    ) * inv_delta;
}

/*
 * L_y
 *
 * wf       = Source for operation (REAL complex *; input).
 * workspace= Workspace grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nzz      = # of points along z for real grid (INT).
 * x0       = origin x0 (CUREAL).
 * z0       = origin z0 (CUREAL).
 * step     = step length (CUREAL).
 *
 */

extern "C" void grid_cuda_wf_lyW(CUCOMPLEX *gwf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, CUREAL x0, CUREAL z0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_ly_gpu<<<blocks,threads>>>(gwf, workspace, inv_delta, nx, ny, nz, nzz, nx/2, nz/2, x0, z0, step);
  cuda_error_check();
}

/********************************************************************************************************************/

/*
 * LZ.
 *
 */

__global__ void grid_cuda_wf_lz_gpu(CUCOMPLEX *wf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, INT nx2, INT ny2, CUREAL x0, CUREAL y0, CUREAL step) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx2;
  CUCOMPLEX wp0, wp1, wm1, wp2, wm2;
  REAL x, y;

  if(i >= nx || j >= ny || k >= nz) return;

  idx2 = (i * ny + j) * nzz + k;
 
  x = ((REAL) (i - nx2)) * step - x0;
  y = ((REAL) (j - ny2)) * step - y0;    

  wp0 = wf[(i * ny + j) * nz + k];

  // x
  if(i == 0) wm1 = wf[((nx-1)*ny + j)*nz + k];
  else wm1 = wf[((i-1)*ny + j)*nz + k];
  if(i == nx-1) wp1 = wf[j*nz + k]; // i = 0
  else wp1 = wf[((i+1)*ny + j)*nz + k];

  // y
  if(j == 0) wm2 = wf[(i * ny + ny - 1) * nz + k];
  else wm2 = wf[(i * ny + j - 1) * nz + k];
  if(j == ny - 1) wp2 = wf[i * ny * nz + k];
  else wp2 = wf[(i * ny + j + 1) * nz + k];

  workspace[idx2] = (x * CUCIMAG(CUCONJ(wp0) * (wp2 - wm2)) /* x * p_y */
                    -y * CUCIMAG(CUCONJ(wp0) * (wp1 - wm1)) /* -y * p_x */
                    ) * inv_delta;
}

/*
 * L_z
 *
 * wf       = Source for operation (REAL complex *; input).
 * workspace=  Workspace grid (CUREAL *; output).
 * inv_delta= hbar / (2 * mass * step) (CUREAL; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 * nzz      = # of points along z for real grid (INT).
 * x0       = origin x0 (CUREAL).
 * y0       = origin y0 (CUREAL).
 * step     = step length (CUREAL).
 *
 */

extern "C" void grid_cuda_wf_lzW(CUCOMPLEX *gwf, CUREAL *workspace, CUREAL inv_delta, INT nx, INT ny, INT nz, INT nzz, CUREAL x0, CUREAL y0, CUREAL step) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  if(dst->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }
  CudaSetDevice(dst->GPUs[0]);

  grid_cuda_wf_lz_gpu<<<blocks,threads>>>(gwf, workspace, inv_delta, nx, ny, nz, nzz, nx/2, ny/2, x0, y0, step);
  cuda_error_check();
}
