/*
 * CUDA device code (REAL complex; cgrid).
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "cuda-vars.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Kinetic energy propagation in Fourier space.
 *
 * Only periodic version implemented.
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_fft_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL cx, CUREAL cy, CUREAL cz, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL maxk, CUCOMPLEX time_mass, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz, kk;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = cx * ((CUREAL) i) - kx0;
  else
    kx = cx * ((CUREAL) (i - nx)) - kx0;

  if (jj <= ny2)
    ky = cy * ((CUREAL) jj) - ky0;
  else
    ky = cy * ((CUREAL) (jj - ny)) - ky0;

  if (k <= nz2)
    kz = cz * ((CUREAL) k) - kz0; 
  else
    kz = cz * ((CUREAL) (k - nz)) - kz0;
      
  kk = kx * kx + ky * ky + kz * kz;
  /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
  b[idx] = b[idx] * CUCEXP(time_mass * kk) * norm;
}

/*
 * Propagate kinetic energy in Fourier space.
 *
 * wf       = Source/destination grid for operation (cudaXtDesc *; input/output).
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * time_mass= Time step & mass (CUCOMPLEX; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void grid_cuda_wf_propagate_kinetic_fftW(cudaXtDesc *grid, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, INT nx, INT ny, INT nz) {

  INT space = 1, segx = 0, segy = 0, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  SETUP_VARIABLES_SEG(grid);
  CUREAL cx, cy, cz;

  cx = 2.0 * M_PI / (((CUREAL) nx) * step);
  cy = 2.0 * M_PI / (((CUREAL) ny) * step);
  cz = 2.0 * M_PI / (((CUREAL) nz) * step);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_fft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], norm, cx, cy, cz, kx0, ky0, kz0, M_PI / step, 
        time_mass, nnx1, nny1, nz, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_fft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], norm, cx, cy, cz, kx0, ky0, kz0, M_PI / step, 
        time_mass, nnx2, nny2, nz, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * Kinetic energy propagation in Fourier space (Cayley).
 *
 * Only periodic version implemented.
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cfft_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL cx, CUREAL cy, CUREAL cz, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL maxk, CUCOMPLEX time_mass, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2, INT seg) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz, kk;
  CUCOMPLEX tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = cx * ((CUREAL) i) - kx0;
  else
    kx = cx * ((CUREAL) (i - nx)) - kx0;

  if (jj <= ny2)
    ky = cy * ((CUREAL) jj) - ky0;
  else
    ky = cy * ((CUREAL) (jj - ny)) - ky0;

  if (k <= nz2)
    kz = cz * ((CUREAL) k) - kz0; 
  else
    kz = cz * ((CUREAL) (k - nz)) - kz0;
      
  kk = kx * kx + ky * ky + kz * kz;
  /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
  /* exp ~ (1 + 0.5 * x) / (1 - 0.5 * x) */
  tmp = 0.5 * time_mass * kk;
  b[idx] = b[idx] * norm * (1.0 + tmp) / (1.0 - tmp);
}

/*
 * Propagate kinetic energy in Fourier space (CFFT).
 *
 * wf       = Source/destination grid for operation (cudaXtDesc *; input/output).
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * time_mass= Time step & mass (CUCOMPLEX; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cfftW(cudaXtDesc *grid, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, INT nx, INT ny, INT nz) {

  INT space = 1, segx = 0, segy = 0, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  SETUP_VARIABLES_SEG(grid);
  CUREAL cx, cy, cz;

  cx = 2.0 * M_PI / (((CUREAL) nx) * step);
  cy = 2.0 * M_PI / (((CUREAL) ny) * step);
  cz = 2.0 * M_PI / (((CUREAL) nz) * step);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_cfft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) grid->data[i], norm, cx, cy, cz, kx0, ky0, kz0, M_PI / step, 
        time_mass, nnx1, nny1, nz, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(grid->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_cfft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) grid->data[i], norm, cx, cy, cz, kx0, ky0, kz0, M_PI / step, 
        time_mass, nnx2, nny2, nz, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
