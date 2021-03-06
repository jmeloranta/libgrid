/*
 * CUDA device code (REAL complex; cgrid).
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "cuda-vars.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Kinetic energy propagation in Fourier space.
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_fft_gpu(CUCOMPLEX *b, CUREAL cx, CUREAL cy, CUREAL cz, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUCOMPLEX time_mass, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = cx * ((CUREAL) i) - kx0;
  else
    kx = cx * ((CUREAL) (i - nx)) - kx0;

  if (jj <= ny2)
    ky = cy * ((CUREAL) jj) - ky0;
  else
    ky = cy * ((CUREAL) (jj - nyy)) - ky0;

  if (k <= nz2)
    kz = cz * ((CUREAL) k) - kz0; 
  else
    kz = cz * ((CUREAL) (k - nz)) - kz0;
      
  /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
  b[idx] = b[idx] * CUCEXP(time_mass * (kx * kx + ky * ky + kz * kz));
}

/*
 * Propagate kinetic energy in Fourier space.
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input/output).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * time_mass= Time step & mass (CUCOMPLEX; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_propagate_kinetic_fftW(gpu_mem_block *grid, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, INT nx, INT ny, INT nz) {

  INT segx = 0, segy = 0, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUREAL cx, cy, cz;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): wf_propagate_kinetic_fft wrong subformat.\n");
    abort();
  }

  cx = 2.0 * M_PI / (((CUREAL) nx) * step);
  cy = 2.0 * M_PI / (((CUREAL) ny) * step);
  cz = 2.0 * M_PI / (((CUREAL) nz) * step);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_fft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], cx, cy, cz, kx0, ky0, kz0, time_mass, 
      nnx1, nny1, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_fft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], cx, cy, cz, kx0, ky0, kz0, time_mass, 
      nnx2, nny2, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * Kinetic energy propagation in Fourier space (with Lanczos cutoff).
 *
 * TODO: Can we improve performance by optimizing the if statements?
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cfft_gpu(CUCOMPLEX *b, CUREAL cx, CUREAL cy, CUREAL cz, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUCOMPLEX time_mass, CUREAL kmax2, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz, tot;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = cx * ((CUREAL) i) - kx0;
  else
    kx = cx * ((CUREAL) (i - nx)) - kx0;

  if (jj <= ny2)
    ky = cy * ((CUREAL) jj) - ky0;
  else
    ky = cy * ((CUREAL) (jj - nyy)) - ky0;

  if (k <= nz2)
    kz = cz * ((CUREAL) k) - kz0; 
  else
    kz = cz * ((CUREAL) (k - nz)) - kz0;
      
  /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
  tot = kx * kx + ky * ky + kz * kz;
  if(tot < kmax2)
    b[idx] = b[idx] * CUCEXP(time_mass * tot);
  else
    b[idx] = CUMAKE(0.0, 0.0); 
}

/*
 * Propagate kinetic energy in Fourier space (CFFT) with Lanczos cutoff.
 *
 * wf       = Source/destination grid for operation (gpu_mem_block *; input/output).
 * kx0      = Momentum shift of origin along X (CUREAL; input).
 * ky0      = Momentum shift of origin along Y (CUREAL; input).
 * kz0      = Momentum shift of origin along Z (CUREAL; input).
 * step     = Spatial step length (CUREAL; input).
 * time_mass= Time step & mass (CUCOMPLEX; input).
 * kmax2    = Maximum |k-vector|^2 to propagate (CUREAL).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cfftW(gpu_mem_block *grid, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, CUREAL kmax2, INT nx, INT ny, INT nz) {

  INT segx = 0, segy = 0, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  CUREAL cx, cy, cz;

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): wf_propagate_kinetic_fft wrong subformat.\n");
    abort();
  }

  cx = 2.0 * M_PI / (((CUREAL) nx) * step);
  cy = 2.0 * M_PI / (((CUREAL) ny) * step);
  cz = 2.0 * M_PI / (((CUREAL) nz) * step);

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_cfft_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], cx, cy, cz, kx0, ky0, kz0,
        time_mass, kmax2, nnx1, nny1, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    grid_cuda_wf_propagate_kinetic_cfft_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], cx, cy, cz, kx0, ky0, kz0,
        time_mass, kmax2, nnx2, nny2, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
