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

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/********************************************************************************************************************/

/*
 * Kinetic energy propagation in Fourier space.
 *
 * Only periodic version implemented.
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_fft_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL cx, CUREAL cy, CUREAL cz, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, INT nx, INT ny, INT nz, INT nx2, INT ny2, INT nz2) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = cx * ((CUREAL) i) - kx0;
  else
    kx = cx * ((CUREAL) (i - nx)) - kx0;

  if (j <= ny2)
    ky = cy * ((CUREAL) j) - ky0;
  else
    ky = cy * ((CUREAL) (j - ny)) - ky0;

  if (k <= nz2)
    kz = cz * ((CUREAL) k) - kz0; 
  else
    kz = cz * ((CUREAL) (k - nz)) - kz0;
      
  /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
  b[idx] = b[idx] * CUCEXP(time_mass * (kx * kx + ky * ky + kz * kz)) * norm;
}

/*
 * Propagate kinetic energy in Fourier space.
 *
 * wf       = Source/destination grid for operation (REAL complex *; input/output).
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

extern "C" void grid_cuda_wf_propagate_kinetic_fftW(CUCOMPLEX *grid, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, CUCOMPLEX time_mass, INT nx, INT ny, INT nz) {

  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  CUREAL cx, cy, cz;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  cx = 2.0 * M_PI / (((CUREAL) nx) * step);
  cy = 2.0 * M_PI / (((CUREAL) ny) * step);
  cz = 2.0 * M_PI / (((CUREAL) nz) * step);

  grid_cuda_wf_propagate_kinetic_fft_gpu<<<blocks,threads>>>(grid, norm, cx, cy, cz, kx0, ky0, kz0, step, time_mass, nx, ny, nz, nx2, ny2, nz2);
  cuda_error_check();
}

/********************************************************************************************************************/
