/*
 * CUDA device code (REAL complex = CUCOMPLEX; cgrid) involving differentiation.
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 * Since the ordering of the data on multiple GPUs changes after FFT,
 * we must keep track of this. The variable space in cgrid structure (structs.h)
 * is zero if the grid is currently in real space and one if it is in reciprocal space (i.e., after FFT).
 * The real space data is distributed on GPUs according to the first index (x as in data[x][y][z])
 * where as the resiprocal space data is distributed along the second index (y as in data[x][y][z]).
 *
 * SETUP_VARIABLES and SETUP_VARIABLES_SEG are macros that follow the correct distribution over GPUs
 * based on the value of space variable. The latter also keeps track of the actual index so that
 * routines that need to access proper grid indices can do so.
 *
 * TODO: These are far from optimal and slow.
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "cgrid_bc-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

extern void *grid_gpu_mem;            /* host memory holder for the block used for reductions */
extern cudaXtDesc *grid_gpu_mem_addr; /* the corresponding GPU memory addresses */
extern "C" void cuda_error_check();

/*
 * dst = FD_X(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * dst = FD_X(src)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;
 
  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = FD_Y(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_y_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Y(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = FD_Z(src).
 *
 */

__global__ void cgrid_cuda_fd_gradient_z_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) - cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz));
}

/*
 * B = FD_Z(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_gradient_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz)
                         + cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz)
                         - 6.0 * src[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplacian in dst.
 *
 */

extern "C" void cgrid_cuda_fd_laplaceW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_X(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_x_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_X(src)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_Y(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_y_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_Y(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * dst = LAPLACE_Z(src).
 *
 */

__global__ void cgrid_cuda_fd_laplace_z_gpu(CUCOMPLEX *src, CUCOMPLEX *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  dst[idx] = inv_delta2 * (cgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz) + cgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz) - 2.0 * src[idx]);
}

/*
 * dst = LAPLACE_Z(src)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fd_laplace_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = src->gpu_info->subFormat;
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *SRC = src->gpu_info->descriptor, *DST = dst->gpu_info->descriptor;

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  cgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>((CUCOMPLEX *) SRC->data[0], (CUCOMPLEX *) DST->data[0], inv_delta2, bc, nx, ny, nz);

  cuda_error_check();
}

/*
 * FFT gradient_x
 *
 */

__global__ void cgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *b, CUREAL lx, INT nx, INT ny, INT nz, INT nx2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = lx * (CUREAL) i;
  else 
    kx = lx * (CUREAL) (i - nx);

  b[idx] = b[idx] * CUMAKE(0.0, kx);     // multiply by I * kx
}

/*
 * FFT gradient_x
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_xW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nx2 = nx / 2;
  REAL lx = 2.0 * M_PI / (((CUREAL) nx) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_x wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], lx, nnx1, nny1, nz, nx2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], lx, nnx2, nny2, nz, nx2);
  }

  cuda_error_check();
}

/*
 * FFT gradient_y
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *b, CUREAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (jj <= ny2)
    ky = ly * (CUREAL) jj;
  else 
    ky = ly * (CUREAL) (jj - nyy);

  b[idx] = b[idx] * CUMAKE(0.0, ky);    // multiply by I * ky
}

/*
 * FFT gradient_y
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_yW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT ny2 = ny / 2, segx = 0, segy = 0;  // segx unused
  REAL ly = 2.0 * M_PI / (((CUREAL) ny) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_y wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], ly, nnx1, nny1, nz, ny, ny2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], ly, nnx2, nny2, nz, ny, ny2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT gradient_z
 *
 * B = B' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *b, CUREAL lz, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;

  if (k <= nz2)
    kz = lz * (CUREAL) k;
  else 
    kz = lz * (CUREAL) (k - nz);

  b[idx] = b[idx] * CUMAKE(0.0, kz);   // multiply by I * kz
}

/*
 * FFT gradient_z
 *
 * dst        = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step       = Spatial step length (CUREAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_gradient_zW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nz2 = nz / 2;
  REAL lz = 2.0 * M_PI / (((CUREAL) nz) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_z wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], lz, nnx1, nny1, nz, nz2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_gradient_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], lz, nnx2, nny2, nz, nz2);
  }

  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL lx, CUREAL ly, CUREAL lz, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (i <= nx2)
    kx = lx * (CUREAL) i;
  else 
    kx = lx * (CUREAL) (i - nx);
      
  if (jj <= ny2)
    ky = ly * (CUREAL) jj;
  else 
    ky = ly * (CUREAL) (jj - nyy);
      
  if (k <= nz2)
    kz = lz * (CUREAL) k;
  else 
    kz = lz *  (CUREAL) (k - nz);

  b[idx] = -b[idx] * (kx * kx + ky * ky + kz * kz);
}

/*
 * FFT laplace
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step     = Spatial step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_laplaceW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, segx = 0, segy = 0; // segx not used
  REAL lx = 2.0 * M_PI / (((CUREAL) nx) * step), ly = 2.0 * M_PI / (((CUREAL) ny) * step), lz = 2.0 * M_PI / (((CUREAL) nz) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], lx, ly, lz, nnx1, nny1, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], lx, ly, lz, nnx2, nny2, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (X).
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_x_gpu(CUCOMPLEX *b, CUREAL lx, INT nx, INT ny, INT nz, INT nx2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (i <= nx2)
    kx = lx * (CUREAL) i;
  else 
    kx = lx * (CUREAL) (i - nx);
      
  b[idx] = -b[idx] * kx * kx;
}

/*
 * FFT laplace (X)
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step     = Spatial step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_laplace_xW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nx2 = nx / 2, segx = 0, segy = 0; // segx not used
  REAL lx = 2.0 * M_PI / (((CUREAL) nx) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], lx, nnx1, nny1, nz, nx2);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], lx, nnx2, nny2, nz, nx2);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (Y).
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_y_gpu(CUCOMPLEX *b, CUREAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  CUREAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (jj <= ny2)
    ky = ly * (CUREAL) jj;
  else 
    ky = ly * (CUREAL) (jj - nyy);
      
  b[idx] = -b[idx] * ky * ky;
}

/*
 * FFT laplace (Y)
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step     = Spatial step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_laplace_yW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT ny2 = ny / 2, segx = 0, segy = 0; // segx not used
  REAL ly = 2.0 * M_PI / (((CUREAL) ny) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], ly, nnx1, nny1, nz, ny, ny2, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], ly, nnx2, nny2, nz, ny, ny2, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (Z).
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void cgrid_cuda_fft_laplace_z_gpu(CUCOMPLEX *b, CUREAL lz, INT nx, INT ny, INT nz, INT nz2) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if (k <= nz2)
    kz = lz * (CUREAL) k;
  else 
    kz = lz * (CUREAL) (k - nz);
      
  b[idx] = -b[idx] * kz * kz;
}

/*
 * FFT laplace (Z)
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step     = Spatial step length (CUREAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_fft_laplace_zW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(dst);
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  INT nz2 = nz / 2, segx = 0, segy = 0; // segx not used
  REAL lz = 2.0 * M_PI / (((CUREAL) nz) * step);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DST->data[i], lz, nnx1, nny1, nz, nz2);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_fft_laplace_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DST->data[i], lz, nnx2, nny2, nz, nz2);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}

/*
 * FFT laplace expectation value.
 *
 * B = <B''> in Fourier space.
 *
 * Only periodic version implemented.
 *
 * Normalization done in cgrid-cuda.c
 *
 */

__global__ void cgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUCOMPLEX *blocks, CUREAL lx, CUREAL ly, CUREAL lz, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg;
  INT d = blockDim.x * blockDim.y * blockDim.z, idx, idx2, t;
  CUREAL kx, ky, kz;
  extern __shared__ CUREAL els2[];

  if(i >= nx || j >= ny || k >= nz) return;

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++)
      els2[t] = 0.0;
  }
  __syncthreads();

  idx = (i * ny + j) * nz + k;

  if (i <= nx2)
    kx = lx * (CUREAL) i;
  else 
    kx = lx * (CUREAL) (i - nx);
      
  if (jj <= ny2)
    ky = ly * (CUREAL) jj;
  else 
    ky = ly * (CUREAL) (jj - nyy);
      
  if (k <= nz2)
    kz = lz * (CUREAL) k;
  else 
    kz = lz * (CUREAL) (k - nz);

  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els2[idx2] -= (kx * kx + ky * ky + kz * kz) * (CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]));
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2].x += els2[t];  // reduce threads
    }
  }
}

/*
 * FFT laplace expectation value
 *
 * dst      = Source/destination grid for operation (gpu_mem_block *; input/output).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * value    = Expectation value (REAL; output).
 *
 * Normalization done in cgrid-cuda.c
 *
 */

extern __global__ void cgrid_cuda_block_init(CUCOMPLEX *);
extern "C" void cgrid_cuda_reduce_all(CUCOMPLEX *, INT);

extern "C" void cgrid_cuda_fft_laplace_expectation_valueW(gpu_mem_block *dst, CUREAL step, INT nx, INT ny, INT nz, CUCOMPLEX *value) {

  SETUP_VARIABLES_SEG(dst)
  cudaXtDesc *DST = dst->gpu_info->descriptor;
  CUCOMPLEX tmp;
  INT nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, segx = 0, segy = 0; // segx not used
  int s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);
  CUREAL lx = 2.0 * M_PI / (step * (CUREAL) nx), ly = 2.0 * M_PI / (step * (CUREAL) ny), lz = 2.0 * M_PI / (step * (CUREAL) nz);

  if(dst->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_expectation_value wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_block_init<<<b31/CUDA_THREADS_PER_BLOCK,CUDA_THREADS_PER_BLOCK>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i]);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], lx, ly, lz, nnx1, nny1, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx1;
    segy += dsegy1;
    cgrid_cuda_reduce_all((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b31); // reduce over blocks
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DST->GPUs[i]);
    cgrid_cuda_block_init<<<b32/CUDA_THREADS_PER_BLOCK,CUDA_THREADS_PER_BLOCK>>>((CUCOMPLEX *) grid_gpu_mem_addr->data[i]);
    // Blocks, Threads, dynamic memory size
    cgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) DST->data[i], (CUCOMPLEX *) grid_gpu_mem_addr->data[i], lx, ly, lz, nnx2, nny2, nz, ny, nx2, ny2, nz2, segy);
    segx += dsegx2;
    segy += dsegy2;
    cgrid_cuda_reduce_all((CUCOMPLEX *) grid_gpu_mem_addr->data[i], b32); // reduce over blocks
  }

  // Reduce over GPUs
  *value = CUMAKE(0.0,0.0);
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUCOMPLEX), &tmp);
    value->x += tmp.x;  /// + overloaded to device function - work around!
  }

  cuda_error_check();
}

/*
 * Poisson equation.
 *
 */

__global__ void cgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nz, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  kx = COS(ilx * (CUREAL) i);
  ky = COS(ily * (CUREAL) jj);
  kz = COS(ilz * (CUREAL) k);
  if(i || jj || k)
    grid[idx] = grid[idx] * step2 / (2.0 * (kx + ky + kz - 3.0));
  else
    grid[idx] = CUMAKE(0.0, 0.0);
}

/*
 * Solve Poisson.
 *
 * grid    = Grid specifying the RHS (gpu_mem_block *; input/output).
 * step2   = Spatial step ^ 2 (CUREAL; input).
 * nx      = # of points along x (INT; input).
 * ny      = # of points along y (INT; input).
 * nz      = # of points along z (INT; input).
 *
 */

extern "C" void cgrid_cuda_poissonW(gpu_mem_block *grid, CUREAL step2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_SEG(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  
  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = 2.0 * M_PI / ((CUREAL) nz);
  INT segx = 0, segy = 0; // segx not used

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): poisson wrong subFormat.\n");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_poisson_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], step2, ilx, ily, ilz, nnx1, nny1, nz, segy);
    segx += dsegx1;
    segy += dsegy1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    cgrid_cuda_poisson_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], step2, ilx, ily, ilz, nnx2, nny2, nz, segy);
    segx += dsegx2;
    segy += dsegy2;
  }

  cuda_error_check();
}
