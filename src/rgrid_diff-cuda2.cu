/*
 * CUDA device code (REAL; rgrid) involving differentiation.
 *
 * blockDim = # of threads
 * gridDim = # of blocks
 *
 * nzz: 2 * (nz / 2 + 1) for real space
 *      (nz / 2 + 1) for reciprocal space
 *
 * x, y, z: split along x for GPUs in real space   (uses nnx1, nnx2)
 *          split along y for GPUs in reciprocal space (uses nny1, nny2)
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include "cuda-math.h"
#include "rgrid_bc-cuda.h"
#include "cuda-vars.h"
#include "cuda.h"

extern void *grid_gpu_mem;
extern cudaXtDesc *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * B = FD_X(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_x_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_X(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_x must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_x_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = FD_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_y_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Y(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_y must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_y_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = FD_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_gradient_z_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta * (rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz));
}

/*
 * B = FD_Z(A)
 *
 * src      = Source for operation (gpu_mem_block *; input).
 * dst      = Destination for operation (gpu_mem_block *; input).
 * inv_delta= 1 / (2 * step) (REAL; input).
 * bc       = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fd_gradient_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_gradient_z must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_gradient_z_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = LAPLACE(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       + rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                       - 6.0 * src[idx]);
}

/*
 * B = LAPLACE(A)
 *
 * src       = Source for operation (gpu_mem_block *; input).
 * dst       = Destination for operation (gpu_mem_block *; input).
 * inv_delta2= 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * Returns the value of integral.
 *
 */

extern "C" void rgrid_cuda_fd_laplaceW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = LAPLACE_X(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_x_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_x_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_x_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_X(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (REAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_xW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_x must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_x_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = LAPLACE_Y(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_y_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_y_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_y_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
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
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_yW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_y must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_y_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * B = LAPLACE_Z(A).
 *
 */

__global__ void rgrid_cuda_fd_laplace_z_gpu(CUREAL *src, CUREAL *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz, INT nzz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  dst[idx] = inv_delta2 * (rgrid_cuda_bc_z_plus(src, bc, i, j, k, nx, ny, nz, nzz) + rgrid_cuda_bc_z_minus(src, bc, i, j, k, nx, ny, nz, nzz)
                         - 2.0 * src[idx]);
}

/*
 * B = LAPLACE_Z(A)
 *
 * src        = Source for operation (gpu_mem_block *; input).
 * dst        = Destination for operation (gpu_mem_block *; output).
 * inv_delta2 = 1 / (2 * step) (CUREAL; input).
 * bc         = Boundary condition (char; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 * Returns laplace in dst.
 *
 */

extern "C" void rgrid_cuda_fd_laplace_zW(gpu_mem_block *src, gpu_mem_block *dst, CUREAL inv_delta2, char bc, INT nx, INT ny, INT nz) {

  dst->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *DST = dst->gpu_info->descriptor, *SRC = src->gpu_info->descriptor;

  if(src->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): fd_laplace_z must be in real space (INPLACE).");
    abort();
  }

  if(DST->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");
    abort();
  }

  cudaSetDevice(DST->GPUs[0]);
  rgrid_cuda_fd_laplace_z_gpu<<<blocks,threads>>>((CUREAL *) SRC->data[0], (CUREAL *) DST->data[0], inv_delta2, bc, nx, ny, nz, nzz);

  cuda_error_check();
}

/*
 * FFT gradient (x).
 *
 */

__global__ void rgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *gradient, REAL kx0, REAL step, REAL norm, REAL lx, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kx * norm);
}

/*
 * Gradient of grid in Fourier space (X).
 *
 * gradient_x = Source & destination for operation (gpu_mem_block *; input/output).
 * kx0        = Baseline momentum (grid->kx0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_xW(gpu_mem_block *gradient_x, REAL kx0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_x);
  cudaXtDesc *GRADIENT_X = gradient_x->gpu_info->descriptor;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step);

  if(gradient_x->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_x must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], kx0, step, norm, lx, nx, nny1, nzz, nx / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], kx0, step, norm, lx, nx, nny2, nzz, nx / 2);
  }

  cuda_error_check();
}

/*
 * FFT gradient (y).
 *
 */

__global__ void rgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *gradient, REAL ky0, REAL step, REAL norm, REAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  REAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, ky * norm);
}

/*
 * Gradient of grid in Fourier space (Y).
 *
 * gradient_y = Source & destination for operation (gpu_mem_block *; input/output).
 * ky0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_yW(gpu_mem_block *gradient_y, REAL ky0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_y);
  cudaXtDesc *GRADIENT_Y = gradient_y->gpu_info->descriptor;
  INT seg = 0;
  REAL ly = 2.0 * M_PI / (((REAL) ny) * step);

  if(gradient_y->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_y must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_Y->GPUs[i]);
    rgrid_cuda_fft_gradient_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], ky0, step, norm, ly, nx, nny1, nzz, ny, ny / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Y->GPUs[i]);
    rgrid_cuda_fft_gradient_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], ky0, step, norm, ly, nx, nny2, nzz, ny, ny / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT gradient (z).
 *
 */

__global__ void rgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *gradient, REAL kz0, REAL step, REAL norm, REAL lz, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kz * norm);
}

/*
 * Gradient of grid in Fourier space (Z).
 *
 * gradient_z = Source & destination for operation (gpu_mem_block *; input/output).
 * kz0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_zW(gpu_mem_block *gradient_z, REAL kz0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_z);
  cudaXtDesc *GRADIENT_Z = gradient_z->gpu_info->descriptor;
  REAL lz = M_PI / (((REAL) nzz - 1) * step);

  if(gradient_z->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_z must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], kz0, step, norm, lz, nx, nny1, nzz, nzz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], kz0, step, norm, lz, nx, nny2, nzz, nzz / 2);
  }

  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void rgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  b[idx] = b[idx] * (-(kx * kx + ky * ky + kz * kz) * norm);
}

/*
 * FFT laplace
 *
 * laplace  = Source/destination grid for operation (gpu_mem_block *; input/output).
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Only periodic boundaries!
 *
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_fft_laplaceW(gpu_mem_block *laplace, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace);
  cudaXtDesc *LAPLACE = laplace->gpu_info->descriptor;
  INT seg = 0;

  if(laplace->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE->data[i], norm, kx0, ky0, kz0, 
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE->data[i], norm, kx0, ky0, kz0,
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (x).
 *
 */

__global__ void rgrid_cuda_fft_laplace_x_gpu(CUCOMPLEX *laplace, REAL kx0, REAL step, REAL norm, REAL lx, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  laplace[idx] = laplace[idx] * CUMAKE(-kx * kx * norm, 0.0);
}

/*
 * Laplace of grid in Fourier space (X).
 *
 * laplace_x  = Source & destination for operation (gpu_mem_block *; input/output).
 * kx0        = Baseline momentum (grid->kx0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_xW(gpu_mem_block *laplace_x, REAL kx0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace_x);
  cudaXtDesc *LAPLACE_X = laplace_x->gpu_info->descriptor;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step);

  if(laplace_x->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_x must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE_X->GPUs[i]);
    rgrid_cuda_fft_laplace_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_X->data[i], kx0, step, norm, lx, nx, nny1, nzz, nx / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_X->GPUs[i]);
    rgrid_cuda_fft_laplace_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_X->data[i], kx0, step, norm, lx, nx, nny2, nzz, nx / 2);
  }

  cuda_error_check();
}

/*
 * FFT laplace (y).
 *
 */

__global__ void rgrid_cuda_fft_laplace_y_gpu(CUCOMPLEX *laplace, REAL ky0, REAL step, REAL norm, REAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  REAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  laplace[idx] = laplace[idx] * CUMAKE(-ky * ky * norm, 0.0);
}

/*
 * Laplace of grid in Fourier space (Y).
 *
 * laplace_y  = Source & destination for operation (gpu_mem_block *; input/output).
 * ky0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_yW(gpu_mem_block *laplace_y, REAL ky0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace_y);
  cudaXtDesc *LAPLACE_Y = laplace_y->gpu_info->descriptor;
  INT seg = 0;
  REAL ly = 2.0 * M_PI / (((REAL) ny) * step);

  if(laplace_y->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_y must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE_Y->GPUs[i]);
    rgrid_cuda_fft_laplace_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_Y->data[i], ky0, step, norm, ly, nx, nny1, nzz, ny, ny / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_Y->GPUs[i]);
    rgrid_cuda_fft_laplace_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_Y->data[i], ky0, step, norm, ly, nx, nny2, nzz, ny, ny / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (z).
 *
 */

__global__ void rgrid_cuda_fft_laplace_z_gpu(CUCOMPLEX *laplace, REAL kz0, REAL step, REAL norm, REAL lz, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;
  laplace[idx] = laplace[idx] * CUMAKE(-kz * kz * norm, 0.0);
}

/*
 * Laplace of grid in Fourier space (Z).
 *
 * laplace_z  = Source & destination for operation (gpu_mem_block *; input/output).
 * kz0        = Baseline momentum (grid->ky0; REAL; input).
 * step       = Step size (REAL; input).
 * norm       = FFT norm (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_zW(gpu_mem_block *laplace_z, REAL kz0, REAL step, REAL norm, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace_z);
  cudaXtDesc *LAPLACE_Z = laplace_z->gpu_info->descriptor;
  REAL lz = M_PI / (((REAL) nzz - 1) * step);

  if(laplace_z->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_z must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE_Z->GPUs[i]);
    rgrid_cuda_fft_laplace_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_Z->data[i], kz0, step, norm, lz, nx, nny1, nzz, nzz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_Z->GPUs[i]);
    rgrid_cuda_fft_laplace_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_Z->data[i], kz0, step, norm, lz, nx, nny2, nzz, nzz / 2);
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
 * Normalization done in rgrid-cuda.c
 *
 */

__global__ void rgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUREAL *blocks, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

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

  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  idx2 = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  els2[idx2] -= (kx * kx + ky * ky + kz * kz) * (CUCREAL(b[idx]) * CUCREAL(b[idx]) + CUCIMAG(b[idx]) * CUCIMAG(b[idx]));
  __syncthreads();

  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for(t = 0; t < d; t++) {
      idx2 = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      blocks[idx2] += els2[t];  // reduce threads
    }
  }
}

/*
 * FFT laplace expectation value
 *
 * laplace  = Source/destination grid for operation (gpu_mem_block *; input/output).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 * sum      = Expectation value (REAL; output).
 *
 * Only periodic boundaries!  In Fourier space.
 *
 * Normalization done in rgrid-cuda.c
 *
 */

extern __global__ void rgrid_cuda_block_init(CUREAL *, INT);
extern __global__ void rgrid_cuda_block_reduce(CUREAL *, INT);

extern "C" void rgrid_cuda_fft_laplace_expectation_valueW(gpu_mem_block *laplace, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz, CUREAL *value) {

  SETUP_VARIABLES_RECIPROCAL(laplace);
  cudaXtDesc *LAPLACE = laplace->gpu_info->descriptor;
  CUREAL tmp;
  INT seg = 0, s = CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK * CUDA_THREADS_PER_BLOCK, b31 = blocks1.x * blocks1.y * blocks1.z, b32 = blocks2.x * blocks2.y * blocks2.z;
  extern int cuda_get_element(void *, int, size_t, size_t, void *);

  if(laplace->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_expectation_value must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks1,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) LAPLACE->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                               kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
                               step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) LAPLACE->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                               kx0, ky0, kz0, 2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
                               step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    seg += nny2;
  }

  // Reduce over GPUs
  *value = 0.0;
  for(i = 0; i < ngpu2; i++) {
    cuda_get_element(grid_gpu_mem, i, 0, sizeof(CUREAL), &tmp);
    *value += tmp;
  }

  cuda_error_check();
}


/*
 * FFT div.
 *
 * dst = div (fx, fy, fz)
 *
 */

__global__ void rgrid_cuda_fft_div_gpu(CUCOMPLEX *dst, CUCOMPLEX *fx, CUCOMPLEX *fy, CUCOMPLEX *fz, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx - kx0;
  else
    kx = -((REAL) (nx - i)) * lx - kx0;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly - ky0;
  else
    ky = -((REAL) (nyy - jj)) * ly - ky0;
  if(k < nz2) 
    kz = ((REAL) k) * lz - kz0;
  else
    kz = -((REAL) (nz - k)) * lz - kz0;        

  dst[idx] = CUMAKE(0.0, norm) * (kx * fx[idx] + ky * fy[idx] + kz * fz[idx]);
}

/*
 * FFT div
 *
 * div      = Destination grid for operation (gpu_mem_block *; output).
 * fx       = Vector field X component (gpu_mem_block *; input).
 * fy       = Vector field Y component (gpu_mem_block *; input).
 * fz       = Vector field Z component (gpu_mem_block *; input).
 * norm     = FFT norm (grid->fft_norm) (REAL; input).
 * kx0      = Momentum shift of origin along X (REAL; input).
 * ky0      = Momentum shift of origin along Y (REAL; input).
 * kz0      = Momentum shift of origin along Z (REAL; input).
 * step     = Spatial step length (REAL; input).
 * nx       = # of points along x (INT; input).
 * ny       = # of points along y (INT; input).
 * nz       = # of points along z (INT; input).
 *
 * Only periodic boundaries!
 *
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_fft_divW(gpu_mem_block *div, gpu_mem_block *fx, gpu_mem_block *fy, gpu_mem_block *fz, CUREAL norm, CUREAL kx0, CUREAL ky0, CUREAL kz0, CUREAL step, INT nx, INT ny, INT nz) {

  div->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  SETUP_VARIABLES_RECIPROCAL(div);
  cudaXtDesc *DIV = div->gpu_info->descriptor, *FX = fx->gpu_info->descriptor, *FY = fy->gpu_info->descriptor, *FZ = fz->gpu_info->descriptor;
  INT seg = 0;

  if(fx->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED || fy->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED || fz->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_div must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(DIV->GPUs[i]);
    rgrid_cuda_fft_div_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DIV->data[i], (CUCOMPLEX *) FX->data[i], (CUCOMPLEX *) FY->data[i], (CUCOMPLEX *) FZ->data[i], norm, kx0, ky0, kz0, 
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DIV->GPUs[i]);
    rgrid_cuda_fft_div_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DIV->data[i], (CUCOMPLEX *) FX->data[i], (CUCOMPLEX *) FY->data[i], (CUCOMPLEX *) FZ->data[i], norm, kx0, ky0, kz0,
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * |rot|
 *
 */

__global__ void rgrid_cuda_abs_rot_gpu(CUREAL *rot, CUREAL *fx, CUREAL *fy, CUREAL *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz, INT nzz) { 
 
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUREAL tmp;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nzz + k;

  /* x: (d/dy) fz - (d/dz) fy */
  tmp = inv_delta * (rgrid_cuda_bc_y_plus(fz, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(fz, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_z_plus(fy, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(fy, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = tmp * tmp;

  /* y: (d/dz) fx - (d/dx) fz */
  tmp = inv_delta * (rgrid_cuda_bc_z_plus(fx, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_z_minus(fx, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_x_plus(fz, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(fz, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = rot[idx] + tmp * tmp;

  /* z: (d/dx) fy - (d/dy) fx */
  tmp = inv_delta * (rgrid_cuda_bc_x_plus(fy, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_x_minus(fy, bc, i, j, k, nx, ny, nz, nzz)
                     - rgrid_cuda_bc_y_plus(fx, bc, i, j, k, nx, ny, nz, nzz) - rgrid_cuda_bc_y_minus(fx, bc, i, j, k, nx, ny, nz, nzz));
  rot[idx] = rot[idx] + tmp * tmp;
  rot[idx] = SQRT(rot[idx]);
}

/*
 * |rot|
 *
 * rot       = Grid to be operated on (gpu_mem_block *; input/output).
 * fx        = x component of the field (gpu_mem_block *; input).
 * fy        = y component of the field (gpu_mem_block *; input).
 * fz        = z component of the field (gpu_mem_block *; input).
 * inv_delta = 1 / (2 * step) (CUREAL; input).
 * bc        = Boundary condition (char; input).
 * nx        = # of points along x (INT; input).
 * ny        = # of points along y (INT; input).
 * nz        = # of points along z (INT; input).
 *
 * TODO: For this it probably makes sense to force transferring the blocks to host memory and do the operation there.
 *
 */

extern "C" void rgrid_cuda_abs_rotW(gpu_mem_block *rot, gpu_mem_block *fx, gpu_mem_block *fy, gpu_mem_block *fz, CUREAL inv_delta, char bc, INT nx, INT ny, INT nz) {

  rot->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  INT nzz = 2 * (nz / 2 + 1);
  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
  cudaXtDesc *ROT = rot->gpu_info->descriptor, *FX = fx->gpu_info->descriptor, *FY = fy->gpu_info->descriptor, *FZ = fz->gpu_info->descriptor;

  if(fx->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || fy->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE || fz->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): abs_rot must be in real space (INPLACE).");
    abort();
  }

  cudaSetDevice(ROT->GPUs[0]);
  rgrid_cuda_abs_rot_gpu<<<blocks,threads>>>((CUREAL *) ROT->data[0], (CUREAL *) FX->data[0], (CUREAL *) FY->data[0], (CUREAL *) FZ->data[0], 
                                             inv_delta, bc, nx, ny, nz, nzz);

  cuda_error_check();
}
