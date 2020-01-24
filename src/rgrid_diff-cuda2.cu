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

__global__ void rgrid_cuda_fft_gradient_x_gpu(CUCOMPLEX *gradient, REAL step, REAL lx, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(i < nx2) 
    kx = ((REAL) i) * lx;
  else
    kx = -((REAL) (nx - i)) * lx;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kx);
}

/*
 * Gradient of grid in Fourier space (X).
 *
 * gradient_x = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_xW(gpu_mem_block *gradient_x, REAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_x);
  cudaXtDesc *GRADIENT_X = gradient_x->gpu_info->descriptor;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step);

  if(gradient_x->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_x must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], step, lx, nx, nny1, nzz, nx / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_X->GPUs[i]);
    rgrid_cuda_fft_gradient_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_X->data[i], step, lx, nx, nny2, nzz, nx / 2);
  }

  cuda_error_check();
}

/*
 * FFT gradient (y).
 *
 */

__global__ void rgrid_cuda_fft_gradient_y_gpu(CUCOMPLEX *gradient, REAL step, REAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  REAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly;
  else
    ky = -((REAL) (nyy - jj)) * ly;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, ky);
}

/*
 * Gradient of grid in Fourier space (Y).
 *
 * gradient_y = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_yW(gpu_mem_block *gradient_y, REAL step, INT nx, INT ny, INT nz) {

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
    rgrid_cuda_fft_gradient_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], step, ly, nx, nny1, nzz, ny, ny / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Y->GPUs[i]);
    rgrid_cuda_fft_gradient_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Y->data[i], step, ly, nx, nny2, nzz, ny, ny / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT gradient (z).
 *
 */

__global__ void rgrid_cuda_fft_gradient_z_gpu(CUCOMPLEX *gradient, REAL step, REAL lz, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(k < nz2) 
    kz = ((REAL) k) * lz;
  else
    kz = -((REAL) (nz - k)) * lz;
  gradient[idx] = gradient[idx] * CUMAKE(0.0, kz);
}

/*
 * Gradient of grid in Fourier space (Z).
 *
 * gradient_z = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_gradient_zW(gpu_mem_block *gradient_z, REAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(gradient_z);
  cudaXtDesc *GRADIENT_Z = gradient_z->gpu_info->descriptor;
  REAL lz = M_PI / (((REAL) nzz - 1) * step);

  if(gradient_z->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_gradient_z must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], step, lz, nx, nny1, nzz, nzz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRADIENT_Z->GPUs[i]);
    rgrid_cuda_fft_gradient_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRADIENT_Z->data[i], step, lz, nx, nny2, nzz, nzz / 2);
  }

  cuda_error_check();
}

/*
 * FFT laplace.
 *
 * B = B'' in Fourier space.
 *
 */

__global__ void rgrid_cuda_fft_laplace_gpu(CUCOMPLEX *b, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx;
  else
    kx = -((REAL) (nx - i)) * lx;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly;
  else
    ky = -((REAL) (nyy - jj)) * ly;
  if(k < nz2) 
    kz = ((REAL) k) * lz;
  else
    kz = -((REAL) (nz - k)) * lz;

  b[idx] = -b[idx] * (kx * kx + ky * ky + kz * kz);
}

/*
 * FFT laplace
 *
 * laplace  = Source/destination grid for operation (gpu_mem_block *; input/output).
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

extern "C" void rgrid_cuda_fft_laplaceW(gpu_mem_block *laplace, CUREAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace);
  cudaXtDesc *LAPLACE = laplace->gpu_info->descriptor;
  INT seg = 0;

  if(laplace->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE->data[i], 
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_fft_laplace_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE->data[i],
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny2, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (x).
 *
 */

__global__ void rgrid_cuda_fft_laplace_x_gpu(CUCOMPLEX *laplace, REAL step, REAL lx, INT nx, INT ny, INT nz, INT nx2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kx;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(i < nx2) 
    kx = ((REAL) i) * lx;
  else
    kx = -((REAL) (nx - i)) * lx;
  laplace[idx] = laplace[idx] * CUMAKE(-kx * kx, 0.0);
}

/*
 * Laplace of grid in Fourier space (X).
 *
 * laplace_x  = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_xW(gpu_mem_block *laplace_x, REAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace_x);
  cudaXtDesc *LAPLACE_X = laplace_x->gpu_info->descriptor;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step);

  if(laplace_x->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_x must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE_X->GPUs[i]);
    rgrid_cuda_fft_laplace_x_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_X->data[i], step, lx, nx, nny1, nzz, nx / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_X->GPUs[i]);
    rgrid_cuda_fft_laplace_x_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_X->data[i], step, lx, nx, nny2, nzz, nx / 2);
  }

  cuda_error_check();
}

/*
 * FFT laplace (y).
 *
 */

__global__ void rgrid_cuda_fft_laplace_y_gpu(CUCOMPLEX *laplace, REAL step, REAL ly, INT nx, INT ny, INT nz, INT nyy, INT ny2, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, jj = j + seg, idx;
  REAL ky;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly;
  else
    ky = -((REAL) (nyy - jj)) * ly;
  laplace[idx] = laplace[idx] * CUMAKE(-ky * ky, 0.0);
}

/*
 * Laplace of grid in Fourier space (Y).
 *
 * laplace_y  = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_yW(gpu_mem_block *laplace_y, REAL step, INT nx, INT ny, INT nz) {

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
    rgrid_cuda_fft_laplace_y_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_Y->data[i], step, ly, nx, nny1, nzz, ny, ny / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_Y->GPUs[i]);
    rgrid_cuda_fft_laplace_y_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_Y->data[i], step, ly, nx, nny2, nzz, ny, ny / 2, seg);
    seg += nny2;
  }

  cuda_error_check();
}

/*
 * FFT laplace (z).
 *
 */

__global__ void rgrid_cuda_fft_laplace_z_gpu(CUCOMPLEX *laplace, REAL step, REAL lz, INT nx, INT ny, INT nz, INT nz2) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  REAL kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(k < nz2) 
    kz = ((REAL) k) * lz;
  else
    kz = -((REAL) (nz - k)) * lz;
  laplace[idx] = laplace[idx] * CUMAKE(-kz * kz, 0.0);
}

/*
 * Laplace of grid in Fourier space (Z).
 *
 * laplace_z  = Source & destination for operation (gpu_mem_block *; input/output).
 * step       = Step size (REAL; input).
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 *
 */

extern "C" void rgrid_cuda_fft_laplace_zW(gpu_mem_block *laplace_z, REAL step, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(laplace_z);
  cudaXtDesc *LAPLACE_Z = laplace_z->gpu_info->descriptor;
  REAL lz = M_PI / (((REAL) nzz - 1) * step);

  if(laplace_z->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): fft_laplace_z must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(LAPLACE_Z->GPUs[i]);
    rgrid_cuda_fft_laplace_z_gpu<<<blocks1,threads>>>((CUCOMPLEX *) LAPLACE_Z->data[i], step, lz, nx, nny1, nzz, nzz / 2);
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE_Z->GPUs[i]);
    rgrid_cuda_fft_laplace_z_gpu<<<blocks2,threads>>>((CUCOMPLEX *) LAPLACE_Z->data[i], step, lz, nx, nny2, nzz, nzz / 2);
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

__global__ void rgrid_cuda_fft_laplace_expectation_value_gpu(CUCOMPLEX *b, CUREAL *blocks, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

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
    kx = ((REAL) i) * lx;
  else
    kx = -((REAL) (nx - i)) * lx;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly;
  else
    ky = -((REAL) (nyy - jj)) * ly;
  if(k < nz2) 
    kz = ((REAL) k) * lz;
  else
    kz = -((REAL) (nz - k)) * lz;

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

extern "C" void rgrid_cuda_fft_laplace_expectation_valueW(gpu_mem_block *laplace, CUREAL step, INT nx, INT ny, INT nz, CUREAL *value) {

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
                               2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
                               step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    rgrid_cuda_block_reduce<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b31);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(LAPLACE->GPUs[i]);
    rgrid_cuda_block_init<<<1,1>>>((CUREAL *) grid_gpu_mem_addr->data[i], b32);
    // Blocks, Threads, dynamic memory size
    rgrid_cuda_fft_laplace_expectation_value_gpu<<<blocks2,threads,s*sizeof(CUREAL)>>>((CUCOMPLEX *) LAPLACE->data[i], (CUREAL *) grid_gpu_mem_addr->data[i], 
                               2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step),
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

__global__ void rgrid_cuda_fft_div_gpu(CUCOMPLEX *dst, CUCOMPLEX *fx, CUCOMPLEX *fy, CUCOMPLEX *fz, CUREAL lx, CUREAL ly, CUREAL lz, CUREAL step, INT nx, INT ny, INT nz, INT nyy, INT nx2, INT ny2, INT nz2, INT seg) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  
  if(i < nx2) 
    kx = ((REAL) i) * lx;
  else
    kx = -((REAL) (nx - i)) * lx;
  if(jj < ny2) 
    ky = ((REAL) jj) * ly;
  else
    ky = -((REAL) (nyy - jj)) * ly;
  if(k < nz2) 
    kz = ((REAL) k) * lz;
  else
    kz = -((REAL) (nz - k)) * lz;

  dst[idx] = CUMAKE(0.0, 1.0) * (kx * fx[idx] + ky * fy[idx] + kz * fz[idx]);
}

/*
 * FFT div
 *
 * div      = Destination grid for operation (gpu_mem_block *; output).
 * fx       = Vector field X component (gpu_mem_block *; input).
 * fy       = Vector field Y component (gpu_mem_block *; input).
 * fz       = Vector field Z component (gpu_mem_block *; input).
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

extern "C" void rgrid_cuda_fft_divW(gpu_mem_block *div, gpu_mem_block *fx, gpu_mem_block *fy, gpu_mem_block *fz, CUREAL step, INT nx, INT ny, INT nz) {

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
    rgrid_cuda_fft_div_gpu<<<blocks1,threads>>>((CUCOMPLEX *) DIV->data[i], (CUCOMPLEX *) FX->data[i], (CUCOMPLEX *) FY->data[i], (CUCOMPLEX *) FZ->data[i],
        2.0 * M_PI / (((REAL) nx) * step), 2.0 * M_PI / (((REAL) ny) * step), M_PI / (((REAL) nzz - 1) * step), step, nx, nny1, nzz, ny, nx / 2, ny / 2, nzz / 2, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(DIV->GPUs[i]);
    rgrid_cuda_fft_div_gpu<<<blocks2,threads>>>((CUCOMPLEX *) DIV->data[i], (CUCOMPLEX *) FX->data[i], (CUCOMPLEX *) FY->data[i], (CUCOMPLEX *) FZ->data[i],
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


/*
 * Poisson equation.
 *
 */

__global__ void rgrid_cuda_poisson_gpu(CUCOMPLEX *grid, CUREAL step2, CUREAL ilx, CUREAL ily, CUREAL ilz, INT nx, INT ny, INT nzz, INT seg) {
  
  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx, jj = j + seg;
  CUREAL kx, ky, kz;

  if(i >= nx || j >= ny || k >= nzz) return;

  idx = (i * ny + j) * nzz + k;
  kx = COS(ilx * (CUREAL) i);
  ky = COS(ily * (CUREAL) jj);
  kz = COS(ilz * (CUREAL) k);
  if(i || jj || k)
    grid[idx] = grid[idx] * step2 / (2.0 * (kx + ky + kz - 3.0));
  else
    grid[idx] = CUMAKE(0.0,0.0);
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
 * In Fourier space.
 *
 */

extern "C" void rgrid_cuda_poissonW(gpu_mem_block *grid, CUREAL step2, INT nx, INT ny, INT nz) {

  SETUP_VARIABLES_RECIPROCAL(grid);
  cudaXtDesc *GRID = grid->gpu_info->descriptor;
  INT seg = 0;
  CUREAL ilx = 2.0 * M_PI / ((CUREAL) nx), ily = 2.0 * M_PI / ((CUREAL) ny), ilz = M_PI / ((CUREAL) nzz);

  if(grid->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
    fprintf(stderr, "libgrid(cuda): poisson must be in Fourier space (INPLACE_SHUFFLED).");
    abort();
  }

  for(i = 0; i < ngpu1; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_poisson_gpu<<<blocks1,threads>>>((CUCOMPLEX *) GRID->data[i], step2, ilx, ily, ilz, nx, nny1, nzz, seg);
    seg += nny1;
  }

  for(i = ngpu1; i < ngpu2; i++) {
    cudaSetDevice(GRID->GPUs[i]);
    rgrid_cuda_poisson_gpu<<<blocks2,threads>>>((CUCOMPLEX *) GRID->data[i], step2, ilx, ily, ilz, nx, nny2, nzz, seg);
    seg += nny2;
  }

  cuda_error_check();
}
