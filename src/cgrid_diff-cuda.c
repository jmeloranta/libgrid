/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * REAL complex (cgrid) versions involving differentiation.
 *
 * Functions return 0 if operation was successful and -1 if not. The latter case usually means that it was against the 
 * policy function for the operation to run on GPU.
 *
 */

#include "grid.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>


/* 
 * Differentiate a grid with respect to x (central difference).
 *
 * src       = source grid (cgrid *; input).
 * dst       = destination grid (cgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_gradient_x(cgrid *src, cgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_gradient_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to y (central difference).
 *
 * src       = source grid (cgrid *; input).
 * dst       = destination grid (cgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_gradient_y(cgrid *src, cgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to z (central difference).
 *
 * src       = source grid (cgrid *; input).
 * dst       = destination grid (cgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_gradient_z(cgrid *src, cgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_gradient_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/* 
 * Laplace of a grid (central difference).
 *
 * src        = source grid (cgrid *; input).
 * dst        = destination grid (cgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_laplace(cgrid *src, cgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplaceW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (x component). This is the second derivative with respect to x.
 *
 * src        = source grid (cgrid *; input).
 * dst        = destination grid (cgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_laplace_x(cgrid *src, cgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplace_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (y component). This is the second derivative with respect to y.
 *
 * src        = source grid (cgrid *; input).
 * dst        = destination grid (cgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_laplace_y(cgrid *src, cgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplace_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (z component). This is the second derivative with respect to z.
 *
 * src        = source grid (cgrid *; input).
 * dst        = destination grid (cgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_laplace_z(cgrid *src, cgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplace_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);
  return 0;
}

/*
 * Differentiate grid in the Fourier space along x.
 *
 * src    = source grid (cgrid *; input).
 * dst    = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_gradient_x(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  if (dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_gradient_xW(cuda_block_address(dst->value), dst->fft_norm, dst->kx0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Differentiate grid in the Fourier space along y.
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_gradient_y(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_gradient_yW(cuda_block_address(dst->value), dst->fft_norm, dst->ky0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Differentiate grid in the Fourier space along z.
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_gradient_z(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_gradient_zW(cuda_block_address(dst->value), dst->fft_norm, dst->kz0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Calculate second derivative (laplacian) of a grid (in Fourier space).
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_laplaceW(cuda_block_address(dst->value), dst->fft_norm, dst->kx0, dst->ky0, dst->kz0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Calculate second derivative (X) of a grid (in Fourier space).
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace_x(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_laplace_xW(cuda_block_address(dst->value), dst->fft_norm, dst->kx0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Calculate second derivative (Y) of a grid (in Fourier space).
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace_y(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_laplace_yW(cuda_block_address(dst->value), dst->fft_norm, dst->ky0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Calculate second derivative (Z) of a grid (in Fourier space).
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace_z(cgrid *src, cgrid *dst) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  cgrid_cuda_fft_laplace_zW(cuda_block_address(dst->value), dst->fft_norm, dst->kz0, dst->step, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Calculate expectation value of laplace operator in the Fourier space (int grid^* grid'').
 *
 * laplace = laplace of grid (cgrid *; output).
 * value   = expectation value (REAL *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace_expectation_value(cgrid *laplace, REAL *value) {

  REAL step = laplace->step, norm = laplace->fft_norm;
  REAL complex value2;

  if(laplace->host_lock) {
    cuda_remove_block(laplace->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace->value, laplace->grid_len, laplace->cufft_handle, laplace->id, 1) < 0) return -1;

  cgrid_cuda_fft_laplace_expectation_valueW(cuda_block_address(laplace->value), laplace->kx0, laplace->ky0, laplace->kz0, laplace->step, laplace->nx, laplace->ny, laplace->nz, (CUCOMPLEX *) &value2);
  *value = CREAL(value2);

  if(laplace->nx != 1) *value *= step;
  if(laplace->ny != 1) *value *= step;
  if(laplace->nz != 1) *value *= step;
  *value *= norm;

  return 0;
}
