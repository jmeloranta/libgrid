/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
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

extern void *grid_gpu_mem; // Defined in cgrid.c

/* 
 * Differentiate a grid with respect to x (central difference).
 *
 * src       = source for gradient (rgrid *; input).
 * dst       = destination for gradient (rgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_gradient_x(rgrid *src, rgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_gradient_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, src->nx, src->ny, src->nz);

  return 0;
}

/* 
 * Differentiate a grid with respect to y (central difference).
 *
 * src       = source for gradient (rgrid *; input).
 * dst       = destination for gradient (rgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_gradient_y(rgrid *src, rgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_gradient_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, src->nx, src->ny, src->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to z (central difference).
 * 
 * src       = source for gradient (rgrid *; input).
 * dst       = destination for gradient (rgrid *; output).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc        = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_gradient_z(rgrid *src, rgrid *dst, REAL inv_delta, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_gradient_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, src->nx, src->ny, src->nz);
  return 0;
}

/* 
 * Laplace of a grid (central difference).
 *
 * src        = source for gradient (rgrid *; input).
 * dst        = destination for gradient (rgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_laplace(rgrid *src, rgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_laplaceW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (x component). This is the second derivative with respect to x.
 *
 * src        = source for gradient (rgrid *; input).
 * dst        = destination for gradient (rgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_laplace_x(rgrid *src, rgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_laplace_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (y component). This is the second derivative with respect to y.
 *
 * src        = source for gradient (rgrid *; input).
 * dst        = destination for gradient (rgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_laplace_y(rgrid *src, rgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_laplace_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (z component). This is the second derivative with respect to z.
 *
 * src        = source for gradient (rgrid *; input).
 * dst        = destination for gradient (rgrid *; output).
 * inv_delta2 = 1 / (step * step) (REAL; input).
 * bc         = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_laplace_z(rgrid *src, rgrid *dst, REAL inv_delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;

  rgrid_cuda_fd_laplace_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, src->nx, src->ny, src->nz);
  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along x.
 *
 * gradient_x = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_gradient_x(rgrid *gradient_x) {

  if(gradient_x->host_lock) {
    cuda_remove_block(gradient_x->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(gradient_x->value, gradient_x->grid_len, gradient_x->cufft_handle_c2r, gradient_x->id, 1) < 0) return -1;

  rgrid_cuda_fft_gradient_xW(cuda_block_address(gradient_x->value), gradient_x->kx0, gradient_x->step, gradient_x->fft_norm, gradient_x->nx, gradient_x->ny, gradient_x->nz);

  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along y.
 *
 * gradient_y = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_gradient_y(rgrid *gradient_y) {

  if(gradient_y->host_lock) {
    cuda_remove_block(gradient_y->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(gradient_y->value, gradient_y->grid_len, gradient_y->cufft_handle_c2r, gradient_y->id, 1) < 0) return -1;

  rgrid_cuda_fft_gradient_yW(cuda_block_address(gradient_y->value), gradient_y->ky0, gradient_y->step, gradient_y->fft_norm, gradient_y->nx, gradient_y->ny, gradient_y->nz);

  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along z.
 *
 * gradient_z = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_gradient_z(rgrid *gradient_z) {

  if(gradient_z->host_lock) {
    cuda_remove_block(gradient_z->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(gradient_z->value, gradient_z->grid_len, gradient_z->cufft_handle_c2r, gradient_z->id, 1) < 0) return -1;

  rgrid_cuda_fft_gradient_zW(cuda_block_address(gradient_z->value), gradient_z->kz0, gradient_z->step, gradient_z->fft_norm, gradient_z->nx, gradient_z->ny, gradient_z->nz);

  return 0;
}

/*
 * Calculate laplacian of a grid (in Fourier space).
 *
 * laplace = grid for the operation (rgrid *; output).
 *
 */

EXPORT char rgrid_cuda_fft_laplace(rgrid *laplace) {

  if(laplace->host_lock) {
    cuda_remove_block(laplace->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace->value, laplace->grid_len, laplace->cufft_handle_c2r, laplace->id, 1) < 0) return -1;

  rgrid_cuda_fft_laplaceW(cuda_block_address(laplace->value), laplace->fft_norm, laplace->kx0, laplace->ky0, laplace->kz0, laplace->step, laplace->nx, laplace->ny, laplace->nz);

  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along x twice.
 *
 * laplace_x = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_laplace_x(rgrid *laplace_x) {

  if(laplace_x->host_lock) {
    cuda_remove_block(laplace_x->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace_x->value, laplace_x->grid_len, laplace_x->cufft_handle_c2r, laplace_x->id, 1) < 0) return -1;

  rgrid_cuda_fft_laplace_xW(cuda_block_address(laplace_x->value), laplace_x->kx0, laplace_x->step, laplace_x->fft_norm, laplace_x->nx, laplace_x->ny, laplace_x->nz);

  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along y twice.
 *
 * laplace_y = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_laplace_y(rgrid *laplace_y) {

  if(laplace_y->host_lock) {
    cuda_remove_block(laplace_y->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace_y->value, laplace_y->grid_len, laplace_y->cufft_handle_c2r, laplace_y->id, 1) < 0) return -1;

  rgrid_cuda_fft_laplace_yW(cuda_block_address(laplace_y->value), laplace_y->ky0, laplace_y->step, laplace_y->fft_norm, laplace_y->nx, laplace_y->ny, laplace_y->nz);

  return 0;
}

/* 
 * Differentiate real grid in the Fourier space along z twice.
 *
 * laplace_z = grid for differentiation (rgrid *; input/output).
 *
 */

EXPORT char rgrid_cuda_fft_laplace_z(rgrid *laplace_z) {

  if(laplace_z->host_lock) {
    cuda_remove_block(laplace_z->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace_z->value, laplace_z->grid_len, laplace_z->cufft_handle_c2r, laplace_z->id, 1) < 0) return -1;

  rgrid_cuda_fft_laplace_zW(cuda_block_address(laplace_z->value), laplace_z->kz0, laplace_z->step, laplace_z->fft_norm, laplace_z->nx, laplace_z->ny, laplace_z->nz);

  return 0;
}

/*
 * Calculate expectation value of laplace operator in the Fourier space (int grid^* grid'').
 *
 * laplace = laplace of grid (cgrid *; output).
 * value   = expectation value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_fft_laplace_expectation_value(rgrid *laplace, REAL *value) {

  REAL step = laplace->step, norm = laplace->fft_norm;

  if(laplace->host_lock) {
    cuda_remove_block(laplace->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(laplace->value, laplace->grid_len, laplace->cufft_handle_c2r, laplace->id, 1) < 0) return -1;

  rgrid_cuda_fft_laplace_expectation_valueW(cuda_block_address(laplace->value), laplace->kx0, laplace->ky0, laplace->kz0, laplace->step, laplace->nx, laplace->ny, laplace->nz, (CUREAL *) value);

  cuda_get_element(grid_gpu_mem, 0, 0, sizeof(REAL), value);

  if(laplace->nx != 1) *value *= step;
  if(laplace->ny != 1) *value *= step;
  if(laplace->nz != 1) *value *= step;
  *value *= norm;

  return 0;
}

/*
 * Calculate div of a vector field (in Fourier space).
 *
 * div = result (rgrid *; output).
 * fx  = x component of the field (rgrid *; input). In Fourier space.
 * fy  = y component of the field (rgrid *; input). In Fourier space.
 * fz  = z component of the field (rgrid *; input). In Fourier space.
 *
 */

EXPORT char rgrid_cuda_fft_div(rgrid *div, rgrid *fx, rgrid *fy, rgrid *fz) {

  if(div->host_lock || fx->host_lock || fy->host_lock || fz->host_lock) {
    cuda_remove_block(fx->value, 1);
    cuda_remove_block(fy->value, 1);
    cuda_remove_block(fz->value, 1);
    cuda_remove_block(div->value, 0);
    return -1;
  }

  if(cuda_four_block_policy(div->value, div->grid_len, div->cufft_handle_c2r, div->id, 1, fx->value, fx->grid_len, fx->cufft_handle_c2r, fx->id, 1,
                            fy->value, fy->grid_len, fy->cufft_handle_c2r, fy->id, 1, fz->value, fz->grid_len, fz->cufft_handle_c2r, fz->id, 1) < 0) return -1;

  rgrid_cuda_fft_divW(cuda_block_address(div->value), cuda_block_address(fx->value), cuda_block_address(fy->value), cuda_block_address(fz->value), div->fft_norm, 
                      div->kx0, div->ky0, div->kz0, div->step, div->nx, div->ny, div->nz);

  return 0;
}

/*
 * Calculate |rot| (|curl|; |\Nabla\times|) of a vector field (i.e., magnitude).
 *
 * rot       = Absolute value of rot (rgrid *; output).
 * fx        = X component of the vector field (rgrid *; input).
 * fy        = Y component of the vector field (rgrid *; input).
 * fz        = Z component of the vector field (rgrid *; input).
 * inv_delta = 1 / (2 * step) (REAL; input).
 * bc       = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_abs_rot(rgrid *rot, rgrid *fx, rgrid *fy, rgrid *fz, REAL inv_delta, char bc) {

  if(rot->host_lock || fx->host_lock || fy->host_lock || fz->host_lock || cuda_ngpus() > 1) {  // TODO: Does not execute on GPU for multi-GPU situation
    cuda_remove_block(fx->value, 1);
    cuda_remove_block(fy->value, 1);
    cuda_remove_block(fz->value, 1);
    cuda_remove_block(rot->value, 0);
    return -1;
  }

  if(cuda_four_block_policy(rot->value, rot->grid_len, rot->cufft_handle_r2c, rot->id, 0, fx->value, fx->grid_len, fx->cufft_handle_r2c, fx->id, 1,
                            fy->value, fy->grid_len, fy->cufft_handle_r2c, fy->id, 1, fz->value, fz->grid_len, fz->cufft_handle_r2c, fz->id, 1) < 0) return -1;

  rgrid_cuda_abs_rotW(cuda_block_address(rot->value), cuda_block_address(fx->value), cuda_block_address(fy->value), cuda_block_address(fz->value), inv_delta, bc, rot->nx, rot->ny, rot->nz);
  return 0;
}
