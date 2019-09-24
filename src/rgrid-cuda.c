/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * REAL complex (cgrid) versions.
 *
 * Functions return 0 if operation was successful and -1 if not. The latter case usually means that it was against the 
 * policy function for the operation to run on GPU.
 *
 */

#include "grid.h"

extern void *grid_gpu_mem; // Defined in cgrid.c

EXPORT void rgrid_cuda_init(size_t len) {

  cgrid_cuda_init(len);
}

/*
 * Convolute two grids (in Fourier space).
 *
 * dst  = Destination (rgrid *; output).
 * src1 = Source 1 (rgrid *; input).
 * src2 = Source 2 (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_fft_convolute(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_c2r, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_c2r, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_c2r, dst->id, 0) < 0) return -1;

  rgrid_cuda_fft_convoluteW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->fft_norm2, src1->nx, src1->ny, src1->nz);

  return 0;
}

/* 
 * Rise a grid to given power.
 *
 * dst    = destination grid (rgrid *; output).
 * src    = source grid (rgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 */

EXPORT char rgrid_cuda_power(rgrid *dst, rgrid *src, REAL exponent) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
    return -1;

  rgrid_cuda_powerW(cuda_block_address(dst->value), cuda_block_address(src->value), exponent, src->nx, src->ny, src->nz);

  return 0;
}

/* 
 * Rise a |grid| to given power.
 *
 * dst      = destination grid (rgrid *; output).
 * src      = source grid (rgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 */

EXPORT char rgrid_cuda_abs_power(rgrid *dst, rgrid *src, REAL exponent) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
    return -1;

  rgrid_cuda_abs_powerW(cuda_block_address(dst->value), cuda_block_address(src->value), exponent, src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (rgrid *; input/output).
 * c    = multiplier (REAL; input).
 *
 */

EXPORT char rgrid_cuda_multiply(rgrid *grid, REAL c) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiplyW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Multiply grid by a constant (in fft space).
 *
 * grid = grid to be multiplied (rgrid *; input/output).
 * c    = multiplier (REAL; input).
 *
 */

EXPORT char rgrid_cuda_multiply_fft(rgrid *grid, REAL c) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_c2r, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_fftW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add two grids: dst = src1 + src2
 *
 * dst  = destination grid (rgrid *; output).
 * src1 = 1st of the grids to be added (rgrid *; input).
 * src2 = 2nd of the grids to be added (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_sum(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_sumW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/*
 * Subtract two grids: dst = src1 - src2
 *
 * dst  = destination grid (rgrid *; output).
 * src1 = 1st of the grids to be added (rgrid *; input).
 * src2 = 2nd of the grids to be added (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_difference(rgrid *dst, rgrid *src1, rgrid *src2) {
  
  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_differenceW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/* 
 * Calculate product of two grids: dst = src1 * src2
 *
 * dst   = destination grid (rgrid *; output).
 * src1  = 1st source grid (rgrid *; input).
 * src2  = 2nd source grid (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_product(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/* 
 * Divide two grids: dst = src1 / src2
 *
 * dst  = destination grid (rgrid *; output).
 * src1 = 1st source grid (rgrid *; input).
 * src2 = 2nd source grid (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_division(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_divisionW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/* 
 * "Safely" divide two grids: dst = src1 / (src2 + eps)
 *
 * dst  = destination grid (rgrid *; output).
 * src1 = 1st source grid (rgrid *; input).
 * src2 = 2nd source grid (rgrid *; input).
 * eps  = Epsilon (REAL).
 *
 */

EXPORT char rgrid_cuda_division_eps(rgrid *dst, rgrid *src1, rgrid *src2, REAL eps) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_division_epsW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), eps, src1->nx, src1->ny, src1->nz);

  return 0;
}

/*
 * Add a constant to grid.
 *
 * grid = grid to be operated on (rgrid *; input/output).
 * c    = constant (REAL; input).
 *
 */

EXPORT char rgrid_cuda_add(rgrid *grid, REAL c) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_addW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (rgrid *; input/output).
 * cm   = multiplier (REAL; input).
 * ca   = constant to be added (REAL; input).
 *
 */

EXPORT char rgrid_cuda_multiply_and_add(rgrid *grid, REAL cm, REAL ca) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_and_addW(cuda_block_address(grid->value), cm, ca, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (rgrid *; input/output).
 * ca   = constant to be added (REAL; input).
 * cm   = multiplier (REAL; input).
 *
 */

EXPORT char rgrid_cuda_add_and_multiply(rgrid *grid, REAL ca, REAL cm) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_add_and_multiplyW(cuda_block_address(grid->value), ca, cm, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add scaled grid (multiply/add): dst = dst + d * src
 *
 * dst = destination grid for the operation (rgrid *; input/output).
 * d   = multiplier for the operation (REAL; input).
 * src = source grid for the operation (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_add_scaled(rgrid *dst, REAL d, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 1) < 0)
    return -1;

  rgrid_cuda_add_scaledW(cuda_block_address(dst->value), d, cuda_block_address(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Perform the following operation: dst = dst + d * src1 * src2.
 *
 * dst  = destination grid (rgrid *; input/output).
 * d    = constant multiplier (REAL; input).
 * src1 = 1st source grid (rgrid *; input).
 * src2 = 2nd source grid (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_add_scaled_product(rgrid *dst, REAL d, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, 
                             src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 1) < 0) return -1;

  rgrid_cuda_add_scaled_productW(cuda_block_address(dst->value), d, cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/*
 * Copy two areas in GPU. Source grid is on GPU.
 *
 * dst = Copy of grid (rgrid *; output).
 * src = Grid to be copied (rgrid *; input).
 * 
 */

EXPORT char rgrid_cuda_copy(rgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_copy_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, src->value, src->grid_len, src->cufft_handle_r2c, src->id) < 0) 
    return -1;

  cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  return 0;
}

/*
 * Set grid value to constant.
 *
 * grid = grid to be set to constant (rgrid *; output).
 * c    = constant value (REAL; input).
 *
 */

EXPORT char rgrid_cuda_constant(rgrid *grid, REAL c) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 0) < 0) return -1;

  rgrid_cuda_constantW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Integrate over a grid.
 *
 * grid  = grid for integration (rgrid *; input).
 * value = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_integral(rgrid *grid, REAL *value) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_integralW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  if(grid->nz != 1) *value *= grid->step;

  return 0;
}

/*
 * Integrate over a grid with limits.
 *
 * grid = grid to be integrated (cgrid *; input).
 * xl   = lower limit for x (REAL; input).
 * xu   = upper limit for x (REAL; input).
 * yl   = lower limit for y (REAL; input).
 * yu   = upper limit for y (REAL; input).
 * zl   = lower limit for z (REAL; input).
 * zu   = upper limit for z (REAL; input).
 * value = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_integral_region(rgrid *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, REAL *value) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_integral_regionW(cuda_block_address(grid->value), il, iu, jl, ju, kl, ku, grid->nx, grid->ny, grid->nz, value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  if(grid->nz != 1) *value *= grid->step;

  return 0;
}

/* 
 * Integrate over the grid squared (int grid^2).
 *
 * grid  = grid for integration (rgrid *; input).
 * value = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_integral_of_square(rgrid *grid, REAL *value) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_integral_of_squareW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  if(grid->nz != 1) *value *= grid->step;

  return 0;
}

/*
 * Calculate overlap between two grids (int src1 src2).
 *
 * src1  = first grid for integration (rgrid *; input).
 * src2  = second grid for integration (rgrid *; input).
 * value = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_integral_of_product(rgrid *src1, rgrid *src2, REAL *value) {

  if(src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    return -1;
  }

  if(cuda_two_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1) < 0) 
    return -1;

  rgrid_cuda_integral_of_productW(cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz, value);

  if(src1->nx != 1) *value *= src1->step;
  if(src1->ny != 1) *value *= src1->step;
  if(src1->nz != 1) *value *= src1->step;

  return 0;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int src2 src1 src2 = int src1 src2^2).
 *
 * src1  = first grid for integration (rgrid *; input).
 * src2  = second grid for integration (rgrid *; input).
 * value = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_grid_expectation_value(rgrid *src1, rgrid *src2, REAL *value) {

  if(src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    return -1;
  }

  if(cuda_two_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1) < 0)
    return -1;

  rgrid_cuda_grid_expectation_valueW(cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz, value);

  if(src1->nx != 1) *value *= src1->step;
  if(src1->ny != 1) *value *= src1->step;
  if(src1->nz != 1) *value *= src1->step;

  return 0;
}

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
 * Calculate dot product of the gradient of the grid.
 *
 * src           = source for gradient (rgrid *; input).
 * dst           = destination for gradient (rgrid *; output).
 * inv2_delta2   = 1 / (2.0 * step * 2.0 * step) (REAL; input).
 * bc            = boundary condition (char; input).
 *
 */

EXPORT char rgrid_cuda_fd_gradient_dot_gradient(rgrid *src, rgrid *dst, REAL inv_2delta2, char bc) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
    return -1;

  rgrid_cuda_fd_gradient_dot_gradientW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_2delta2, bc, src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Get the maximum value contained in a grid.
 *
 * grid   = Grid (rgrid *; input).
 * value  = maximum value found (REAL *; output).
 *
 */

EXPORT REAL rgrid_cuda_max(rgrid *grid, REAL *value) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  grid_cuda_maxW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);
  cuda_get_element(grid_gpu_mem, 0, 0, sizeof(REAL), value);
  return 0;
}

/*
 * Get the minimum value contained in a grid.
 *
 * grid   = Grid (rgrid *; input).
 * value  = maximum value found (REAL *; output).
 *
 */

EXPORT REAL rgrid_cuda_min(rgrid *grid, REAL *value) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  grid_cuda_minW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);
  cuda_get_element(grid_gpu_mem, 0, 0, sizeof(REAL), value);
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

  if(rot->host_lock || fx->host_lock || fy->host_lock || fz->host_lock) {
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

/* 
 * Rise a grid to given integer power.
 *
 * dst      = destination grid (rgrid *; output).
 * src      = source grid (rgrid *; input).
 * exponent = exponent to be used (INT; input).
 *
 */

EXPORT char rgrid_cuda_ipower(rgrid *dst, rgrid *src, INT exponent) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
    return -1;

  rgrid_cuda_ipowerW(cuda_block_address(dst->value), cuda_block_address(src->value), exponent, src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Set a value to given grid based on upper/lower limit thresholds of another grid (possibly the same).
 *
 * dst  = destination grid (rgrid *; input/output).
 * src  = source grid for evaluating the thresholds (rgrid *; input). May be equal to dest.
 * ul   = upper limit threshold for the operation (REAL; input).
 * ll   = lower limit threshold for the operation (REAL; input).
 * uval = value to set when the upper limit was exceeded (REAL; input).
 * lval = value to set when the lower limit was exceeded (REAL; input).
 *
 */

EXPORT char rgrid_cuda_threshold_clear(rgrid *dst, rgrid *src, REAL ul, REAL ll, REAL uval, REAL lval) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0)
    return -1;

  rgrid_cuda_threshold_clearW(cuda_block_address(dst->value), cuda_block_address(src->value), ul, ll, uval, lval, src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Zero a range of complex grid.
 *
 * grid = grid to be cleared (rgrid *; input/output).
 * lx       = Lower limit for x index (INT; input).
 * hx       = Upper limit for x index (INT; input).
 * ly       = Lower limit for y index (INT; input).
 * hy       = Upper limit for y index (INT; input).
 * lz       = Lower limit for z index (INT; input).
 * hz       = Upper limit for z index (INT; input).
 *
 */

EXPORT char rgrid_cuda_zero_index(rgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(hx > grid->nx || lx < 0 || hy > grid->ny || ly < 0 || hz > grid->nz || lz < 0) return 0; // not part of the grid

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_zero_indexW(cuda_block_address(grid->value), lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Solve Poisson equation (starting in Fourier space).
 *
 * grid = destination grid (rgrid *; input/output).
 *
 * NOTE: grid must be FFT'd first so that grid->value is complex.
 *
 */

EXPORT char rgrid_cuda_poisson(rgrid *grid) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_c2r, grid->id, 1) < 0) return -1;

  rgrid_cuda_poissonW(cuda_block_address(grid->value), grid->fft_norm, grid->step * grid->step, grid->nx, grid->ny, grid->nz);

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
 * Calculate second derivative of a grid (in Fourier space).
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
 * Multiply grid by coordinate x.
 *
 * grid = grid for the operation (rgrid *; output).
 *
 */

EXPORT char rgrid_cuda_multiply_by_x(rgrid *grid) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_by_xW(cuda_block_address(grid->value), grid->x0, grid->step, grid->nx, grid->ny, grid->nz);

  return 0;
}


/*
 * Multiply grid by coordinate x.
 *
 * grid = grid for the operation (rgrid *; output).
 *
 */

EXPORT char rgrid_cuda_multiply_by_y(rgrid *grid) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_by_yW(cuda_block_address(grid->value), grid->y0, grid->step, grid->nx, grid->ny, grid->nz);

  return 0;
}


/*
 * Multiply grid by coordinate z.
 *
 * grid = grid for the operation (rgrid *; output).
 *
 */

EXPORT char rgrid_cuda_multiply_by_z(rgrid *grid) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_by_zW(cuda_block_address(grid->value), grid->z0, grid->step, grid->nx, grid->ny, grid->nz);

  return 0;
}
