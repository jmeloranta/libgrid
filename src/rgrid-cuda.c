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

  rgrid_cuda_fft_convoluteW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/*
 * Multiply two grids (in Fourier space).
 *
 * dst  = Destination (rgrid *; output).
 * src1 = Source 1 (rgrid *; input).
 * src2 = Source 2 (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_fft_product(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_c2r, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_c2r, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_c2r, dst->id, 0) < 0) return -1;

  rgrid_cuda_fft_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

  return 0;
}

/*
 * Add two grids (in Fourier space).
 *
 * dst  = Destination (rgrid *; output).
 * src1 = Source 1 (rgrid *; input).
 * src2 = Source 2 (rgrid *; input).
 *
 */

EXPORT char rgrid_cuda_fft_sum(rgrid *dst, rgrid *src1, rgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_c2r, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_c2r, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle_c2r, dst->id, 0) < 0) return -1;

  rgrid_cuda_fft_sumW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), src1->nx, src1->ny, src1->nz);

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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

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

EXPORT char rgrid_cuda_fft_multiply(rgrid *grid, REAL c) {

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle_c2r, grid->id, 1) < 0) return -1;

  rgrid_cuda_fft_multiplyW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

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
 * (int opgrid dgrid^2).
 *
 * dgrid   = first grid for integration (rgrid *; input).
 * opgrid  = second grid for integration (rgrid *; input).
 * value   = integral value (REAL *; output).
 *
 */

EXPORT char rgrid_cuda_grid_expectation_value(rgrid *dgrid, rgrid *opgrid, REAL *value) {

  if(dgrid->host_lock || opgrid->host_lock) {
    cuda_remove_block(dgrid->value, 1);
    cuda_remove_block(opgrid->value, 1);
    return -1;
  }

  if(cuda_two_block_policy(opgrid->value, opgrid->grid_len, opgrid->cufft_handle_r2c, opgrid->id, 1, dgrid->value, dgrid->grid_len, dgrid->cufft_handle_r2c, dgrid->id, 1) < 0)
    return -1;

  rgrid_cuda_grid_expectation_valueW(cuda_block_address(dgrid->value), cuda_block_address(opgrid->value), dgrid->nx, dgrid->ny, dgrid->nz, value);

  if(dgrid->nx != 1) *value *= dgrid->step;
  if(dgrid->ny != 1) *value *= dgrid->step;
  if(dgrid->nz != 1) *value *= dgrid->step;

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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  rgrid_cuda_maxW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);

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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  rgrid_cuda_minW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, value);

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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  rgrid_cuda_zero_indexW(cuda_block_address(grid->value), lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);
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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

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

  if(cuda_misc_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  rgrid_cuda_multiply_by_zW(cuda_block_address(grid->value), grid->z0, grid->step, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Natural logarithm of absolute value of grid.
 *
 * dst    = destination grid (rgrid *; output).
 * src    = source grid (rgrid *; input).
 * eps    = exponent to be used (REAL; input).
 *
 */

EXPORT char rgrid_cuda_log(rgrid *dst, rgrid *src, REAL eps) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0)
    return -1;

  rgrid_cuda_logW(cuda_block_address(dst->value), cuda_block_address(src->value), eps, src->nx, src->ny, src->nz);

  return 0;
}
