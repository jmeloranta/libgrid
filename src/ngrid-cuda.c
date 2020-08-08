/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * REAL complex (cgrid) / REAL (rgrid) mixed routines.
 *
 * Functions return 0 if operation was successful and -1 if not. The latter case usually means that it was against the 
 * policy function for the operation to run on GPU.
 *
 */

#include "grid.h"
#include <cufft.h>

/*
 * Copy a real grid to a complex grid (to real part).
 *
 * dst   = Destination grid (cgrid *; output).
 * src   = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_real_to_complex_re(cgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0) 
    return -1;

  grid_cuda_real_to_complex_reW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Copy a real grid to a complex grid (to imaginary part).
 *
 * dst   = Destination grid (cgrid *; output).
 * src   = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_real_to_complex_im(cgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0) 
    return -1;

  grid_cuda_real_to_complex_imW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Add a real grid to a complex grid (to real part).
 *
 * dst   = Destination grid (cgrid *; output).
 * src   = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_add_real_to_complex_re(cgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0) 
    return -1;

  grid_cuda_add_real_to_complex_reW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Add a real grid to a complex grid (to imag part).
 *
 * dst   = Destination grid (cgrid *; output).
 * src   = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_add_real_to_complex_im(cgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0) 
    return -1;

  grid_cuda_add_real_to_complex_imW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Product of real grid with sqnorm of complex grid.
 *
 * dst  = Destination grid (rgrid *; output).
 * src1 = Source grid 1 (rgrid *; input).
 * src2 = Source grid 2 (cgrid *; input).
 *
 */

EXPORT char grid_cuda_product_norm(rgrid *dst, rgrid *src1, cgrid *src2) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1,
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  grid_cuda_product_normW(cuda_find_block(dst->value), cuda_find_block(src1->value), cuda_find_block(src2->value), dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Divide real grid with sqnorm of complex grid.
 *
 * dst  = Destination grid (rgrid *; output).
 * src1 = Source grid 1 (rgrid *; input).
 * src2 = Source grid 2 (cgrid *; input).
 * eps  = Epsilon for division (REAL; input).
 *
 */

EXPORT char grid_cuda_division_norm(rgrid *dst, rgrid *src1, cgrid *src2, REAL eps) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1,
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  grid_cuda_division_normW(cuda_find_block(dst->value), cuda_find_block(src1->value), cuda_find_block(src2->value), eps, dst->nx, dst->ny, dst->nz);

  return 0;
}

/*
 * Product of a real grid with a complex grid.
 *
 * dst   = Destination grid (cgrid *; output).
 * src   = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_product_complex_with_real(cgrid *dst, rgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1, src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1) < 0) 
    return -1;

  grid_cuda_product_complex_with_realW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);

  return 0;
}

/*
 * Copy imaginary part of a complex grid to a real grid.
 *
 * dst   = Destination grid (rgrid *; output).
 * src   = Source grid (cgrid *; input).
 *
 */

EXPORT char grid_cuda_complex_im_to_real(rgrid *dst, cgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) 
    return -1;

  grid_cuda_complex_im_to_realW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Copy real part of a complex grid to a real grid.
 *
 * dst   = Destination grid (rgrid *; output).
 * src   = Source grid (cgrid *; input).
 *
 */

EXPORT char grid_cuda_complex_re_to_real(rgrid *dst, cgrid *src) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    return -1;
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) 
    return -1;

  grid_cuda_complex_re_to_realW(cuda_find_block(dst->value), cuda_find_block(src->value), src->nx, src->ny, src->nz);
  return 0;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int opgrid dgrid^2).
 *
 * dgrid   = first grid for integration (cgrid *; input).
 * opgrid  = second grid for integration (rgrid *; input).
 * value   = integral value (REAL *; output).
 *
 */

EXPORT char grid_cuda_grid_expectation_value(cgrid *dgrid, rgrid *opgrid, REAL *value) {

  if(dgrid->host_lock || opgrid->host_lock) {
    cuda_remove_block(dgrid->value, 1);
    cuda_remove_block(opgrid->value, 1);
    return -1;
  }

  if(cuda_two_block_policy(opgrid->value, opgrid->grid_len, opgrid->cufft_handle_r2c, opgrid->id, 1, dgrid->value, dgrid->grid_len, dgrid->cufft_handle, dgrid->id, 1) < 0)
    return -1;

  grid_cuda_grid_expectation_valueW(cuda_find_block(dgrid->value), cuda_find_block(opgrid->value), dgrid->nx, dgrid->ny, dgrid->nz, value);

  if(dgrid->nx != 1) *value *= dgrid->step;
  if(dgrid->ny != 1) *value *= dgrid->step;
  if(dgrid->nz != 1) *value *= dgrid->step;

  return 0;
}
