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

  grid_cuda_real_to_complex_reW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);

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

  grid_cuda_real_to_complex_imW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);

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

  grid_cuda_add_real_to_complex_reW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);

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

  grid_cuda_add_real_to_complex_imW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);

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

  grid_cuda_product_complex_with_realW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);
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

  grid_cuda_complex_im_to_realW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);
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

  grid_cuda_complex_re_to_realW(cuda_block_address(dst->value), cuda_block_address(src->value), src->nx, src->ny, src->nz);
  return 0;
}
