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
 * dest   = Destination grid (cgrid *; output).
 * source = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_real_to_complex_re(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_real_to_complex_reW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Copy a real grid to a complex grid (to imaginary part).
 *
 * dest   = Destination grid (cgrid *; output).
 * source = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_real_to_complex_im(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_real_to_complex_imW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Add a real grid to a complex grid (to real part).
 *
 * dest   = Destination grid (cgrid *; output).
 * source = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_add_real_to_complex_re(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_add_real_to_complex_reW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Add a real grid to a complex grid (to imag part).
 *
 * dest   = Destination grid (cgrid *; output).
 * source = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_add_real_to_complex_im(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_add_real_to_complex_imW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Product of a real grid with a complex grid.
 *
 * dest   = Destination grid (cgrid *; output).
 * source = Source grid (rgrid *; input).
 *
 */

EXPORT char grid_cuda_product_complex_with_real(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_product_complex_with_realW(cuda_block_address(dest->value), cuda_block_address(source->value), 
                                       dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Copy imaginary part of a complex grid to a real grid.
 *
 * dest   = Destination grid (rgrid *; output).
 * source = Source grid (cgrid *; input).
 *
 */

EXPORT char grid_cuda_complex_im_to_real(rgrid *dest, cgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_complex_im_to_realW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Copy real part of a complex grid to a real grid.
 *
 * dest   = Destination grid (rgrid *; output).
 * source = Source grid (cgrid *; input).
 *
 */

EXPORT char grid_cuda_complex_re_to_real(rgrid *dest, cgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_complex_re_to_realW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}
