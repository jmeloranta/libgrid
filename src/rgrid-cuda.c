/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to the functions in .cu :-(
 *
 * These are denoted in the .cu file with suffix W appended to the function name.
 *
 * REAL complex (cgrid) versions.
 *
 */

#include "grid.h"
#include <cufft.h>

extern void *grid_gpu_mem; // Defined in cgrid.c

EXPORT void rgrid_cuda_init(size_t len) {

  cgrid_cuda_init(len);
}

/*
 * Convolute two grids (in Fourier space).
 *
 * gridc = Destination (rgrid *).
 * grida = Source 1 (rgrid *).
 * gridb = Source 2 (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_fft_convolute(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_fft_convoluteW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->fft_norm2, grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Rise a grid to given power.
 *
 * gridb    = destination grid (rgrid *).
 * grida    = source grid (rgrid *).
 * exponent = exponent to be used (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_power(rgrid *gridb, rgrid *grida, REAL exponent) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 0) < 0) return -1;

  rgrid_cuda_powerW((CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), exponent, grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Rise a |grid| to given power.
 *
 * gridb    = destination grid (rgrid *).
 * grida    = source grid (rgrid *).
 * exponent = exponent to be used (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_abs_power(rgrid *gridb, rgrid *grida, REAL exponent) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 0) < 0) return -1;

  rgrid_cuda_abs_powerW((CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), exponent, grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (rgrid *).
 * c    = multiplier (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_multiply(rgrid *grid, REAL c) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiplyW((CUREAL *) cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Multiply grid by a constant (in fft space).
 *
 * grid = grid to be multiplied (rgrid *).
 * c    = multiplier (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_multiply_fft(rgrid *grid, REAL c) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_fftW((CUCOMPLEX *) cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add two  grids ("gridc = grida + gridb").
 *
 * gridc = destination grid (rgrid *).
 * grida = 1st of the grids to be added (rgrid *).
 * gridb = 2nd of the grids to be added (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_sum(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_sumW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Subtract two  grids ("gridc = grida - gridb").
 *
 * gridc = destination grid (rgrid *).
 * grida = 1st of the grids to be added (rgrid *).
 * gridb = 2nd of the grids to be added (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_difference(rgrid *gridc, rgrid *grida, rgrid *gridb) {
  
  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_differenceW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Calculate product of two grids ("gridc = grida * gridb").
 *
 * gridc = destination grid (rgrid *).
 * grida = 1st source grid (rgrid *).
 * gridb = 2nd source grid (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_product(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_productW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Divide two grids ("gridc = grida / gridb").
 *
 * gridc = destination grid (rgrid *).
 * grida = 1st source grid (rgrid *).
 * gridb = 2nd source grid (rgrid *).
 *
 * No return value.
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_division(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_divisionW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * "Safely" divide two grids ("gridc = grida / (gridb + eps)").
 *
 * gridc = destination grid (rgrid *).
 * grida = 1st source grid (rgrid *).
 * gridb = 2nd source grid (rgrid *).
 * eps   = Epsilon (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_division_eps(rgrid *gridc, rgrid *grida, rgrid *gridb, REAL eps) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  rgrid_cuda_division_epsW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), eps, grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Add a constant to grid.
 *
 * grid = grid to be operated on (rgrid *).
 * c    = constant (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_add(rgrid *grid, REAL c) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_addW((CUREAL *) cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (rgrid *).
 * cm   = multiplier (REAL).
 * ca   = constant to be added (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_multiply_and_add(rgrid *grid, REAL cm, REAL ca) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_multiply_and_addW((CUREAL *) cuda_block_address(grid->value), cm, ca, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (rgrid *).
 * ca   = constant to be added (REAL).
 * cm   = multiplier (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_add_and_multiply(rgrid *grid, REAL ca, REAL cm) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_add_and_multiplyW((CUREAL *) cuda_block_address(grid->value), ca, cm, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add scaled grid (multiply/add): gridc = gridc + d * grida
 *
 * gridc = destination grid for the operation (rgrid *).
 * d     = multiplier for the operation (REAL).
 * grida = source grid for the operation (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_add_scaled(rgrid *gridc, REAL d, rgrid *grida) {

  if(cuda_two_block_policy(gridc->value, gridc->grid_len, gridc->id, 1, grida->value, grida->grid_len, grida->id, 1) < 0) return -1;

  rgrid_cuda_add_scaledW((CUREAL *) cuda_block_address(gridc->value), d, (CUREAL *) cuda_block_address(grida->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Perform the following operation: gridc = gridc + d * grida * gridb.
 *
 * gridc = destination grid (rgrid *).
 * d     = constant multiplier (REAL).
 * grida = 1st source grid (rgrid *).
 * gridb = 2nd source grid (rgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_add_scaled_product(rgrid *gridc, REAL d, rgrid *grida, rgrid *gridb) {

  if(cuda_three_block_policy(gridc->value, gridc->grid_len, gridc->id, 1, grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  rgrid_cuda_add_scaled_productW((CUREAL *) cuda_block_address(gridc->value), d, (CUREAL *) cuda_block_address(grida->value), (CUREAL *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Copy two areas in GPU. Source grid is on GPU.
 *
 */

EXPORT char rgrid_cuda_copy(rgrid *copy, rgrid *grid) {

  if(cuda_copy_policy(copy->value, copy->grid_len, copy->id, grid->value, grid->grid_len, grid->id) < 0) return -1;

  cuda_gpu2gpu(cuda_find_block(copy->value), cuda_find_block(grid->value), 0);
  return 0;
}

/*
 * Set grid value to constant.
 *
 */

EXPORT char rgrid_cuda_constant(rgrid *grid, REAL c) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 0) < 0) return -1;

  rgrid_cuda_constantW(cuda_block_address(grid->value), c, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Integrate over a grid.
 *
 */

EXPORT char rgrid_cuda_integral(rgrid *grid, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_integralW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  *value *= grid->step;

  return 0;
}

/*
 * Integrate over a grid with limits.
 *
 */

EXPORT char rgrid_cuda_integral_region(rgrid *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_integral_regionW(cuda_block_address(grid->value), il, iu, jl, ju, kl, ku, grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  *value *= grid->step;

  return 0;
}

/* 
 * Integrate over the grid squared (int grid^2).
 *
 */

EXPORT char rgrid_cuda_integral_of_square(rgrid *grid, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  rgrid_cuda_integral_of_squareW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);

  if(grid->nx != 1) *value *= grid->step;
  if(grid->ny != 1) *value *= grid->step;
  *value *= grid->step;

  return 0;
}

/*
 * Calculate overlap between two grids (int grida gridb).
 *
 */

EXPORT char rgrid_cuda_integral_of_product(rgrid *grida, rgrid *gridb, REAL *value) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  rgrid_cuda_integral_of_productW(cuda_block_address(grida->value), cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);

  if(grida->nx != 1) *value *= grida->step;
  if(grida->ny != 1) *value *= grida->step;
  *value *= grida->step;

  return 0;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int gridb grida gridb = int grida gridb^2).
 *
 */

EXPORT char rgrid_cuda_grid_expectation_value(rgrid *grida, rgrid *gridb, REAL *value) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  rgrid_cuda_grid_expectation_valueW(cuda_block_address(grida->value), cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);

  if(grida->nx != 1) *value *= grida->step;
  if(grida->ny != 1) *value *= grida->step;
  *value *= grida->step;

  return 0;
}

/* 
 * Differentiate a grid with respect to x (central difference).
 * grid = source, gradient = dest.
 *
 */

EXPORT char rgrid_cuda_fd_gradient_x(rgrid *grid, rgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  rgrid_cuda_fd_gradient_xW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to y (central difference).
 * grid = source, gradient = dest.
 *
 */

EXPORT char rgrid_cuda_fd_gradient_y(rgrid *grid, rgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  rgrid_cuda_fd_gradient_yW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to z (central difference).
 * grid = source, gridb = gradient.
 *
 */

EXPORT char rgrid_cuda_fd_gradient_z(rgrid *grid, rgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  rgrid_cuda_fd_gradient_zW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Laplace of a grid (central difference).
 * grid = source, laplace = dest.
 *
 */

EXPORT char rgrid_cuda_fd_laplace(rgrid *grid, rgrid *laplace, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplace->value, laplace->grid_len, laplace->id, 0) < 0) return -1;

  rgrid_cuda_fd_laplaceW(cuda_block_address(grid->value), cuda_block_address(laplace->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (x component). This is the second derivative with respect to x.
 * grid = source, laplacex = dest.
 *
 */

EXPORT char rgrid_cuda_fd_laplace_x(rgrid *grid, rgrid *laplacex, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacex->value, laplacex->grid_len, laplacex->id, 0) < 0) return -1;

  rgrid_cuda_fd_laplace_xW(cuda_block_address(grid->value), cuda_block_address(laplacex->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (y component). This is the second derivative with respect to y.
 * grid = source, laplacey = dest.
 *
 */

EXPORT char rgrid_cuda_fd_laplace_y(rgrid *grid, rgrid *laplacey, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacey->value, laplacey->grid_len, laplacey->id, 0) < 0) return -1;

  rgrid_cuda_fd_laplace_yW(cuda_block_address(grid->value), cuda_block_address(laplacey->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (z component). This is the second derivative with respect to z.
 * grida = source, gridb = dest.
 *
 */

EXPORT char rgrid_cuda_fd_laplace_z(rgrid *grid, rgrid *laplacez, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacez->value, laplacez->grid_len, laplacez->id, 0) < 0) return -1;

  rgrid_cuda_fd_laplace_zW(cuda_block_address(grid->value), cuda_block_address(laplacez->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate dot product of the gradient of the grid.
 *
 */

EXPORT char rgrid_cuda_fd_gradient_dot_gradient(rgrid *grid, rgrid *grad_dot_grad, REAL inv_2delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, grad_dot_grad->value, grad_dot_grad->grid_len, grad_dot_grad->id, 0) < 0) return -1;

  rgrid_cuda_fd_gradient_dot_gradientW(cuda_block_address(grid->value), cuda_block_address(grad_dot_grad->value), inv_2delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Copy a real grid to a complex grid (to real part).
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
 */

EXPORT char grid_cuda_real_to_complex_im(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_real_to_complex_imW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Add a real grid to a complex grid (to real part).
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
 */

EXPORT char grid_cuda_add_real_to_complex_im(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_add_real_to_complex_imW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Product of a real grid with a complex grid.
 *
 */

EXPORT char grid_cuda_product_complex_with_real(cgrid *dest, rgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_product_complex_with_realW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Copy imaginary part of a complex grid to a real grid.
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
 */

EXPORT char grid_cuda_complex_re_to_real(rgrid *dest, cgrid *source) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 0, source->value, source->grid_len, source->id, 1) < 0) return -1;

  grid_cuda_complex_re_to_realW(cuda_block_address(dest->value), cuda_block_address(source->value), dest->nx, dest->ny, dest->nz);
  return 0;
}

/*
 * Get the maximum value contained in a grid.
 *
 */

EXPORT REAL rgrid_cuda_max(rgrid *grid, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  grid_cuda_maxW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);
  return 0;
}

/*
 * Get the minimum value contained in a grid.
 *
 */

EXPORT REAL rgrid_cuda_min(rgrid *grid, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  grid_cuda_minW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);
  return 0;
}

/*
 * Calculate |rot| (|curl|; |\Nabla\times|) of a vector field (i.e., magnitude).
 *
 */

EXPORT char rgrid_cuda_abs_rot(rgrid *rot, rgrid *fx, rgrid *fy, rgrid *fz, REAL inv_delta) {

  if(cuda_four_block_policy(rot->value, rot->grid_len, rot->id, 0, fx->value, fx->grid_len, fx->id, 1,
                            fy->value, fy->grid_len, fy->id, 1, fz->value, fz->grid_len, fz->id, 1) < 0) return -1;

  rgrid_cuda_abs_rotW(cuda_block_address(rot->value), cuda_block_address(fx->value), cuda_block_address(fy->value),
                       cuda_block_address(fz->value), rot->nx, rot->ny, rot->nz, inv_delta);
  return 0;
}

/* 
 * Rise a grid to given integer power.
 *
 * gridb    = destination grid (rgrid *).
 * grida    = source grid (rgrid *).
 * exponent = exponent to be used (INT).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char rgrid_cuda_ipower(rgrid *gridb, rgrid *grida, INT exponent) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 0) < 0) return -1;

  rgrid_cuda_ipowerW((CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), exponent, grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Set a value to given grid based on upper/lower limit thresholds of another grid (possibly the same).
 *
 * dest = destination grid (rgrid *; input/output).
 * src  = source grid for evaluating the thresholds (rgrid *; input). May be equal to dest.
 * ul   = upper limit threshold for the operation (REAL; input).
 * ll   = lower limit threshold for the operation (REAL; input).
 * uval = value to set when the upper limit was exceeded (REAL; input).
 * lval = value to set when the lower limit was exceeded (REAL; input).
 *
 * No return value.
 */

EXPORT char rgrid_cuda_threshold_clear(rgrid *dest, rgrid *src, REAL ul, REAL ll, REAL uval, REAL lval) {

  if(cuda_two_block_policy(dest->value, dest->grid_len, dest->id, 1, src->value, src->grid_len, src->id, 1) < 0) return -1;

  rgrid_cuda_threshold_clearW((CUREAL *) cuda_block_address(dest->value), (CUREAL *) cuda_block_address(src->value), ul, ll, uval, lval, dest->nx, dest->ny, dest->nz);

  return 0;
}
