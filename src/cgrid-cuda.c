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

void *grid_gpu_mem = NULL; // Temp space for GPU (host memory pointer; host_mem)
void *grid_gpu_mem_addr = NULL; // cuda_block_address() of grid_gpu_mem

/*
 * Initialize the CUDA portion of cgrid routines.
 *
 */

EXPORT void cgrid_cuda_init(size_t len) { /* We use FFTW malloc routines just in case FFTW should be used for the grid */

  static size_t prev_len = 0;

  if(!cuda_status()) return;  // TODO: must be called somehow if cuda is activated later
  if(prev_len < len) {
    if(grid_gpu_mem) {
      if(prev_len) {
        cuda_unlock_block(grid_gpu_mem);
        cuda_remove_block(grid_gpu_mem, 0);
      }
#ifdef SINGLE_PREC
      fftwf_free(grid_gpu_mem);
#else
      fftw_free(grid_gpu_mem);
#endif
    }
    prev_len = len;
#ifdef SINGLE_PREC
    if (!(grid_gpu_mem = (void *) fftwf_malloc(len))) {
#else
    if (!(grid_gpu_mem = (void *) fftw_malloc(len))) {
#endif
      fprintf(stderr, "libgrid(CUDA): Not enough memory in cgrid_cuda_init().\n");
      exit(1);
    }
    if(!(cuda_add_block(grid_gpu_mem, len, "GPU TEMP", 0))) {
      fprintf(stderr, "libgrid(CUDA): Failed to allocate temporary space on GPU.\n");
      exit(1);
    }
    grid_gpu_mem_addr = cuda_block_address(grid_gpu_mem);
    cuda_lock_block(grid_gpu_mem);
  }
}

/*
 * Convolute two grids (in Fourier space).
 *
 * gridc = Destination (cgrid *).
 * grida = Source 1 (cgrid *).
 * gridb = Source 2 (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_fft_convolute(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_fft_convoluteW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->fft_norm2, grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Rise a |grid| to given power.
 *
 * gridb    = destination grid (cgrid *).
 * grida    = 1st source grid (cgrid *).
 * exponent = exponent to be used (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_abs_power(cgrid *gridb, cgrid *grida, REAL exponent) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 0) < 0) return -1;

  cgrid_cuda_abs_powerW((CUCOMPLEX *) cuda_block_address(gridb->value), (CUCOMPLEX *) cuda_block_address(grida->value), exponent, grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Rise grid to given power.
 *
 * gridb    = destination grid (cgrid *).
 * grida    = 1st source grid (cgrid *).
 * exponent = exponent to be used (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_power(cgrid *gridb, cgrid *grida, REAL exponent) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 0) < 0) return -1;

  cgrid_cuda_powerW((CUCOMPLEX *) cuda_block_address(gridb->value), (CUCOMPLEX *) cuda_block_address(grida->value), exponent, grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (cgrid *).
 * c    = multiplier (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_multiply(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_multiplyW((CUCOMPLEX *) cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Add two grids ("gridc = grida + gridb").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st of the grids to be added (cgrid *).
 * gridb = 2nd of the grids to be added (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_sum(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_sumW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Subtract two grids ("gridc = grida - gridb").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st of the grids to be added (cgrid *).
 * gridb = 2nd of the grids to be added (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_difference(cgrid *gridc, cgrid *grida, cgrid *gridb) {
  
  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_differenceW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Calculate product of two grids ("gridc = grida * gridb").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st source grid (cgrid *).
 * gridb = 2nd source grid (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_product(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_productW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Calculate conjugate product of two grids ("gridc = grida^* X gridb").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st source grid (cgrid *).
 * gridb = 2nd source grid (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_conjugate_product(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_conjugate_productW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * Divide two grids ("gridc = grida / gridb").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st source grid (cgrid *).
 * gridb = 2nd source grid (cgrid *).
 *
 * No return value.
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_division(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_divisionW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/* 
 * "Safely" divide two grids ("gridc = grida / (gridb + eps)").
 *
 * gridc = destination grid (cgrid *).
 * grida = 1st source grid (cgrid *).
 * gridb = 2nd source grid (cgrid *).
 * eps   = Epsilon (REAL).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_division_eps(cgrid *gridc, cgrid *grida, cgrid *gridb, REAL eps) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;

  cgrid_cuda_division_epsW((CUCOMPLEX *) cuda_block_address(gridc->value), (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), eps, grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Add a constant to grid.
 *
 * grid = grid to be operated on (cgrid *).
 * c    = constant (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_add(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_addW((CUCOMPLEX *) cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (cgrid *).
 * cm   = multiplier (REAL complex).
 * ca   = constant to be added (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_multiply_and_add(cgrid *grid, REAL complex cm, REAL complex ca) {

  CUCOMPLEX ccm, cca;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  ccm.x = CREAL(cm);
  ccm.y = CIMAG(cm);
  cca.x = CREAL(ca);
  cca.y = CIMAG(ca);
  cgrid_cuda_multiply_and_addW((CUCOMPLEX *) cuda_block_address(grid->value), ccm, cca, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (cgrid *).
 * ca   = constant to be added (REAL complex).
 * cm   = multiplier (REAL complex).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_add_and_multiply(cgrid *grid, REAL complex ca, REAL complex cm) {

  CUCOMPLEX ccm, cca;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  ccm.x = CREAL(cm);
  ccm.y = CIMAG(cm);
  cca.x = CREAL(ca);
  cca.y = CIMAG(ca);
  cgrid_cuda_add_and_multiplyW((CUCOMPLEX *) cuda_block_address(grid->value), cca, ccm, grid->nx, grid->ny, grid->nz);

  return 0;
}

/* 
 * Add scaled grid (multiply/add): gridc = gridc + d * grida
 *
 * gridc = destination grid for the operation (cgrid *).
 * d     = multiplier for the operation (REAL complex).
 * grida = source grid for the operation (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_add_scaled(cgrid *gridc, REAL complex d, cgrid *grida) {

  CUCOMPLEX dd;

  if(cuda_two_block_policy(gridc->value, gridc->grid_len, gridc->id, 1, grida->value, grida->grid_len, grida->id, 1) < 0) return -1;

  dd.x = CREAL(d);
  dd.y = CIMAG(d);
  cgrid_cuda_add_scaledW((CUCOMPLEX *) cuda_block_address(gridc->value), dd, (CUCOMPLEX *) cuda_block_address(grida->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Perform the following operation: gridc = gridc + d * grida * gridb.
 *
 * gridc = destination grid (cgrid *).
 * d     = constant multiplier (REAL complex).
 * grida = 1st source grid (cgrid *).
 * gridb = 2nd source grid (cgrid *).
 *
 * Return value: 0 = OK, -1 = cannot be done on GPU.
 *
 */

EXPORT char cgrid_cuda_add_scaled_product(cgrid *gridc, REAL complex d, cgrid *grida, cgrid *gridb) {

  CUCOMPLEX dd;

  if(cuda_three_block_policy(gridc->value, gridc->grid_len, gridc->id, 1, grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  dd.x = CREAL(d);
  dd.y = CIMAG(d);
  cgrid_cuda_add_scaled_productW((CUCOMPLEX *) cuda_block_address(gridc->value), dd, (CUCOMPLEX *) cuda_block_address(grida->value), (CUCOMPLEX *) cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);

  return 0;
}

/*
 * Copy two areas in GPU. Source grid is on GPU.
 *
 */

EXPORT char cgrid_cuda_copy(cgrid *copy, cgrid *grid) {

  if(cuda_copy_policy(copy->value, copy->grid_len, copy->id, grid->value, grid->grid_len, grid->id) < 0) return -1;

  cuda_gpu2gpu(cuda_find_block(copy->value), cuda_find_block(grid->value), 0);
  return 0;
}

/*
 * Set grid value to constant.
 *
 */

EXPORT char cgrid_cuda_constant(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 0) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_constantW(cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz);

  return 0;
}

/*
 * Integrate over a grid.
 *
 */

EXPORT char cgrid_cuda_integral(cgrid *grid, REAL complex *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_integralW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL complex), value);
  *value *= grid->step * grid->step * grid->step;
  return 0;
}

/*
 * Integrate over a grid with limits.
 *
 */

EXPORT char cgrid_cuda_integral_region(cgrid *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, REAL complex *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_integral_regionW(cuda_block_address(grid->value), il, iu, jl, ju, kl, ku, grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL complex), value);
  *value *= grid->step * grid->step * grid->step;
  return 0;
}

/* 
 * Integrate over the grid squared (int grid^2).
 *
 */

EXPORT char cgrid_cuda_integral_of_square(cgrid *grid, REAL *value) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_integral_of_squareW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);
  *value *= grid->step * grid->step * grid->step;
  return 0;
}

/*
 * Calculate overlap between two grids (int grida gridb).
 *
 */

EXPORT char cgrid_cuda_integral_of_conjugate_product(cgrid *grida, cgrid *gridb, REAL complex *value) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  cgrid_cuda_integral_of_conjugate_productW(cuda_block_address(grida->value), cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);
  *value *= grida->step * grida->step * grida->step;
  return 0;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int gridb grida gridb = int grida gridb^2).
 *
 */

EXPORT char cgrid_cuda_grid_expectation_value(cgrid *grida, cgrid *gridb, REAL complex *value) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1) < 0) return -1;

  cgrid_cuda_grid_expectation_valueW(cuda_block_address(grida->value), cuda_block_address(gridb->value), grida->nx, grida->ny, grida->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL complex), value);
  *value *= grida->step * grida->step * grida->step;
  return 0;
}

/* 
 * Differentiate a grid with respect to x (central difference).
 * grid = source, gradient = dest.
 *
 */

EXPORT char cgrid_cuda_fd_gradient_x(cgrid *grid, cgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_xW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to y (central difference).
 * grid = source, gradient = dest.
 *
 */

EXPORT char cgrid_cuda_fd_gradient_y(cgrid *grid, cgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_yW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Differentiate a grid with respect to z (central difference).
 * grid = source, gradient = dest.
 *
 */

EXPORT char cgrid_cuda_fd_gradient_z(cgrid *grid, cgrid *gradient, REAL inv_delta) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, gradient->value, gradient->grid_len, gradient->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_zW(cuda_block_address(grid->value), cuda_block_address(gradient->value), inv_delta, grid->nx, grid->ny, grid->nz);
  return 0;
}

/* 
 * Laplace of a grid (central difference).
 * grid = source, laplace = dest.
 *
 */

EXPORT char cgrid_cuda_fd_laplace(cgrid *grid, cgrid *laplace, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplace->value, laplace->grid_len, laplace->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplaceW(cuda_block_address(grid->value), cuda_block_address(laplace->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (x component). This is the second derivative with respect to x.
 * grid = source, laplacex = dest.
 *
 */

EXPORT char cgrid_cuda_fd_laplace_x(cgrid *grid, cgrid *laplacex, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacex->value, laplacex->grid_len, laplacex->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplace_xW(cuda_block_address(grid->value), cuda_block_address(laplacex->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (y component). This is the second derivative with respect to y.
 * grid = source, laplacey = dest.
 *
 */

EXPORT char cgrid_cuda_fd_laplace_y(cgrid *grid, cgrid *laplacey, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacey->value, laplacey->grid_len, laplacey->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplace_yW(cuda_block_address(grid->value), cuda_block_address(laplacey->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate vector laplacian of the grid (z component). This is the second derivative with respect to z.
 * grid = source, laplacez = dest.
 *
 */

EXPORT char cgrid_cuda_fd_laplace_z(cgrid *grid, cgrid *laplacez, REAL inv_delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplacez->value, laplacez->grid_len, laplacez->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplace_zW(cuda_block_address(grid->value), cuda_block_address(laplacez->value), inv_delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate dot product of the gradient of the grid.
 *
 */

EXPORT char cgrid_cuda_fd_gradient_dot_gradient(cgrid *grid, cgrid *grad_dot_grad, REAL inv_2delta2) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, grad_dot_grad->value, grad_dot_grad->grid_len, grad_dot_grad->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_dot_gradientW(cuda_block_address(grid->value), cuda_block_address(grad_dot_grad->value), inv_2delta2, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Take complex conjugate of  grid.
 * 
 */

EXPORT char cgrid_cuda_conjugate(cgrid *conjugate, cgrid *grid) {

  if(cuda_two_block_policy(conjugate->value, conjugate->grid_len, conjugate->id, 0, grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_conjugateW(cuda_block_address(conjugate->value), cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Differentiate grid in the Fourier space along x.
 *
 */

EXPORT char cgrid_cuda_fft_gradient_x(cgrid *grid, cgrid *gradient_x) {

  if(cuda_two_block_policy(gradient_x->value, gradient_x->grid_len, gradient_x->id, 0, grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  if (gradient_x != grid) cuda_gpu2gpu(cuda_find_block(gradient_x->value), cuda_find_block(grid->value), 0);

  cgrid_cuda_fft_gradient_xW(cuda_block_address(gradient_x->value), grid->fft_norm, grid->kx0, grid->step, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Differentiate grid in the Fourier space along y.
 *
 */

EXPORT char cgrid_cuda_fft_gradient_y(cgrid *grid, cgrid *gradient_y) {

  if(cuda_two_block_policy(gradient_y->value, gradient_y->grid_len, gradient_y->id, 0, grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  if (gradient_y != grid) cuda_gpu2gpu(cuda_find_block(gradient_y->value), cuda_find_block(grid->value), 0);

  cgrid_cuda_fft_gradient_yW(cuda_block_address(gradient_y->value), grid->fft_norm, grid->ky0, grid->step, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Differentiate grid in the Fourier space along z.
 *
 */

EXPORT char cgrid_cuda_fft_gradient_z(cgrid *grid, cgrid *gradient_z) {

  if(cuda_two_block_policy(gradient_z->value, gradient_z->grid_len, gradient_z->id, 0, grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  if (gradient_z != grid) cuda_gpu2gpu(cuda_find_block(gradient_z->value), cuda_find_block(grid->value), 0);

  cgrid_cuda_fft_gradient_zW(cuda_block_address(gradient_z->value), grid->fft_norm, grid->kx0, grid->step, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate second derivative of a grid (in Fourier space).
 *
 */

EXPORT char cgrid_cuda_fft_laplace(cgrid *grid, cgrid *laplace) {

  if(cuda_two_block_policy(laplace->value, laplace->grid_len, laplace->id, 0, grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  if (laplace != grid) cuda_gpu2gpu(cuda_find_block(laplace->value), cuda_find_block(grid->value), 0);

  cgrid_cuda_fft_laplaceW(cuda_block_address(laplace->value), grid->fft_norm, grid->kx0, grid->ky0, grid->kz0, grid->step, grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Calculate expectation value of laplace operator in the Fourier space (int grid^* grid'').
 *
 */

EXPORT char cgrid_cuda_fft_laplace_expectation_value(cgrid *grid, cgrid *laplace, REAL *value) {

  if(cuda_two_block_policy(grid->value, grid->grid_len, grid->id, 1, laplace->value, laplace->grid_len, laplace->id, 0) < 0) return -1;

  if (laplace != grid)  cuda_gpu2gpu(cuda_find_block(laplace->value), cuda_find_block(grid->value), 0);

  cgrid_cuda_fft_laplace_expectation_valueW(cuda_block_address(laplace->value), grid->fft_norm, grid->kx0, grid->ky0, grid->kz0, grid->step, grid->nx, grid->ny, grid->nz);
  cuda_get_element(grid_gpu_mem, 0, sizeof(REAL), value);
  *value *= grid->step * grid->step * grid->step;
  return 0;
}

/*
 * Clear real part of complex grid.
 *
 */

EXPORT char cgrid_cuda_zero_re(cgrid *grid) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_reW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Clear imaginary part of complex grid.
 *
 */

EXPORT char cgrid_cuda_zero_im(cgrid *grid) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_imW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz);
  return 0;
}

/*
 * Zero a range of complex grid.
 *
 */

EXPORT char cgrid_cuda_zero_index(cgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  if(hx >= grid->nx || lx < 0 || hy >= grid->ny || ly < 0 || hz >= grid->nz || lz < 0) return 0; // not part of the grid
  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_indexW(cuda_block_address(grid->value), lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz);
  return 0;
}
