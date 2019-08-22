/*
 * Since .cu cannot deal with complex data type, we need to use wrapper routines to functions in .cu :-(
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
#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cufft.h>

/* Global variables */
void *grid_gpu_mem = NULL; // Temp space for GPU (host memory pointer; host_)
cudaXtDesc *grid_gpu_mem_addr = NULL; // cuda_block_address() of grid_gpu_mem

/*
 * Initialize the CUDA portion of cgrid routines. Common block for each GPU for reduction.
 *
 */

EXPORT void cgrid_cuda_init(size_t len) {

  static size_t prev_len = 0;
  int *gpus;

  if(!cuda_status()) return;  // TODO: must be called somehow if cuda is activated later
  gpus = cuda_gpus();
  cudaSetDevice(gpus[0]);
  if(prev_len < len) {
    if(grid_gpu_mem) {
      if(prev_len) {
        cuda_unlock_block(grid_gpu_mem);
        cuda_remove_block(grid_gpu_mem, 0);
      }
      cudaFreeHost(grid_gpu_mem);    
    }
    prev_len = len;
    if(cudaMallocHost((void **) &grid_gpu_mem, len) != cudaSuccess) {
      fprintf(stderr, "libgrid(CUDA): Not enough memory in cgrid_cuda_init().\n");
      abort();
    }
    if(!(cuda_add_block(grid_gpu_mem, len, -1, "GPU TEMP", 0))) {  /* Reserve memory on ALL GPUs (cufft_handle = -1) */
      fprintf(stderr, "libgrid(CUDA): Failed to allocate temporary space on GPU.\n");
      abort();
    }
    grid_gpu_mem_addr = cuda_block_address(grid_gpu_mem);
    cuda_lock_block(grid_gpu_mem);
  }
}

/*
 * Convolute two grids (in Fourier space).
 *
 * dst   = Destination (cgrid *; output).
 * src1  = Source 1 (cgrid *; input).
 * src2  = Source 2 (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_fft_convolute(cgrid *dst, cgrid *src1, cgrid *src2) {

  if(src1->space == 0 || src2->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  dst->space = 1; // Result in Fourier space
  cgrid_cuda_fft_convoluteW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->fft_norm2, dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * Rise a |grid| to given power.
 *
 * dst      = destination grid (cgrid *; output).
 * src      = source grid (cgrid *; input).
 * exponent = exponent to be used (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_abs_power(cgrid *dst, cgrid *src, REAL exponent) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_abs_powerW(cuda_block_address(dst->value), cuda_block_address(src->value), exponent, dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * Rise grid to given power.
 *
 * dst      = destination grid (cgrid *; output).
 * src      = 1st source grid (cgrid *; input).
 * exponent = exponent to be used (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_power(cgrid *dst, cgrid *src, REAL exponent) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_powerW(cuda_block_address(dst->value), cuda_block_address(src->value), exponent, dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (cgrid *; input/output).
 * c    = multiplier (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_multiply(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_multiplyW(cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/*
 * Add two grids: dst = src1 + src2
 *
 * dst  = destination grid (cgrid *; output).
 * src1 = 1st of the grids to be added (cgrid *; input).
 * src2 = 2nd of the grids to be added (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_sum(cgrid *dst, cgrid *src1, cgrid *src2) {

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_sumW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Subtract two grids: dst = src1 - src2
 *
 * dst = destination grid (cgrid *; output).
 * src1 = 1st of the grids for subtraction (cgrid *; input).
 * src2 = 2nd of the grids for subtraction (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_difference(cgrid *dst, cgrid *src1, cgrid *src2) {
  
  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_differenceW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * Calculate product of two grids: dst = src1 * src2
 *
 * dst  = destination grid (cgrid *; output).
 * src1 = 1st source grid (cgrid *; input).
 * src2 = 2nd source grid (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_product(cgrid *dst, cgrid *src1, cgrid *src2) {

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * Calculate conjugate product of two grids: dst = src1^* X src2
 *
 * dst  = destination grid (cgrid *; output).
 * src1 = 1st source grid (complex conjugated) (cgrid *; input).
 * src2 = 2nd source grid (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_conjugate_product(cgrid *dst, cgrid *src1, cgrid *src2) {

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_conjugate_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * Divide two grids: dst = src1 / src2
 *
 * dst  = destination grid (cgrid *; output).
 * src1 = 1st source grid (cgrid *; input).
 * src2 = 2nd source grid (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_division(cgrid *dst, cgrid *src1, cgrid *src2) {

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_divisionW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/* 
 * "Safely" divide two grids: dst = src1 / (src2 + eps)
 *
 * dst  = destination grid (cgrid *; output).
 * src1 = 1st source grid (cgrid *; input).
 * src2 = 2nd source grid (cgrid *; input).
 * eps  = Epsilon (REAL; input).
 *
 */

EXPORT char cgrid_cuda_division_eps(cgrid *dst, cgrid *src1, cgrid *src2, REAL eps) {

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_division_epsW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), eps, dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Add a constant to grid.
 *
 * grid = grid to be operated on (cgrid *; input/output).
 * c    = constant (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_add(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_addW(cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (cgrid *; input/output).
 * cm   = multiplier (REAL complex; input).
 * ca   = constant to be added (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_multiply_and_add(cgrid *grid, REAL complex cm, REAL complex ca) {

  CUCOMPLEX ccm, cca;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  ccm.x = CREAL(cm);
  ccm.y = CIMAG(cm);
  cca.x = CREAL(ca);
  cca.y = CIMAG(ca);
  cgrid_cuda_multiply_and_addW(cuda_block_address(grid->value), ccm, cca, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (cgrid *; input/output).
 * ca   = constant to be added (REAL complex; input).
 * cm   = multiplier (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_add_and_multiply(cgrid *grid, REAL complex ca, REAL complex cm) {

  CUCOMPLEX ccm, cca;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  ccm.x = CREAL(cm);
  ccm.y = CIMAG(cm);
  cca.x = CREAL(ca);
  cca.y = CIMAG(ca);
  cgrid_cuda_add_and_multiplyW(cuda_block_address(grid->value), cca, ccm, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/* 
 * Add scaled grid (multiply/add): dst = dst + d * src
 *
 * dst   = destination grid for the operation (cgrid *; input/output).
 * d     = multiplier for the operation (REAL complex; input).
 * src   = source grid for the operation (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_add_scaled(cgrid *dst, REAL complex d, cgrid *src) {

  CUCOMPLEX dd;

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1) < 0) return -1;

  dd.x = CREAL(d);
  dd.y = CIMAG(d);
  cgrid_cuda_add_scaledW(cuda_block_address(dst->value), dd, cuda_block_address(src->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Perform the following operation: dst = dst + d * src1 * src2.
 *
 * dst   = destination grid (cgrid *; input/output).
 * d     = constant multiplier (REAL complex; input).
 * src1  = 1st source grid (cgrid *; input).
 * src2  = 2nd source grid (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_add_scaled_product(cgrid *dst, REAL complex d, cgrid *src1, cgrid *src2) {

  CUCOMPLEX dd;

  if(cuda_three_block_policy(src1->value, src1->grid_len, src1->cufft_handle, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle, src2->id, 1, 
                             dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1) < 0) return -1;

  dd.x = CREAL(d);
  dd.y = CIMAG(d);
  cgrid_cuda_add_scaled_productW(cuda_block_address(dst->value), dd, cuda_block_address(src1->value), cuda_block_address(src2->value), dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Copy grids.
 *
 * dst  = destination (cgrid *; output).
 * src  = source (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_copy(cgrid *dst, cgrid *src) {

  if(cuda_copy_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, src->value, src->grid_len, src->cufft_handle, src->id) < 0) return -1;

  cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));
  dst->space = src->space;
  return 0;
}

/*
 * Set grid value to constant.
 *
 * grid = grid for operation (cgrid *; output).
 * c    = constant (REAL complex; input).
 *
 */

EXPORT char cgrid_cuda_constant(cgrid *grid, REAL complex c) {

  CUCOMPLEX cc;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 0) < 0) return -1;

  cc.x = CREAL(c);
  cc.y = CIMAG(c);
  cgrid_cuda_constantW(cuda_block_address(grid->value), cc, grid->nx, grid->ny, grid->nz, grid->space);

  return 0;
}

/*
 * Integrate over given grid.
 *
 * grid  = grid for integration (cgrid *; input).
 * value = integration result (REAL complex *; output).
 *
 */

EXPORT char cgrid_cuda_integral(cgrid *grid, REAL complex *value) {

  REAL step = grid->step;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_integralW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, (CUCOMPLEX *) value, grid->space);

  if(grid->nx != 1) *value *= step;
  if(grid->ny != 1) *value *= step;
  *value *= step;

  return 0;
}

/*
 * Integrate over a grid with limits.
 *
 * grid  = grid to be integrated (cgrid *; input).
 * xl    = lower limit for x (REAL; input).
 * xu    = upper limit for x (REAL; input).
 * yl    = lower limit for y (REAL; input).
 * yu    = upper limit for y (REAL; input).
 * zl    = lower limit for z (REAL; input).
 * zu    = upper limit for z (REAL; input).
 * value = integration result (REAL complex *; output).
 *
 */

EXPORT char cgrid_cuda_integral_region(cgrid *grid, INT il, INT iu, INT jl, INT ju, INT kl, INT ku, REAL complex *value) {

  REAL step = grid->step;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_integral_regionW(cuda_block_address(grid->value), il, iu, jl, ju, kl, ku, grid->nx, grid->ny, grid->nz, (CUCOMPLEX *) value, grid->space);

  if(grid->nx != 1) *value *= step;
  if(grid->ny != 1) *value *= step;
  *value *= step;

  return 0;
}

/* 
 * Integrate over the grid squared (int grid^2).
 *
 * grid  = grid for integration (cgrid *; input).
 * value = value of the integral (REAL *; output).
 *
 */

EXPORT char cgrid_cuda_integral_of_square(cgrid *grid, REAL *value) {

  REAL step = grid->step;

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_integral_of_squareW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, (CUCOMPLEX *) value, grid->space);

  if(grid->nx != 1) *value *= step;
  if(grid->ny != 1) *value *= step;
  *value *= step;

  return 0;
}

/*
 * Calculate overlap between two grids (int grida gridb).
 *
 * grida = grid A (cgrid *; input).
 * gridb = grid B (cgrid *; input).
 * value = result of the integration (REAL complex *; output).
 *
 */

EXPORT char cgrid_cuda_integral_of_conjugate_product(cgrid *grida, cgrid *gridb, REAL complex *value) {

  REAL step = grida->step;

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->cufft_handle, grida->id, 1, gridb->value, gridb->grid_len, gridb->cufft_handle, gridb->id, 1) < 0) return -1;

  cgrid_cuda_integral_of_conjugate_productW(cuda_block_address(grida->value), cuda_block_address(gridb->value), gridb->nx, gridb->ny, gridb->nz, (CUCOMPLEX *) value, gridb->space);

  if(gridb->nx != 1) *value *= step;
  if(gridb->ny != 1) *value *= step;
  *value *= step;

  return 0;
}

/*
 * Calculate the expectation value over a grid: int gridb grida gridb = int grida gridb^2
 *
 * grida = grid A (cgrid *; input).
 * gridb = grid B (cgrid *; input).
 * value = result of integration (REAL complex *; output).
 *
 */

EXPORT char cgrid_cuda_grid_expectation_value(cgrid *grida, cgrid *gridb, REAL complex *value) {

  REAL step = grida->step;

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->cufft_handle, grida->id, 1, gridb->value, gridb->grid_len, gridb->cufft_handle, gridb->id, 1) < 0) return -1;

  cgrid_cuda_grid_expectation_valueW(cuda_block_address(grida->value), cuda_block_address(gridb->value), gridb->nx, gridb->ny, gridb->nz, (CUCOMPLEX *) value, gridb->space);

  if(gridb->nx != 1) *value *= step;
  if(gridb->ny != 1) *value *= step;
  *value *= step;

  return 0;
}

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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_gradient_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_gradient_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplaceW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplace_xW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0)  return -1;

  cgrid_cuda_fd_laplace_yW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
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

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_fd_laplace_zW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_delta2, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
  return 0;
}

/*
 * Calculate dot product of the gradient of the grid.
 *
 * src         = source grid (cgrid *; input).
 * dst         = destination grid (cgrid *; output).
 * inv2_delta2 = 1 / (2.0 * step * 2.0 * step) (REAL; input).
 * bc          = Boundary condition: 0 = Dirichlet, 1 = Neumann, 2 = Periodic (char; input).
 *
 */

EXPORT char cgrid_cuda_fd_gradient_dot_gradient(cgrid *src, cgrid *dst, REAL inv_2delta2, char bc) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0) < 0) return -1;

  cgrid_cuda_fd_gradient_dot_gradientW(cuda_block_address(src->value), cuda_block_address(dst->value), inv_2delta2, bc, dst->nx, dst->ny, dst->nz);  // FD are single GPU, no dst->space needed
  return 0;
}

/*
 * Take complex conjugate of grid.
 * 
 * dst    = destination grid (cgrid *; output).
 * src    = source grid (cgrid *; input).
 *
 */

EXPORT char cgrid_cuda_conjugate(cgrid *dst, cgrid *src) {

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  cgrid_cuda_conjugateW(cuda_block_address(dst->value), cuda_block_address(src->value), dst->nx, dst->ny, dst->nz, dst->space);
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

  if(src->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0)  return -1;

  if (dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  dst->space = 1;
  cgrid_cuda_fft_gradient_xW(cuda_block_address(dst->value), dst->fft_norm, dst->kx0, dst->step, dst->nx, dst->ny, dst->nz, dst->space);

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

  if(src->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  dst->space = 1;
  cgrid_cuda_fft_gradient_yW(cuda_block_address(dst->value), dst->fft_norm, dst->ky0, dst->step, dst->nx, dst->ny, dst->nz, dst->space);

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

  if(src->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  dst->space = 1;
  cgrid_cuda_fft_gradient_zW(cuda_block_address(dst->value), dst->fft_norm, dst->kz0, dst->step, dst->nx, dst->ny, dst->nz, dst->space);

  return 0;
}

/*
 * Calculate second derivative of a grid (in Fourier space).
 *
 * src   = source grid (cgrid *; input).
 * dst   = destination grid (cgrid *; output).
 *
 */

EXPORT char cgrid_cuda_fft_laplace(cgrid *src, cgrid *dst) {

  if(src->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_two_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 0, src->value, src->grid_len, src->cufft_handle, src->id, 1) < 0) return -1;

  if(dst != src) cuda_gpu2gpu(cuda_find_block(dst->value), cuda_find_block(src->value));

  dst->space = 1;
  cgrid_cuda_fft_laplaceW(cuda_block_address(dst->value), dst->fft_norm, dst->kx0, dst->ky0, dst->kz0, dst->step, dst->nx, dst->ny, dst->nz, dst->space);

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

  REAL step = laplace->step, norm = laplace->fft_norm, value2;

  if(laplace->space == 0) {
    fprintf(stderr, "libgrid(CUDA): Data not in Fourier space (convolution).\n");
    exit(1);
  }

  if(cuda_one_block_policy(laplace->value, laplace->grid_len, laplace->cufft_handle, laplace->id, 1) < 0) return -1;

  cgrid_cuda_fft_laplace_expectation_valueW(cuda_block_address(laplace->value), laplace->kx0, laplace->ky0, laplace->kz0, laplace->step, laplace->nx, laplace->ny, laplace->nz, (CUCOMPLEX *) &value2, laplace->space);
  *value = CREAL(value2);

  if(laplace->nx != 1) *value *= step;
  if(laplace->ny != 1) *value *= step;
  *value *= step * norm;

  return 0;
}

/*
 * Clear real part of complex grid.
 *
 * grid = grid to be cleared (cgrid *; input/output).
 *
 */

EXPORT char cgrid_cuda_zero_re(cgrid *grid) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_reW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, grid->space);
  return 0;
}

/*
 * Clear imaginary part of complex grid.
 *
 * grid = grid to be cleared (cgrid *; input/output).
 *
 */

EXPORT char cgrid_cuda_zero_im(cgrid *grid) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_imW(cuda_block_address(grid->value), grid->nx, grid->ny, grid->nz, grid->space);
  return 0;
}

/*
 * Zero a range of complex grid.
 *
 * grid = grid to be cleared (cgrid *; input/output).
 * lx       = Lower limit for x index (INT; input).
 * hx       = Upper limit for x index (INT; input).
 * ly       = Lower limit for y index (INT; input).
 * hy       = Upper limit for y index (INT; input).
 * lz       = Lower limit for z index (INT; input).
 * hz       = Upper limit for z index (INT; input).
 *
 */

EXPORT char cgrid_cuda_zero_index(cgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cuda_zero_indexW(cuda_block_address(grid->value), lx, hx, ly, hy, lz, hz, grid->nx, grid->ny, grid->nz, grid->space);
  return 0;
}

/*
 * Solve Poisson equation.
 *
 * grid = destination grid (cgrid *; input/output).
 *
 */

EXPORT char cgrid_cuda_poisson(cgrid *grid) {

  if(cuda_one_block_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id, 1) < 0) return -1;

  cgrid_cufft_fft(grid);
  cgrid_cuda_poissonW(cuda_block_address(grid->value), grid->fft_norm, grid->step * grid->step, grid->nx, grid->ny, grid->nz, grid->space);
  cgrid_cufft_fft_inv(grid);

  return 0;
}
