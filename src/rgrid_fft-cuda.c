/*
 * Interface to CUDA. Real version (single or double precision).
 *
 * Due to transfer between main memory and GPU memory, user may have to call these directly.
 *
 * Only periodic boundaries are supported.
 *
 */

#include "grid.h"

char grid_cufft_workarea;  /* gpu block memory holder */
cufftHandle highest_plan = -1;

/*
 * Set up workspace for cufft manually in order to avoid double allocation for the various types of transformations.
 * Just make sure that there is enough space for the worst case.
 *
 */

static void rcufft_workspace(cufftHandle *plan, int nx, int ny, int nz) {

  size_t wrksize;
  gpu_mem_block *block;
  int i;

  cufftCreate(plan);
  if(*plan > highest_plan) highest_plan = *plan;
#ifdef SINGLE_PREC
  if(cufftGetSize3d(*plan, (int) nx, (int) ny, (int) nz, CUFFT_R2C, &wrksize) != CUFFT_SUCCESS) {
#else
  if(cufftGetSize3d(*plan, (int) nx, (int) ny, (int) nz, CUFFT_D2Z, &wrksize) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(cuda): CUFFT plan size failed.\n");
    exit(1);
  }
  cufftSetAutoAllocation(*plan, 0);
  if(!(block = cuda_find_block(&grid_cufft_workarea))) {
    block = cuda_add_block(&grid_cufft_workarea, wrksize, "cufft temp", 0);
    cuda_lock_block(&grid_cufft_workarea);
  } else if(wrksize > block->length) {
    cuda_unlock_block(&grid_cufft_workarea);
    cuda_remove_block(&grid_cufft_workarea, 0);
    block = cuda_add_block(&grid_cufft_workarea, wrksize, "cufft temp", 0);
    cuda_lock_block(&grid_cufft_workarea);
  }

  // need to update work areas of ALL plans!!! We will set it for all workspaces up to the higest plan number so far (hack)
  for (i = 0; i <= highest_plan; i++)
    cufftSetWorkArea(i, cuda_block_address(&grid_cufft_workarea));
}

static void error_check(cufftResult value) {

  switch(value) {
  case CUFFT_SUCCESS:
    fprintf(stderr, "Success.\n");
    break;
  case CUFFT_ALLOC_FAILED:
    fprintf(stderr, "GPU memory allocation failed.\n");
    break;
  case CUFFT_INVALID_VALUE:
    fprintf(stderr, "Invalid parameter values.\n");
    break;
  case CUFFT_INTERNAL_ERROR:
    fprintf(stderr, "Internal driver error.\n");
    break;
  case CUFFT_SETUP_FAILED:
    fprintf(stderr, "Library failed to initialize.\n");
    break;
  case CUFFT_INVALID_SIZE:
    fprintf(stderr, "Dimeion of nx, ny, or nz not supported.\n");
    break;
  default:
    fprintf(stderr, "Unknown cufft error code.\n");    
  }
}

/*
 * Forward FFT using cufft (in-place).
 *
 * grid = Grid to be transformed (rgrid *; input/output).
 * 
 * R2C (float) or D2Z (double) transformation.
 *
 * No return value.
 *
 */

EXPORT int rgrid_cufft_fft(rgrid *grid) {
  
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  cufftResult status;
  size_t len;

  if(grid->cufft_handle_r2c == -1) {
    rcufft_workspace(&(grid->cufft_handle_r2c), (int) nx, (int) ny, (int) nz);
#ifdef SINGLE_PREC
    if((status = cufftMakePlan3d(grid->cufft_handle_r2c, (int) nx, (int) ny, (int) nz, CUFFT_R2C, &len)) != CUFFT_SUCCESS) {
#else /* double */
    if((status = cufftMakePlan3d(grid->cufft_handle_r2c, (int) nx, (int) ny, (int) nz, CUFFT_D2Z, &len)) != CUFFT_SUCCESS) {
#endif
      fprintf(stderr, "libgrid(cuda): Error in forward rplan: ");
      error_check(status);
      fprintf(stderr, "Workspace length = %ld\n", len);
      exit(1);
    }
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->id) < 0) return -1;

#ifdef SINGLE_PREC
  if((status = cufftExecR2C(grid->cufft_handle_r2c, (cufftReal *) cuda_block_address(grid->value), (cufftComplex *) cuda_block_address(grid->value))) != CUFFT_SUCCESS) {
#else
  if((status = cufftExecD2Z(grid->cufft_handle_r2c, (cufftDoubleReal *) cuda_block_address(grid->value), (cufftDoubleComplex *) cuda_block_address(grid->value))) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
    error_check(status);
    exit(1);
  }

  cuda_error_check();

  return 0;
}

/*
 * Backward FFT using cufft when data already in GPU memory. 
 *
 * grid = Grid to be transformed (rgrid *; input/output).
 * 
 * C2R (float) or Z2D (double) transformation.
 *
 * No return value.
 *
 */

EXPORT int rgrid_cufft_fft_inv(rgrid *grid) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  cufftResult status;
  size_t len;

  if(grid->cufft_handle_c2r == -1) {
    rcufft_workspace(&(grid->cufft_handle_c2r), (int) nx, (int) ny, (int) nz);
#ifdef SINGLE_PREC
    if((status = cufftMakePlan3d(grid->cufft_handle_c2r, (int) nx, (int) ny, (int) nz, CUFFT_C2R, &len)) != CUFFT_SUCCESS) {
#else
    if((status = cufftMakePlan3d(grid->cufft_handle_c2r, (int) nx, (int) ny, (int) nz, CUFFT_Z2D, &len)) != CUFFT_SUCCESS) {
#endif
      fprintf(stderr, "libgrid(cuda): Error in backward rplan: ");
      error_check(status);
      fprintf(stderr, "Workspace length = %ld\n", len);
      exit(1);
    }
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->id) < 0) return -1;

#ifdef SINGLE_PREC
  if((status = cufftExecC2R(grid->cufft_handle_c2r, (cufftComplex *) cuda_block_address(grid->value), (cufftReal *) cuda_block_address(grid->value))) != CUFFT_SUCCESS) {
#else /* double */
  if((status = cufftExecZ2D(grid->cufft_handle_c2r, (cufftDoubleComplex *) cuda_block_address(grid->value), (cufftDoubleReal *) cuda_block_address(grid->value))) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(cuda): Error in FFT (backward): ");
    error_check(status);
    exit(1);
  }

  cuda_error_check();

  return 0;
}
