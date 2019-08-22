/*
 * Interface to CUDA. Real version (single or double precision).
 *
 * Due to transfer between main memory and GPU memory, user may have to call these directly.
 *
 * Only periodic boundaries are supported.
 *
 * Supports only periodic boundaries (not vortex).
 *
 */


#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "grid.h"

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

EXPORT int cgrid_cufft_alloc(cgrid *grid) {

  cufftResult status;
  size_t len[MAX_CUDA_DESCRIPTOR_GPUS];

  cufftCreate(&(grid->cufft_handle));
  cufftXtSetGPUs(grid->cufft_handle, cuda_ngpus(), cuda_gpus());  
#ifdef SINGLE_PREC
  if((status = cufftMakePlan3d(grid->cufft_handle, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_C2C, len)) != CUFFT_SUCCESS) {
#else /* double */
  if((status = cufftMakePlan3d(grid->cufft_handle, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_Z2Z, len)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(CUDA): Error in forward cplan: ");
    error_check(status);
    exit(1);
  }
  error_check(status);
  return 0; 
}

/*
 * Forward FFT using cufft (in-place).
 *
 * grid = Grid to be transformed (cgrid *; input/output).
 * 
 * C2C (float) or Z2Z (double) transformation.
 *
 * No return value.
 *
 */

EXPORT int cgrid_cufft_fft(cgrid *grid) {
  
  cufftResult status;

  if(grid->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    exit(1);
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

#ifdef SINGLE_PREC
  if((status = cufftXtExecDescriptorC2C(grid->cufft_handle, (cuda_find_block(grid->value))->gpu_info, (cuda_find_block(grid->value))->gpu_info, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
#else
  if((status = cufftXtExecDescriptorZ2Z(grid->cufft_handle, (cuda_find_block(grid->value))->gpu_info, (cuda_find_block(grid->value))->gpu_info, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(CUDA): Error in CFFT (forward): ");
    error_check(status);
    exit(1);
  }

  cuda_error_check();

  return 0;
}

/*
 * Backward FFT using cufft when data already in GPU memory. 
 *
 * grid = Grid to be transformed (cgrid *; input/output).
 * 
 * C2C (float) or Z2Z (double) transformation.
 *
 * No return value.
 *
 */

EXPORT int cgrid_cufft_fft_inv(cgrid *grid) {

  cufftResult status;

  if(grid->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    exit(1);
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

#ifdef SINGLE_PREC
  if((status = cufftXtExecDescriptorC2C(grid->cufft_handle, (cuda_find_block(grid->value))->gpu_info, (cuda_find_block(grid->value))->gpu_info, CUFFT_INVERSE)) != CUFFT_SUCCESS) {
#else /* double */
  if((status = cufftXtExecDescriptorZ2Z(grid->cufft_handle, (cuda_find_block(grid->value))->gpu_info, (cuda_find_block(grid->value))->gpu_info, CUFFT_INVERSE)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(CUDA): Error in CFFT (backward): ");
    error_check(status);
    exit(1);
  }

  cuda_error_check();

  return 0;
}
