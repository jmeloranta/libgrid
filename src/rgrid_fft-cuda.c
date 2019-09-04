/*
 * Interface to CUDA. Real version (single or double precision).
 *
 * Due to transfer between main memory and GPU memory, user may have to call these directly.
 *
 * Only periodic boundaries are supported.
 *
 */

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
  case CUFFT_INVALID_PLAN:
    fprintf(stderr, "Invalid plan.\n");
    break;
  case CUFFT_ALLOC_FAILED:
    fprintf(stderr, "GPU memory allocation failed.\n");
    break;
  case CUFFT_INVALID_TYPE:
    fprintf(stderr, "Invalid parameter types (invalid type).\n");
    break;
  case CUFFT_INVALID_VALUE:
    fprintf(stderr, "Invalid parameter values (invalid value).\n");
    break;
  case CUFFT_INTERNAL_ERROR:
    fprintf(stderr, "Internal driver error.\n");
    break;
  case CUFFT_EXEC_FAILED:
    fprintf(stderr, "Exec failed.\n");
    break;
  case CUFFT_SETUP_FAILED:
    fprintf(stderr, "Library failed to initialize.\n");
    break;
  case CUFFT_INVALID_SIZE:
    fprintf(stderr, "Dimension of nx, ny, or nz not supported (invalid size).\n");
    break;
  case CUFFT_UNALIGNED_DATA:
    fprintf(stderr, "Unaligned data.\n");
    break;
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    fprintf(stderr, "Incomplete parameter list.\n");
    break;
  case CUFFT_PARSE_ERROR:
    fprintf(stderr, "Parse error.\n");
    break;
  case CUFFT_NO_WORKSPACE:
    fprintf(stderr, "No workspace.\n");
    break;
  case CUFFT_NOT_IMPLEMENTED:
    fprintf(stderr, "Not implemented.\n");
    break;
  case CUFFT_LICENSE_ERROR:
    fprintf(stderr, "License error.\n");
    break;
  case CUFFT_NOT_SUPPORTED:
    fprintf(stderr, "Not supported.\n");
    break;
  default:
    fprintf(stderr, "Unknown cufft error code.\n");    
  }
}

EXPORT void rgrid_cufft_alloc_r2c(rgrid *grid) {

  cufftResult status;
  size_t len[MAX_CUDA_DESCRIPTOR_GPUS];
  int *gpus = cuda_gpus(), ngpus = cuda_ngpus();

  if(cufftCreate(&(grid->cufft_handle_r2c)) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error creating r2c plan.\n");
    abort();
  }
  if(ngpus > 1) {
    if(cufftXtSetGPUs(grid->cufft_handle_r2c, ngpus, gpus) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error allocating GPUs in r2c.\n");
      abort();
    }    
  }
 #ifdef SINGLE_PREC
  if((status = cufftMakePlan3d(grid->cufft_handle_r2c, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_R2C, len)) != CUFFT_SUCCESS) {
#else /* double */
  if((status = cufftMakePlan3d(grid->cufft_handle_r2c, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_D2Z, len)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(cuda): Error in forward r2c rplan: ");
    error_check(status);
    abort();
  }
}

EXPORT void rgrid_cufft_alloc_c2r(rgrid *grid) {

  cufftResult status;
  size_t len[MAX_CUDA_DESCRIPTOR_GPUS];
  int *gpus = cuda_gpus(), ngpus = cuda_ngpus();

  if(cufftCreate(&(grid->cufft_handle_c2r)) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error creating c2r plan.\n");
    abort();
  }
  if(ngpus > 1) {
    if(cufftXtSetGPUs(grid->cufft_handle_c2r, ngpus, gpus) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error allocating GPUs in c2r.\n");
      abort();
    }    
  }
#ifdef SINGLE_PREC
  if((status = cufftMakePlan3d(grid->cufft_handle_c2r, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_C2R, len)) != CUFFT_SUCCESS) {
#else
  if((status = cufftMakePlan3d(grid->cufft_handle_c2r, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_Z2D, len)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(cuda): Error in backward rplan: ");
    error_check(status);
    abort();
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
  
  cufftResult status;
  gpu_mem_block *block;

  if(grid->cufft_handle_r2c == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (r2c).\n");
    abort();
  }

#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    if((status = cufftExecR2C(grid->cufft_handle_r2c, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
    error_check(status);
    abort();
  } else {
    if((status = cufftXtExecDescriptorR2C(grid->cufft_handle_r2c, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
    error_check(status);
    abort();
  }
#else
  if(cuda_ngpus() == 1) {
    if((status = cufftExecD2Z(grid->cufft_handle_r2c, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      error_check(status);
      abort();
    }
  } else {
      if((status = cufftXtExecDescriptorD2Z(grid->cufft_handle_r2c, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      error_check(status);
      abort();
    }
  }
#endif

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

  cufftResult status;
  gpu_mem_block *block;

  if(grid->cufft_handle_c2r == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle_c2r, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (c2r).\n");
    abort();
  }
#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    if((status = cufftExecC2R(grid->cufft_handle_c2r, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      error_check(status);
      abort();
    }
  } else {
    if((status = cufftXtExecDescriptorC2R(grid->cufft_handle_c2r, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      error_check(status);
      abort();
    }
  }
#else /* double */
  if(cuda_ngpus() == 1) {
    if((status = cufftExecZ2D(grid->cufft_handle_c2r, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      error_check(status);
      abort();
    }
  } else {
    if((status = cufftXtExecDescriptorZ2D(grid->cufft_handle_c2r, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      error_check(status);
      abort();
    }
  }
#endif

  cuda_error_check();

  return 0;
}
