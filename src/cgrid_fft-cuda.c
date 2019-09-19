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

EXPORT void cgrid_cufft_alloc(cgrid *grid) {

#ifdef SINGLE_PREC
  grid_cufft_make_plan(&(grid->cufft_handle), CUFFT_C2C, grid->nx, grid->ny, grid->nz);
#else
  grid_cufft_make_plan(&(grid->cufft_handle), CUFFT_Z2Z, grid->nx, grid->ny, grid->nz);
#endif
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
  gpu_mem_block *block;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(grid->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (c2c).\n");
    abort();
  }

#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecC2C(grid->cufft_handle, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0], CUFFT_FORWARD)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (forward): ");
      cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorC2C(grid->cufft_handle, block->gpu_info, block->gpu_info, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (forward): ");
      cufft_error_check(status);
      return -1;
    }
  }
#else
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecZ2Z(grid->cufft_handle, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0], CUFFT_FORWARD)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (forward): ");
      cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorZ2Z(grid->cufft_handle, block->gpu_info, block->gpu_info, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (forward): ");
      cufft_error_check(status);
      return -1;
    }
  }
#endif

  cuda_error_check();

  return 0;
}

/*
 * Inverse FFT using cufft when data already in GPU memory. 
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
  gpu_mem_block *block;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(grid->cufft_handle == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (c2c).\n");
    abort();
  }

#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecC2C(grid->cufft_handle, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0], CUFFT_INVERSE)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (backward): ");
      cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorC2C(grid->cufft_handle, block->gpu_info, block->gpu_info, CUFFT_INVERSE)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (backward): ");
      cufft_error_check(status);
      return -1;
    }
  }
#else /* double */
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecZ2Z(grid->cufft_handle, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0], CUFFT_INVERSE)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (backward): ");
      cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorZ2Z(grid->cufft_handle, block->gpu_info, block->gpu_info, CUFFT_INVERSE)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(CUDA): Error in CFFT (backward): ");
      cufft_error_check(status);
      return -1;
    }
  }
#endif

  cuda_error_check();

  return 0;
}
