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

EXPORT void rgrid_cufft_alloc_r2c(rgrid *grid) {

#ifdef SINGLE_PREC
  grid_cufft_make_plan(&(grid->cufft_handle_r2c), CUFFT_R2C, grid->nx, grid->ny, grid->nz);
#else
  grid_cufft_make_plan(&(grid->cufft_handle_r2c), CUFFT_D2Z, grid->nx, grid->ny, grid->nz);
#endif
}

EXPORT void rgrid_cufft_alloc_c2r(rgrid *grid) {

#ifdef SINGLE_PREC
  grid_cufft_make_plan(&(grid->cufft_handle_c2r), CUFFT_C2R, grid->nx, grid->ny, grid->nz);
#else
  grid_cufft_make_plan(&(grid->cufft_handle_c2r), CUFFT_Z2D, grid->nx, grid->ny, grid->nz);
#endif
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

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(grid->cufft_handle_r2c == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle_r2c, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (r2c).\n");
    abort();
  }

  if(block->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) fprintf(stderr, "libgrid(cuda): FFT with wrong subformat.\n");

#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecR2C(grid->cufft_handle_r2c, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      cuda_cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorR2C(grid->cufft_handle_r2c, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      cuda_cufft_error_check(status);
      return -1;
    }
  }
#else
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecD2Z(grid->cufft_handle_r2c, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      cuda_cufft_error_check(status);
      return -1;
    }
  } else {
    if((status = cufftXtExecDescriptorD2Z(grid->cufft_handle_r2c, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (forward): ");
      cuda_cufft_error_check(status);
      return -1;
    }
  }
#endif

  block->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
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

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(grid->cufft_handle_c2r == -1) {
    fprintf(stderr, "libgrid(cuda): cufft not initialized.\n");
    abort();
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle_c2r, grid->id) < 0) return -1;

  if(!(block = cuda_find_block(grid->value))) {
    fprintf(stderr, "libgrid(cuda): Block not on GPU (c2r).\n");
    abort();
  }

  if(block->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) fprintf(stderr, "libgrid(cuda): IFFT with wrong subformat.\n");

#ifdef SINGLE_PREC
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecC2R(grid->cufft_handle_c2r, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      cuda_cufft_error_check(status);
      abort();
    }
  } else {
    if((status = cufftXtExecDescriptorC2R(grid->cufft_handle_c2r, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      cuda_cufft_error_check(status);
      abort();
    }
  }
#else /* double */
  if(cuda_ngpus() == 1) {
    cudaSetDevice(block->gpu_info->descriptor->GPUs[0]);
    if((status = cufftExecZ2D(grid->cufft_handle_c2r, block->gpu_info->descriptor->data[0], block->gpu_info->descriptor->data[0])) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      cuda_cufft_error_check(status);
      abort();
    }
  } else {
    if((status = cufftXtExecDescriptorZ2D(grid->cufft_handle_c2r, block->gpu_info, block->gpu_info)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error in FFT (inverse): ");
      cuda_cufft_error_check(status);
      abort();
    }
  }
#endif

  block->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  cuda_error_check();

  return 0;
}
