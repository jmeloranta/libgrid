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

  cufftResult status;
  size_t len[MAX_CUDA_DESCRIPTOR_GPUS];
  int *gpus = cuda_gpus(), ngpus = cuda_ngpus();

  if((status = cufftCreate(&(grid->cufft_handle))) != CUFFT_SUCCESS) {
    fprintf(stderr, "libgrid(cuda): Error creating c2c plan.\n");
    cufft_error_check(status);
    return;
  }
  if(ngpus > 1) {
    if((status = cufftXtSetGPUs(grid->cufft_handle, ngpus, gpus)) != CUFFT_SUCCESS) {
      fprintf(stderr, "libgrid(cuda): Error allocating GPUs in c2c.\n");
      cufft_error_check(status);
      return;
    }
  }
#ifdef SINGLE_PREC
  if((status = cufftMakePlan3d(grid->cufft_handle, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_C2C, len)) != CUFFT_SUCCESS) {
#else /* double */
  if((status = cufftMakePlan3d(grid->cufft_handle, (int) grid->nx, (int) grid->ny, (int) grid->nz, CUFFT_Z2Z, len)) != CUFFT_SUCCESS) {
#endif
    fprintf(stderr, "libgrid(CUDA): Error in forward cplan: ");
    cufft_error_check(status);
    return;
  }

  cuda_error_check(status);
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
  grid->space = 1; // Data is now in Fourier space

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
  grid->space = 0; // Data is now in real space

  return 0;
}
