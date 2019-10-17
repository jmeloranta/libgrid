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

/*
 * Propagate kinetic energy in Fourier space.
 *
 * Only periodic boundaries implemented.
 *
 */

EXPORT char grid_cuda_wf_propagate_kinetic_fft(wf *gwf, REAL complex time_mass) {

  CUCOMPLEX t;
  cgrid *grid = gwf->grid;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

  t.x = CREAL(time_mass);
  t.y = CIMAG(time_mass);

  grid_cuda_wf_propagate_kinetic_fftW(cuda_block_address(grid->value), grid->fft_norm, grid->kx0, grid->ky0, grid->kz0, grid->step, t, grid->nx, grid->ny, grid->nz);

 return 0;
}

/*
 * Propagate kinetic energy in Fourier space (with anti-aliasing).
 *
 * Only periodic boundaries implemented.
 *
 */

EXPORT char grid_cuda_wf_propagate_kinetic_cfft(wf *gwf, REAL complex time_mass, REAL cnorm) {

  CUCOMPLEX t;
  cgrid *grid = gwf->grid;

  if(grid->host_lock) {
    cuda_remove_block(grid->value, 1);
    return -1;
  }

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->cufft_handle, grid->id) < 0) return -1;

  t.x = CREAL(time_mass);
  t.y = CIMAG(time_mass);

  grid_cuda_wf_propagate_kinetic_cfftW(cuda_block_address(grid->value), grid->fft_norm, grid->kx0, grid->ky0, grid->kz0, grid->step, t, cnorm, grid->nx, grid->ny, grid->nz);

  return 0;
}
