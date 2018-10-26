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

  if(cuda_fft_policy(grid->value, grid->grid_len, grid->id) < 0) return -1;

  t.x = CREAL(time_mass);
  t.y = CIMAG(time_mass);
  cgrid_cufft_fft(grid);
  grid_cuda_wf_propagate_kinetic_fftW(cuda_block_address(grid->value), grid->fft_norm, grid->kx0, grid->ky0, grid->kz0, grid->step, t, grid->nx, grid->ny, grid->nz);
  cgrid_cufft_fft_inv(grid);
  return 0;
}
