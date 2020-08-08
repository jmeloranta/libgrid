/*
 * Interface to FFTW.
 *
 * These are not included in the manual since it should not be necessary
 * to call these routines directly.
 *
 */

#include "grid.h"

/*
 * Allocate FFT buffers for a given grid.
 * This must be called before FFT can be carried out
 * (called automatically, users don't need to call this)
 * 
 * grid = grid for which the FFT allocation is to be done (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fftw_alloc(cgrid *grid) {

  REAL complex *temp;
  size_t size = sizeof(REAL complex) * ((size_t) (grid->nx * grid->ny * grid->nz));

#if defined(SINGLE_PREC)
  temp = fftwf_malloc(size);
#elif defined(DOUBLE_PREC)
  temp = fftw_malloc(size);
#elif defined(QUAD_PREC)
  temp = fftwl_malloc(size);
#endif

  memcpy(temp, grid->value, size);

#if defined(SINGLE_PREC)
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->plan = fftwf_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftwf_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  memcpy(grid->value, temp, size);
  fftwf_free(temp);
#elif defined(DOUBLE_PREC)
  fftw_plan_with_nthreads((int) grid_threads());
  grid->plan = fftw_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftw_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftw_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);

  memcpy(grid->value, temp, size);
  fftw_free(temp);
#elif defined(QUAD_PREC)
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->plan = fftwl_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftwl_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
		   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  memcpy(grid->value, temp, size);
  fftwl_free(temp);
#endif
}

/*
 * Free FFT buffers. Used only internally.
 *
 * grid = grid for which the FFT buffers are to be freed (cgrid3d *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fftw_free(cgrid *grid) {

#if defined(SINGLE_PREC)
  if(grid->plan) fftwf_destroy_plan(grid->plan);
  if(grid->iplan) fftwf_destroy_plan(grid->iplan);
#elif defined(DOUBLE_PREC)
  if(grid->plan) fftw_destroy_plan(grid->plan);
  if(grid->iplan) fftw_destroy_plan(grid->iplan);
#elif defined(QUAD_PREC)
  if(grid->plan) fftwl_destroy_plan(grid->plan);
  if(grid->iplan) fftwl_destroy_plan(grid->iplan);
#endif
  grid->plan = grid->iplan = NULL;
}

/*
 * Forward FFT using FFTW. Used only internally.
 *
 * grid = grid to be transformed (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fftw(cgrid *grid) {

  if(!grid->plan) {
    if(grid_threads() == 0) {
      fprintf(stderr, "libgrid: Error in cgrid_fft(). Function grid_threads_init() must be called before this function in parallel programs.\n");
      abort();
    }
    cgrid_fftw_alloc(grid);
  }

#if defined(SINGLE_PREC)
  fftwf_execute(grid->plan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->plan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->plan);
#endif
}

/*
 * Backward FFT using FFTW. Used only internally.
 *
 * grid = grid to be transformed (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fftw_inv(cgrid *grid) {

  if(!grid->iplan) {
    if(grid_threads() == 0) {
      fprintf(stderr, "libgrid: Error in cgrid_inverse_fft(). Function grid_threads_init() must be called before this function in parallel programs.\n");
      abort();
    }
    cgrid_fftw_alloc(grid);
  }

#if defined(SINGLE_PREC)
  fftwf_execute(grid->iplan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->iplan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->iplan);
#endif
}
