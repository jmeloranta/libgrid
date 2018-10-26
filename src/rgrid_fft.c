/*
 * Interface to FFTW. Real version.
 *
 * No user callable routines.
 *
 * TODO: Somehow this only works for 3-D arrays where all dimensions
 * are multiples of two! (?) For example, 256, 256, 256 will work
 * fine but 255, 255, 255 does not. This is for periodic.
 *
 */

#include "grid.h"

/*
 * Allocate FFT buffers for a given grid.
 * This must be called before FFT can be carried out
 * (called automatically, users don't need to call this)
 * 
 * grid = grid for which the FFT allocation is to be done (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fftw_alloc(rgrid *grid) {

  size_t s;
  REAL *plan_temp;

  s = (size_t) (2 * grid->nx * grid->ny * (grid->nz/2 + 1));

#if defined(SINGLE_PREC)
  if(!(plan_temp = fftwf_malloc(sizeof(REAL) * s))) {
#elif defined(DOUBLE_PREC)
  if(!(plan_temp = fftw_malloc(sizeof(REAL) * s))) {
#elif defined(QUAD_PREC)
  if(!(plan_temp = fftwl_malloc(sizeof(REAL) * s))) {
#endif
    fprintf(stderr, "libgrid: Out of memory in rgrid_fft().\n");
    return;
  }

  memcpy(plan_temp, grid->value, sizeof(REAL) * s);
 
#if defined(SINGLE_PREC)
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->plan = 
    fftwf_plan_dft_r2c_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, grid->value, (REAL complex *) grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->iplan = 
    fftwf_plan_dft_c2r_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, (REAL complex *) grid->value, grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
#elif defined(DOUBLE_PREC)
  fftw_plan_with_nthreads((int) grid_threads());
  grid->plan = 
    fftw_plan_dft_r2c_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, grid->value, (REAL complex *) grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftw_plan_with_nthreads((int) grid_threads());
  grid->iplan = 
    fftw_plan_dft_c2r_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, (REAL complex *) grid->value, grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
#elif defined(QUAD_PREC)
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->plan = 
    fftwl_plan_dft_r2c_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, grid->value, (REAL complex *) grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->iplan = 
    fftwl_plan_dft_c2r_3d((int) grid->nx, (int) grid->ny, (int) grid->nz, (REAL complex *) grid->value, grid->value, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
#endif
  memcpy(grid->value, plan_temp, sizeof(REAL) * s);
  free(plan_temp);
}

/*
 * Free FFT plans. Used only internally.
 *
 * grid = grid for which the FFT buffers are to be freed (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fftw_free(rgrid *grid) {

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
}


EXPORT void rgrid_fftw(rgrid *grid) {

  if(!grid->plan) {
    if(grid_threads() == 0) {
      fprintf(stderr, "libgrid: Error in rgrid_fft(). Function grid_threads_init() must be called before this function in parallel programs.\n");
      abort();
    }
    rgrid_fftw_alloc(grid);
  }
#if defined(SINGLE_PREC)
  fftwf_execute(grid->plan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->plan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->plan);
#endif
}

EXPORT void rgrid_fftw_inv(rgrid *grid) {

  if (!grid->iplan) {
    if (grid_threads() == 0) {
      fprintf(stderr, "libgrid: Error in rgrid_inverse_fft(). Function grid_threads_init() must be called before this function in parallel programs.\n");
      abort();
    }
    rgrid_fftw_alloc(grid);
  }
#if defined(SINGLE_PREC)
  fftwf_execute(grid->iplan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->iplan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->iplan);
#endif
}
