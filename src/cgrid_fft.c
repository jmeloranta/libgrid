/*
 * Interface to FFTW.
 *
 * No user callable routines.
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
  int n[3] = {(int) grid->nx, (int) grid->ny, (int) grid->nz};
#if defined(SINGLE_PREC)
  fftwf_r2r_kind rfk[3], rbk[3], ifk[3], ibk[3];
#elif defined(DOUBLE_PREC)
  fftw_r2r_kind rfk[3], rbk[3], ifk[3], ibk[3];
#elif defined(QUAD_PREC)
  fftwl_r2r_kind rfk[3], rbk[3], ifk[3], ibk[3];
#endif

#if defined(SINGLE_PREC)
  temp = fftwf_malloc(size);
#elif defined(DOUBLE_PREC)
  temp = fftw_malloc(size);
#elif defined(QUAD_PREC)
  temp = fftwl_malloc(size);
#endif

  memcpy(temp, grid->value, size);

  /* NOTE: To see which boundary condition applies, see if grid->implan (or grid->iimplan) is NULL */
  /* If grid->implan != NULL, this is special even/odd boundary; if it is NULL then the standard PERIODIC boundary */
  if(grid->value_outside == CGRID_PERIODIC_BOUNDARY) {
#if defined(SINGLE_PREC)
    fftwf_plan_with_nthreads((int) grid_threads());
    grid->plan = fftwf_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
    fftwf_plan_with_nthreads((int) grid_threads());
    grid->iplan = fftwf_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
    grid->implan = grid->iimplan = NULL;
    memcpy(grid->value, temp, size);
    fftwf_free(temp);
#elif defined(DOUBLE_PREC)
    fftw_plan_with_nthreads((int) grid_threads());
    grid->plan = fftw_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
    fftw_plan_with_nthreads((int) grid_threads());
    grid->iplan = fftw_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);


    grid->implan = grid->iimplan = NULL;
    memcpy(grid->value, temp, size);
    fftw_free(temp);
#elif defined(QUAD_PREC)
    fftwl_plan_with_nthreads((int) grid_threads());
    grid->plan = fftwl_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				  grid->value, grid->value, FFTW_FORWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
    fftwl_plan_with_nthreads((int) grid_threads());
    grid->iplan = fftwl_plan_dft_3d((int) grid->nx, (int) grid->ny, (int) grid->nz,
				   grid->value, grid->value, FFTW_BACKWARD, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
    grid->implan = grid->iimplan = NULL;
    memcpy(grid->value, temp, size);
    fftwl_free(temp);
#endif
    return;
  } else if(grid->value_outside == CGRID_FFT_EEE_BOUNDARY) {
    rfk[0] = FFTW_REDFT10; rfk[1] = FFTW_REDFT10; rfk[2] = FFTW_REDFT10;
    rbk[0] = FFTW_REDFT01; rbk[1] = FFTW_REDFT01; rbk[2] = FFTW_REDFT01;
    ifk[0] = FFTW_REDFT10; ifk[1] = FFTW_REDFT10; ifk[2] = FFTW_REDFT10;
    ibk[0] = FFTW_REDFT01; ibk[1] = FFTW_REDFT01; ibk[2] = FFTW_REDFT01;
  } else if (grid->value_outside == CGRID_FFT_OEE_BOUNDARY) {
    rfk[0] = FFTW_RODFT10; rfk[1] = FFTW_REDFT10; rfk[2] = FFTW_REDFT10;
    rbk[0] = FFTW_RODFT01; rbk[1] = FFTW_REDFT01; rbk[2] = FFTW_REDFT01;
    ifk[0] = FFTW_RODFT10; ifk[1] = FFTW_REDFT10; ifk[2] = FFTW_REDFT10;
    ibk[0] = FFTW_RODFT01; ibk[1] = FFTW_REDFT01; ibk[2] = FFTW_REDFT01;
  } else if (grid->value_outside == CGRID_FFT_EOE_BOUNDARY) {
    rfk[0] = FFTW_REDFT10; rfk[1] = FFTW_RODFT10; rfk[2] = FFTW_REDFT10;
    rbk[0] = FFTW_REDFT01; rbk[1] = FFTW_RODFT01; rbk[2] = FFTW_REDFT01;
    ifk[0] = FFTW_REDFT10; ifk[1] = FFTW_RODFT10; ifk[2] = FFTW_REDFT10;
    ibk[0] = FFTW_REDFT01; ibk[1] = FFTW_RODFT01; ibk[2] = FFTW_REDFT01;
  } else if (grid->value_outside == CGRID_FFT_EEO_BOUNDARY) {
    rfk[0] = FFTW_REDFT10; rfk[1] = FFTW_REDFT10; rfk[2] = FFTW_RODFT10;
    rbk[0] = FFTW_REDFT01; rbk[1] = FFTW_REDFT01; rbk[2] = FFTW_RODFT01;
    ifk[0] = FFTW_REDFT10; ifk[1] = FFTW_REDFT10; ifk[2] = FFTW_RODFT10;
    ibk[0] = FFTW_REDFT01; ibk[1] = FFTW_REDFT01; ibk[2] = FFTW_RODFT01;
  } else if (grid->value_outside == CGRID_FFT_OOE_BOUNDARY) {
    rfk[0] = FFTW_RODFT10; rfk[1] = FFTW_RODFT10; rfk[2] = FFTW_REDFT10;
    rbk[0] = FFTW_RODFT01; rbk[1] = FFTW_RODFT01; rbk[2] = FFTW_REDFT01;
    ifk[0] = FFTW_RODFT10; ifk[1] = FFTW_RODFT10; ifk[2] = FFTW_REDFT10;
    ibk[0] = FFTW_RODFT01; ibk[1] = FFTW_RODFT01; ibk[2] = FFTW_REDFT01;
  } else if (grid->value_outside == CGRID_FFT_EOO_BOUNDARY) {
    rfk[0] = FFTW_REDFT10; rfk[1] = FFTW_RODFT10; rfk[2] = FFTW_RODFT10;
    rbk[0] = FFTW_REDFT01; rbk[1] = FFTW_RODFT01; rbk[2] = FFTW_RODFT01;
    ifk[0] = FFTW_REDFT10; ifk[1] = FFTW_RODFT10; ifk[2] = FFTW_RODFT10;
    ibk[0] = FFTW_REDFT01; ibk[1] = FFTW_RODFT01; ibk[2] = FFTW_RODFT01;
  } else if (grid->value_outside == CGRID_FFT_OEO_BOUNDARY) {
    rfk[0] = FFTW_RODFT10; rfk[1] = FFTW_REDFT10; rfk[2] = FFTW_RODFT10;
    rbk[0] = FFTW_RODFT01; rbk[1] = FFTW_REDFT01; rbk[2] = FFTW_RODFT01;
    ifk[0] = FFTW_RODFT10; ifk[1] = FFTW_REDFT10; ifk[2] = FFTW_RODFT10;
    ibk[0] = FFTW_RODFT01; ibk[1] = FFTW_REDFT01; ibk[2] = FFTW_RODFT01;
  } else if (grid->value_outside == CGRID_FFT_OOO_BOUNDARY) {
    rfk[0] = FFTW_RODFT10; rfk[1] = FFTW_RODFT10; rfk[2] = FFTW_RODFT10;
    rbk[0] = FFTW_RODFT01; rbk[1] = FFTW_RODFT01; rbk[2] = FFTW_RODFT01;
    ifk[0] = FFTW_RODFT10; ifk[1] = FFTW_RODFT10; ifk[2] = FFTW_RODFT10;
    ibk[0] = FFTW_RODFT01; ibk[1] = FFTW_RODFT01; ibk[2] = FFTW_RODFT01;
  } else {
    fprintf(stderr, "libgrid: Incompatible boundary condition for FFT.\n");
    exit(1);
  }
#if defined(SINGLE_PREC)
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->plan = fftwf_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rfk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftwf_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rbk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->implan = fftwf_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ifk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwf_plan_with_nthreads((int) grid_threads());
  grid->iimplan = fftwf_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ibk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  memcpy(grid->value, temp, size);
  fftwf_free(temp);
#elif defined(DOUBLE_PREC)
  fftw_plan_with_nthreads((int) grid_threads());
  grid->plan = fftw_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rfk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftw_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftw_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rbk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftw_plan_with_nthreads((int) grid_threads());
  grid->implan = fftw_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ifk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftw_plan_with_nthreads((int) grid_threads());
  grid->iimplan = fftw_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ibk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  memcpy(grid->value, temp, size);
  fftw_free(temp);
#elif defined(QUAD_PREC)
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->plan = fftwl_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rfk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->iplan = fftwl_plan_many_r2r(3, n, 1, (REAL *) grid->value, n, 2, 0, (REAL *) grid->value, n, 2, 0, rbk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->implan = fftwl_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ifk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
  fftwl_plan_with_nthreads((int) grid_threads());
  grid->iimplan = fftwl_plan_many_r2r(3, n, 1, ((REAL *) grid->value) + 1, n, 2, 0, ((REAL *) grid->value) + 1, n, 2, 0, ibk, ((unsigned int) grid_get_fftw_flags()) | FFTW_DESTROY_INPUT);
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
  if(grid->implan) fftwf_destroy_plan(grid->implan);
  if(grid->iimplan) fftwf_destroy_plan(grid->iimplan);
#elif defined(DOUBLE_PREC)
  if(grid->plan) fftw_destroy_plan(grid->plan);
  if(grid->iplan) fftw_destroy_plan(grid->iplan);
  if(grid->implan) fftw_destroy_plan(grid->implan);
  if(grid->iimplan) fftw_destroy_plan(grid->iimplan);
#elif defined(QUAD_PREC)
  if(grid->plan) fftwl_destroy_plan(grid->plan);
  if(grid->iplan) fftwl_destroy_plan(grid->iplan);
  if(grid->implan) fftwl_destroy_plan(grid->implan);
  if(grid->iimplan) fftwl_destroy_plan(grid->iimplan);
#endif
  grid->plan = grid->iplan = grid->implan = grid->iimplan = NULL;
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
  if(grid->implan) fftwf_execute(grid->implan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->plan);
  if(grid->implan) fftw_execute(grid->implan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->plan);
  if(grid->implan) fftwl_execute(grid->implan);
#endif
#ifdef USE_CUDA
  grid->space = 1;
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
  if(grid->iimplan) fftwf_execute(grid->iimplan);
#elif defined(DOUBLE_PREC)
  fftw_execute(grid->iplan);
  if(grid->iimplan) fftw_execute(grid->iimplan);
#elif defined(QUAD_PREC)
  fftwl_execute(grid->iplan);
  if(grid->iimplan) fftwl_execute(grid->iimplan);
#endif
#ifdef USE_CUDA
  grid->space = 0;
#endif
}
