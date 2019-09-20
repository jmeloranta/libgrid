/*
 * Threading functions.
 *
 */

#include "grid.h"
#include "private.h"
#include "git-version.h"

static INT grid_nthreads = 0;
static unsigned int plan_flags = FFTW_MEASURE;

#ifndef _OPENMP
int omp_get_max_threads() { return 1;}
int omp_get_thread_num() {return 1;}
void omp_set_num_threads(int threads) {}
int omp_get_num_threads() { return 1;}
void ompc_set_dynamic(int asd) {}
void omp_set_dynamic(int asd) {}
#endif

#ifdef __GNUC__
#include <math.h>
#include <fenv.h>
extern int fegetexcept();
extern int feenableexcept(int);
#endif

/*
 * FFTW specific flags for planning. Default is MEASURE.
 *
 * f = Which planning mode to use (ESTIMATE & MEASURE are fast whereas PATIENT & EXHAUSTIVE are really slow).
 *     However, best overall performance is obtained with the latter two...
 *
 * No return value.
 *
 */

EXPORT void grid_set_fftw_flags(char f) {

  switch (f) {
  case 0:
    plan_flags = FFTW_ESTIMATE;
    break;
  case 1:
    plan_flags = FFTW_MEASURE; /* default */
    break;
  case 2:
    plan_flags = FFTW_PATIENT;
    break;
  case 3:
    plan_flags = FFTW_EXHAUSTIVE;
    break;
  default:
    fprintf(stderr, "libgrid: Unknown FFT flags.\n");
    abort();
  }
}

/*
 * Get FFTW flags for planning.
 *
 * Returns the current planning mode (see above).
 *
 */

EXPORT char grid_get_fftw_flags() {

  switch (plan_flags) {
  case FFTW_ESTIMATE:
    return 0;
  case FFTW_MEASURE: /* default */
    return 1;
  case FFTW_PATIENT:
    return 2;
  case FFTW_EXHAUSTIVE:
    return 3;
  default:
    fprintf(stderr, "libgrid: Unknown FFT flags.\n");
    abort();
  }
}

/*
 * Initialize threads. This must be called before any threaded routines.
 *
 * threads = Number of threads to be used in the calculations (INT).
 *
 * No return value.
 *
 */

EXPORT void grid_threads_init(INT threads) {

  if (threads < 0) {
    fprintf(stderr, "libgrid: Error in grid_threads_init(). Number of threads <= 0.\n");
    abort();
  }

  fprintf(stderr, "libgrid: GIT version ID %s\n", VERSION);

#if defined(SINGLE_PREC)
  fprintf(stderr, "libgrid: Single precision floats.\n");
#elif defined(DOUBLE_PREC)
  fprintf(stderr, "libgrid: Double precision floats.\n");
#elif defined(QUAD_PREC)
  fprintf(stderr, "libgrid: Quad precision floats.\n");
#endif

#ifdef SHORT_INT
  fprintf(stderr, "libgrid: Short integers.\n");
#else
  fprintf(stderr, "libgrid: Long integers.\n");
#endif

  if(threads == 0) {
    threads = (INT) sysconf(_SC_NPROCESSORS_ONLN);
    fprintf(stderr, "libgrid: Allocating all CPUs (" FMT_I ").\n", threads);
  }
  
  if (grid_nthreads && grid_nthreads != threads) {
    fprintf(stderr, "libgrid: Error in grid_threads_init(). This function was called twice with different number of threads.\n");
    abort();	
  }

#ifdef __GNUC__
  /* core dump on floating point exceptions */
  //feenableexcept(fegetexcept()|FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW);
  feenableexcept(fegetexcept()|FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);   // Do not attempt to catch underflow - this is just zero to us...
#endif
  
  /* Without openmp, use just one thread for everything */
#ifdef _OPENMP
  grid_nthreads = threads;
#else
  grid_nthreads = 1;
#endif
  omp_set_num_threads((int) threads);
#if defined(SINGLE_PREC)
  fftwf_init_threads();
#elif defined(DOUBLE_PREC)
  fftw_init_threads();
#elif defined(QUAD_PREC)
  fftwl_init_threads();
#endif
  fprintf(stderr, "libgrid: Initialized with " FMT_I " threads.\n", threads);
  fprintf(stderr, "libgrid: omp_num_threads = %d, omp_max_num_threads = %d.\n",
	  omp_get_num_threads(), omp_get_max_threads());
}

/*
 * Return the max number of threads available.
 *
 */

EXPORT INT grid_threads() {

  return grid_nthreads;
}
