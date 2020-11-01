/*
 * Example: Calculate spherical average of a |complex grid|^2 in Fourier space.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <omp.h>

/* Grid dimensions */
#define NX 256
#define NY 256
#define NZ 256

/* Spatial step length of the grid */
#define STEP 0.1

/* Binning info */
#define BINSTEP (2.0 * M_PI / (NX * STEP))
#define NBINS 256
#define VOLEL 0  /* 0 = Calculate spherical average, 1 = direct sum */

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {0};
#endif

//#undef USE_CUDA

/* Function to be mapped onto the grid */
REAL complex func(void *NA, REAL x, REAL y, REAL z) {

  return EXP(-0.1*(x * x + y * y + z * z));
}

int main(int argc, char **argv) {

  cgrid *grid, *wrk;
  REAL *bins;
  FILE *fp;
  INT i;
 
  /* Initialize threads (0 = use all threads available) */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif
  
  /* Allocate complex grid */
  grid = cgrid_alloc(NX, NY, NZ, STEP, CGRID_PERIODIC_BOUNDARY, NULL, "test grid");
  wrk = cgrid_clone(grid, "wrk");

  /* Map function func() onto the grid */
  cgrid_map(grid, &func, NULL);

  cgrid_abs_power(wrk, grid, 2.0);
  cgrid_write_grid("before", wrk);  

//  printf("Integral before = " FMT_R "\n", cgrid_integral_of_square(grid));

  /* FFT */
  cgrid_fft(grid);

  /* Allocate memory for the bins */
  bins = (REAL *) malloc(sizeof(REAL) * NBINS);

  /* Perform spherical average of |grid|^2 */
  cgrid_spherical_average_reciprocal(grid, NULL, NULL, bins, BINSTEP, NBINS, VOLEL);

  /* Write spherical average to disk */
  if(!(fp = fopen("after.dat", "w"))) {
    fprintf(stderr, "Can't open file for writing.\n");
    exit(1);
  }

  for (i = 0; i < NBINS; i++)
    fprintf(fp, FMT_R " " FMT_R "\n", BINSTEP * (REAL) i, bins[i]);
  fclose(fp);

  return 0;
}
