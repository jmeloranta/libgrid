/*
 * Example: Calculate spherical average of a real grid.
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
#define STEP 0.5

/* Binning info */
#define BINSTEP 0.5
#define NBINS (NX / 2)
#define VOLEL 0   /* 0 = Calculate spherical average, 1 = direct sum */

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {0};
#endif

/* Function to be mapped onto the grid */
REAL func(void *NA, REAL x, REAL y, REAL z) {

  return x * x + y * y + z * z;
}

int main(int argc, char **argv) {

  rgrid *grid;
  REAL *bins;
  FILE *fp;
  INT i;
 
  /* Initialize threads (0 = use all threads available) */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif
  
  /* Allocate real grid */
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "test grid");

  /* Map function func() onto the grid */
  rgrid_map(grid, &func, NULL);

  /* Write the data on disk before starting */
  rgrid_write_grid("before", grid);

  /* Allocate memory for the bins */
  bins = (REAL *) malloc(sizeof(REAL) * NBINS);

  /* Perform spherical average of the grid */
  rgrid_spherical_average(grid, NULL, NULL, bins, BINSTEP, NBINS, VOLEL);

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
