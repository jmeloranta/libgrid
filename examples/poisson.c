/*
 * Example: Solve Poisson equation.
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

/* Grid dimensions */
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.2

/* Right hand side function for Poisson equation (Gaussian) */
REAL gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = 0.2;
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(x * x + y * y + z * z) * inv_width * inv_width);
}

int main(int argc, char **argv) {
  
  rgrid *grid;
  FILE *fp;
  
  /* Initialize with 16 OpenMP threads */
  grid_threads_init(16);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1);
#endif
  
  /* Allocate real grid for the right hand side (and the solution) */
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "Poisson");

  /* Map the right hand side to the grid */
  rgrid_map(grid, gaussian, NULL);

  /* Solve the Poisson equation (result written over the right hand side in grid) */
  rgrid_poisson(grid);  

  /* Write output file */
  if(!(fp = fopen("output", "w"))) {
    fprintf(stderr, "Can't open file for writing.\n");
    exit(1);
  }
  rgrid_write(grid, fp);
  fclose(fp);
  return 0;
}

