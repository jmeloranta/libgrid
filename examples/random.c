/*
 * Example: Generate random numbers (uniform and normal distribution).
 *
 * Output: normal.* and uniform.*   (real grids)
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

/* Grid dimensions */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5

int main(int argc, char **argv) {
  
  rgrid *ngrid, *ugrid;
  
  /* Initialize with all available cores */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1);
#endif
  
  /* Allocate real grid for the right hand side (and the solution) */
  ugrid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "Uniform");
  ngrid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "Normal");

  rgrid_zero(ugrid);
  rgrid_zero(ngrid);

  /* Add random noise */
  rgrid_random(ugrid, 1.0);
  rgrid_random_normal(ngrid, 1.0);

  rgrid_write_grid("uniform", ugrid);
  rgrid_write_grid("normal", ngrid);

  return 0;
}
