/*
 * Example: Perform FFT on a complex grid.
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

/* wave vectors top be mapped onto grid */
#define KX 1.0
#define KY 1.0
#define KZ 1.0

/* Function returning standing wave in x, y, and z directions */
REAL complex func(void *NA, REAL x, REAL y, REAL z) {

  return COS(x * 2.0 * M_PI * KX / (NX * STEP)) + COS(y * 2.0 * M_PI * KY / (NY * STEP)) + COS(z * 2.0 * M_PI * KZ / (NZ * STEP));
}

int main(int argc, char **argv) {

  cgrid *grid;           /* Pointer real grid structure */

  grid_threads_init(0);  /* Use all available cores */

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, 0, NULL);
#endif

  /* Allocate real grid with dimensions NX, NY, NZ and spatial step size STEP */
  /* Periodic boundary condition is assigned to the grid */
  grid = cgrid_alloc(NX, NY, NZ, STEP, CGRID_PERIODIC_BOUNDARY, NULL, "test grid");

  /* Map the standing wave function onto the grid */
  cgrid_map(grid, &func, NULL);

  /* Output the grid before FFT */
  cgrid_write_grid("before", grid);

  /* Perform FFT */
  cgrid_fft(grid);

  /* Perform normalize inverse FFT */
  cgrid_inverse_fft_norm(grid);

  /* Write grid after forward & inverse FFTs (we should get the original grid) */
  cgrid_write_grid("after", grid);

  /* If CUDA in use, output usage statistics */
#ifdef USE_CUDA
  cuda_statistics(1);
#endif

  return 0;
}
