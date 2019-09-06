/*
 * Example: Integrate over gaussian (complex version).
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

/* Equation to be integrated */
REAL complex gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = 0.2;
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(x * x + y * y + z * z) * inv_width * inv_width);
}

int main(int argc, char **argv) {
  
  cgrid *grid;
  REAL complex value;
  
  /* Initialize with all OpenMP threads */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, 0, NULL);
#endif
  
  /* Allocate real grid for the right hand side (and the solution) */
  grid = cgrid_alloc(NX, NY, NZ, STEP, CGRID_PERIODIC_BOUNDARY, NULL, "Poisson1");

  /* Map the right hand side to the grid */
  cgrid_map(grid, gaussian, NULL);

  value = cgrid_integral(grid);
  printf("Integral = (" FMT_R "," FMT_R ")\n", CREAL(value), CIMAG(value));

  return 0;
}

