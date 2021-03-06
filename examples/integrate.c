/*
 * Example: Integrate over gaussian.
 *
 * The result should be equal to 1.
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

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {0};
#endif

/* Equation to be integrated */
REAL gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = 0.2;
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * EXP(-(x * x + y * y + z * z) * inv_width * inv_width);
}

int main(int argc, char **argv) {
  
  rgrid *grid;
  
  /* Initialize with all OpenMP threads */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif
  
  /* Allocate real grid for the right hand side (and the solution) */
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "Poisson1");

  /* Map the right hand side to the grid */
  rgrid_map(grid, gaussian, NULL);

  printf("Integral = " FMT_R "\n", rgrid_integral(grid));

  return 0;
}

