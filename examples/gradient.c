/*
 * Example: Calculate gradient by finite difference and FFT for comparison.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <omp.h>

/* Grid dimensions */
#define NX 128
#define NY 128
#define NZ 128

/* Spatial step length of the grid */
#define STEP 0.5

/* wave vectors top be mapped onto grid */
#define KX 1.0
#define KY 1.0
#define KZ 1.0

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {0};
#endif

/* Function returning standing wave in x, y, and z directions */
REAL func(void *NA, REAL x, REAL y, REAL z) {

  return COS(x * 2.0 * M_PI * KX / (NX * STEP)) + COS(y * 2.0 * M_PI * KY / (NY * STEP)) + COS(z * 2.0 * M_PI * KZ / (NZ * STEP));
}

int main(int argc, char **argv) {

  rgrid *grid;                     /* Pointer real grid structure */
  rgrid *grad_x, *grad_y, *grad_z;  /* Pointers for gradient components */

  grid_threads_init(0);  /* Use all available cores */
  grid_wf_analyze_method(1); // FFT:1 FD:0 

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Allocate real grid with dimensions NX, NY, NZ and spatial step size STEP */
  /* Periodic boundary condition is assigned to the grid */
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "test grid");
  grad_x = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "gradient x");
  grad_y = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "gradient y");
  grad_z = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "gradient z");

  /* Map the standing wave function onto the grid */
  rgrid_map(grid, &func, NULL);

  rgrid_gradient_x(grid, grad_x);
  rgrid_gradient_y(grid, grad_y);
  rgrid_gradient_z(grid, grad_z);

  /* Output the fd gradient */
  rgrid_write_grid("grad_x", grad_x);
  rgrid_write_grid("grad_y", grad_y);
  rgrid_write_grid("grad_z", grad_z);

  /* If CUDA in use, output usage statistics */
#ifdef USE_CUDA
  cuda_statistics(1);
#endif

  return 0;
}
