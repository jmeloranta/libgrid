/*
 * Benchmark for individual libgrid routines.
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

/* Grid dimensions */
#define NX 512
#define NY 512
#define NZ 512
#define STEP 0.2

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {0};
#endif

/* Right hand side function for Poisson equation (Gaussian) */
REAL complex gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = 0.2;
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(x * x + y * y + z * z) * inv_width * inv_width);
}

int main(int argc, char **argv) {
  
  cgrid *grid;
  grid_timer timer;
  INT i;
  
  /* Initialize with all OpenMP threads */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif
  
  /* Allocate real grid for the right hand side (and the solution) */
  grid = cgrid_alloc(NX, NY, NZ, STEP, CGRID_PERIODIC_BOUNDARY, NULL, "Poisson1");

  /* Map the right hand side to the grid */
  cgrid_map(grid, gaussian, NULL);

  cgrid_fft(grid);
  printf("Start\n"); fflush(stdout);
  grid_timer_start(&timer);

  for(i = 0; i < 3000; i++)  // Run for 3000 times
    cgrid_fft_convolute(grid, grid, grid);

  printf("Wall clock time = " FMT_R " seconds.\n", grid_timer_wall_clock_time(&timer)); fflush(stdout);

  return 0;
}

