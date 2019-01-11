/*
 * FFT Example.
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
REAL func(void *NA, REAL x, REAL y, REAL z) {

  return COS(x * 2.0 * M_PI * KX / (NX * STEP)) + COS(y * 2.0 * M_PI * KY / (NY * STEP)) + COS(z * 2.0 * M_PI * KZ / (NZ * STEP));
}

/* Routine to write grid on disk */
void write_grid(char *base, rgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x, y, z;

  /* Write binary grid */
  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid_write(grid, fp);
  fclose(fp);

  /* Write cut along x-axis */
  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  j = NY/2;
  k = NZ/2;
  for(i = 0; i < NX; i++) { 
    x = ((REAL) (i - NX/2)) * STEP;
    fprintf(fp, FMT_R " " FMT_R "\n", x, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  /* Write cut along y-axis */
  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  i = NX/2;
  k = NZ/2;
  for(j = 0; j < NY; j++) {
    y = ((REAL) (j - NY/2)) * STEP;
    fprintf(fp, FMT_R " " FMT_R "\n", y, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  /* Write cut along z-axis */
  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  i = NX/2;
  j = NY/2;
  for(k = 0; k < NZ; k++) {
    z = ((REAL) (k - NZ/2)) * STEP;
    fprintf(fp, FMT_R " " FMT_R "\n", z, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);
}

int main(int argc, char **argv) {

  rgrid *grid;           /* Pointer real grid structure */

  grid_threads_init(0);  /* Use all available cores */

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Allocate real grid with dimensions NX, NY, NZ and spatial step size STEP */
  /* Periodic boundary condition is assigned to the grid */
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "test grid");

  /* Map the standing wave function onto the grid */
  rgrid_map(grid, &func, NULL);

  /* Output the grid before FFT */
  write_grid("before", grid);

  /* Perform FFT */
  rgrid_fft(grid);

  /* Perform normalize inverse FFT */
  rgrid_inverse_fft_norm(grid);

  /* Write grid after forward & inverse FFTs (we should get the original grid) */
  write_grid("after", grid);

  /* If CUDA in use, output usage statistics */
#ifdef USE_CUDA
  cuda_statistics(1);
#endif

  return 0;
}
