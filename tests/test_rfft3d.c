/*
 * Test program for real FFT routines (both FFTW and CUFFT).
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <omp.h>

#define NX 16
#define NY 16
#define NZ 16

#define STEP 0.5

#define KX 1.0
#define KY 1.0
#define KZ 1.0

REAL func(void *NA, REAL x, REAL y, REAL z) {

  return COS(x * 2.0 * M_PI * KX / (NX * STEP)) + COS(y * 2.0 * M_PI * KY / (NY * STEP)) + COS(z * 2.0 * M_PI * KZ / (NZ * STEP));
}

void write_grid(char *base, rgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x, y, z;

#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid_write(grid, fp);
  fclose(fp);

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

  rgrid *grid;

  grid_threads_init(1);
#ifdef USE_CUDA
  cuda_enable(1);
#endif
  grid = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "grid");
  rgrid_map(grid, &func, NULL);

  write_grid("before", grid);
  rgrid_fft(grid);
  rgrid_inverse_fft_norm(grid);
  write_grid("after", grid);
  cuda_statistics(1);

  return 0;
}
