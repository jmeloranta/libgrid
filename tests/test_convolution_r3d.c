/*
 * Test program for real FFT based convolution (both FFTW and CUFFT).
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

REAL func1(void *NA, REAL x, REAL y, REAL z) {

  return 1.0;
}

REAL func2(void *NA, REAL x, REAL y, REAL z) {

  return 1.0;
}


void write_grid(char *base, rgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x, y, z;

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

  rgrid *grid1, *grid2;

  grid_threads_init(1);
#ifdef USE_CUDA
  cuda_enable(1);
#endif
  grid1 = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "grid1");
  grid2 = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "grid2");
  rgrid_map(grid1, &func1, NULL);
  rgrid_map(grid2, &func2, NULL);

  rgrid_fft_convolute(grid1, grid1, grid2);

  write_grid("after", grid1);

  return 0;
}
