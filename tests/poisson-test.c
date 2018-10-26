#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

double complex gaussian(void *arg, double x, double y, double z) {

  double inv_width = 0.2;
  double norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(x * x + y * y + z * z) * inv_width * inv_width);
}

#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.2

main() {
  
  cgrid *grid, *grid2;
  long i;
  
  grid_threads_init(16);
  grid = cgrid_alloc(NX, NY, NZ, STEP, CGRID3D_PERIODIC_BOUNDARY, NULL);
  grid2 = cgrid_alloc(NX, NY, NZ, STEP, CGRID3D_PERIODIC_BOUNDARY, NULL);
  cgrid_map(grid, gaussian, NULL);
  cgrid_fd_laplace(grid, grid2);
  cgrid_poisson(grid2);  
  for (i = 0; i < NZ; i++)
    printf("%le %le %le\n", (i-NZ/2)*1.0, creal(grid->value[NZ * NY * NX/2 + NZ * NY/2 + i]), creal(grid2->value[NZ * NY * NX/2 + NZ * NY/2 + i]));
}

