/*
 * Example: Create a plane wave and compute the compressible & incompressible decomposition.
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>

/* Grid dimensions */
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.2

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define KX (2.0 * 2.0 * M_PI / (NX * STEP))
#define KY (0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ (0.0 * 2.0 * M_PI / (NZ * STEP))
#define AMP 1E-3

#define NBINS 128
#define BINSTEP 0.01

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 1
int gpus[NGPUS] = {3};
#endif

EXPORT REAL complex planewave(void *arg, REAL x, REAL y, REAL z) {

  return SQRT(RHO0) + AMP * CEXP(I * (KX * x + KY * y + KZ * z));
}

int main(int argc, char **argv) {
  
  wf *gwf;
  rgrid *wrk1, *wrk2, *wrk3, *wrk4, *wrk5;
  REAL bins[NBINS];
  INT i;
  FILE *fp;
  
  /* Initialize with all OpenMP threads */
  grid_threads_init(0);

  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif
  
  printf("KX = " FMT_R " Bohr^-1 KY = " FMT_R " Bohr^-1 KZ = " FMT_R " Bohr^-1.\n", KX, KY, KZ);

  /* Allocate real grid for the right hand side (and the solution) */
  gwf = grid_wf_alloc(NX, NY, NZ, STEP, 1.0, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "wavefunction");

  /* Map the plane wave to the grid (including liquid background) */
  grid_wf_map(gwf, planewave, NULL);

  /* Allocate (real) workspaces */
  wrk1 = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "wrk1");
  wrk2 = rgrid_clone(wrk1, "wrk2");
  wrk3 = rgrid_clone(wrk1, "wrk3");
  wrk4 = rgrid_clone(wrk1, "wrk4");
  wrk5 = rgrid_clone(wrk1, "wrk5");

  /* Compute the incompressible part of K.E. & write output to file */
  grid_wf_incomp_KE(gwf, bins, BINSTEP, NBINS, wrk1, wrk2, wrk3, wrk4, wrk5, 1E-5);
  if(!(fp = fopen("incomp.dat", "w"))) {
    fprintf(stderr, "can't open incomp.dat\n");
    exit(1);
  }
  for(i = 0; i < NBINS; i++) fprintf(fp, FMT_R " " FMT_R "\n", BINSTEP * (REAL) i, bins[i]);
  fclose(fp);  

  /* Compute the compressible part of K.E. & write output to file */
  grid_wf_comp_KE(gwf, bins, BINSTEP, NBINS, wrk1, wrk2, wrk3, wrk4, 1E-5);
  if(!(fp = fopen("comp.dat", "w"))) {
    fprintf(stderr, "can't open incomp.dat\n");
    exit(1);
  }
  for(i = 0; i < NBINS; i++) fprintf(fp, FMT_R " " FMT_R "\n", BINSTEP * (REAL) i, bins[i]);
  fclose(fp);

  return 0;
}
