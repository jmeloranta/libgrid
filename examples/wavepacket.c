/*
 * Example: Propagate wavepacket in harmonic potential (3D).
 *
 * Try for example:
 * ./wavepacket 0 128 0.01 200 0.0 0.0 0.0 0.25 0.25 0.25 -2.0 0.0 0.0
 *
 * Although this is 3-D calculation, the above settings will initiate
 * the motion along the x-axis. Therefore it can be visualized by:
 *
 * gview2 output-{?,??,???}.x
 * 
 * (the wildcard syntax is likely different for bash; the above is tcsh)
 *
 * Arguments:
 * 1st  = Number of threads to be used.
 * 2nd  = Number of points in X, Y, Z directions (N x N x N grid).
 * 3rd  = Time step (atomic units).
 * 4th  = Number of iterations to run.
 * 5th  = Initial wave vector (momentum) along X (atomic units).
 * 6th  = Initial wave vector (momentum) along Y (atomic units).
 * 7th  = Initial wave vector (momentum) along Z (atomic units).
 * 8th  = Initial wave packet width in X direction (atomic units).
 * 9th  = Initial wave packet width in Y direction (atomic units).
 * 10th = Initial wave packet width in Z direction (atomic units).
 * 11th = Initial wave packet center along X direction (atomic units).
 * 12th = Initial wave packet center along Y direction (atomic units).
 * 13th = Initial wave packet center along Z direction (atomic units).
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <omp.h>

/* Define this for 4th order accuracy in time */
/* Otherwise for 2nd order accuracy */
//#define FOURTH_ORDER_FFT

/* If using CUDA, use the following GPU allocation */
#ifdef USE_CUDA
#define NGPUS 2
int gpus[NGPUS] = {1, 2};
#endif

REAL complex wavepacket(void *arg, REAL x, REAL y, REAL z);
REAL complex harmonic(void *arg, REAL x, REAL y, REAL z);

/* Wave packet structure */
typedef struct wparams_struct {
  REAL kx, ky, kz;
  REAL wx, wy, wz;
  REAL xc, yc, zc;
} wparams;

/* Harmonic potential parameters */
typedef struct pparams_struct {
  REAL kx, ky, kz;
} pparams;

int main(int argc, char *argv[]) {

  INT l, n, iterations, threads;
  REAL step, lx, time_step;
  REAL complex time;
  wf *gwf = NULL;
  cgrid *potential;
  rgrid *rworkspace;
  char fname[256];
  pparams potential_params;
  wparams wp_params;
  
  /* Parameter check */
  if (argc != 14) {
    fprintf(stderr, "Usage: wavepacket <thr> <npts> <tstep> <iters> <kx> <ky> <kz> <wx> <wy> <wz> <xc> <yc> <zc>\n");
    return -1;
  }
  
  /* Parse command line arguments */
  threads = atoi(argv[1]);
  n = atoi(argv[2]);
  time_step = (REAL) atof(argv[3]);
  iterations = atol(argv[4]);
  wp_params.kx = (REAL) atof(argv[5]);
  wp_params.ky = (REAL) atof(argv[6]);
  wp_params.kz = (REAL) atof(argv[7]);
  wp_params.wx = (REAL) atof(argv[8]);
  wp_params.wy = (REAL) atof(argv[9]);
  wp_params.wz = (REAL) atof(argv[10]);
  wp_params.xc = (REAL) atof(argv[11]);
  wp_params.yc = (REAL) atof(argv[12]);
  wp_params.zc = (REAL) atof(argv[13]);

  if(wp_params.wx == 0.0 || wp_params.wy == 0.0 || wp_params.wz == 0.0) {
    fprintf(stderr, "Width cannot be zero.\n");
    exit(1);
  }
  
  /* Set spatial grid step length based on number of grid points */
  step = 0.4 / (((REAL) n) / 16.0);
  
  fprintf(stderr, "Grid (" FMT_I "X" FMT_I "X" FMT_I ")\n", n, n, n);
  
  /* Potential parameters */
  lx = ((REAL) n) * step;
  /* Force constants for the harmonic potential */
  potential_params.kx = lx * 2.0;
  potential_params.ky = lx * 2.0;
  potential_params.kz = lx * 2.0;
  
  /* Initialize OpenMP threads */
  grid_threads_init(threads);
  
  /* If libgrid was compiled with CUDA support, enable CUDA */
#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif

  /* allocate memory (mass = 1.0) */
  gwf = grid_wf_alloc(n, n, n, step, 1.0, WF_PERIODIC_BOUNDARY, 
#ifdef FOURTH_ORDER_FFT
                      WF_4TH_ORDER_FFT, "WF");
#else
                      WF_2ND_ORDER_FFT, "WF");
#endif
  potential = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "potential");
  rworkspace = rgrid_alloc(n, n, n, step, RGRID_PERIODIC_BOUNDARY, 0, "rworkspace");
  
  /* Initialize wave function */
  grid_wf_map(gwf, wavepacket, &wp_params);

  grid_wf_normalize(gwf);
  
  /* Map potential */
  cgrid_smooth_map(potential, harmonic, &potential_params, 1);
  
  /* Propagate */
  time = time_step;
  for(l = 0; l < iterations; l++) {
    printf("Iteration " FMT_I " with wf norm = " FMT_R "\n", l, grid_wf_norm(gwf));
    /* Write |psi|^2 to output-* files */
    grid_wf_density(gwf, rworkspace);
    sprintf(fname, "output-" FMT_I, l);
    rgrid_write_grid(fname, rworkspace);
    /* Propagate one time step */
    grid_wf_propagate(gwf, potential, time);
  }

  /* Release resources */
  grid_wf_free(gwf);
  rgrid_free(rworkspace);
  cgrid_free(potential);
  
  return 0;
}

/* Function for creating the initial wave packet */
REAL complex wavepacket(void *arg, REAL x, REAL y, REAL z) {

  REAL kx = ((wparams *) arg)->kx;
  REAL ky = ((wparams *) arg)->ky;
  REAL kz = ((wparams *) arg)->kz;
  REAL wx = ((wparams *) arg)->wx;
  REAL wy = ((wparams *) arg)->wy;
  REAL wz = ((wparams *) arg)->wz;
  REAL xc = ((wparams *) arg)->xc;
  REAL yc = ((wparams *) arg)->yc;
  REAL zc = ((wparams *) arg)->zc;
  REAL x2, y2, z2;

  x -= xc;
  y -= yc;
  z -= zc;
  x2 = x / wx; x2 *= x2;
  y2 = y / wy; y2 *= y2;
  z2 = z / wz; z2 *= z2;

  return CEXP(- x2 + I * kx * x - y2  + I * ky * y - z2 + I * kz * z);
}

/* Function for harmonic potential */
REAL complex harmonic(void *arg, REAL x, REAL y, REAL z) {

  pparams params = *((pparams *) arg);

  return 0.5 * (params.kx * params.kx * x * x + params.ky * params.ky * y * y 
                + params.kz * params.kz * z * z);
}
