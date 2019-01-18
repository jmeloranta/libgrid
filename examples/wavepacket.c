/*
 * Example: Propagate wavepacket in harmonic potential (3D).
 *
 * Try for example:
 * ./wavepacket 1 128 10.0 200 8.0 8.0 8.0 0.25 0.25 0.25 -2.0 -2.0 -2.0 > test.dat
 *
 * Arguments:
 * 1st  = Number of threads to be used.
 * 2nd  = Number of points in X, Y, Z directions (N x N x N grid).
 * 3rd  = Time step (atomic units).
 * 4th  = Number of iterations to run.
 * 5th  = Force constant along X (harmonic potential).
 * 6th  = Force constant along Y (harmonic potential).
 * 7th  = Force constant along Z (harmonic potential).
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
  cgrid *potential, *workspace, *workspace2, *sq_grad_pot;
  rgrid *rworkspace;
  char fname[256];
  pparams potential_params;
  wparams wp_params;
  
  /* Parameter check */
  if (argc != 14) {
    fprintf(stderr, "Usage: %s <threads> <points/axis> <time_step> <iterations> <kx> <ky> <kz> <wx> <wy> <wz> <xc> <yc> <zc>\n", argv[0]);
    return -1;
  }
  
  /* Parse command line arguments */
  threads = atoi(argv[1]);
  n = atoi(argv[2]);
  time_step = atof(argv[3]);
  iterations = atol(argv[4]);
  wp_params.kx = atof(argv[5]);
  wp_params.ky = atof(argv[6]);
  wp_params.kz = atof(argv[7]);
  wp_params.wx = atof(argv[8]);
  wp_params.wy = atof(argv[9]);
  wp_params.wz = atof(argv[10]);
  wp_params.xc = atof(argv[11]);
  wp_params.yc = atof(argv[12]);
  wp_params.zc = atof(argv[13]);

  if(wp_params.wx == 0.0 || wp_params.wy == 0.0 || wp_params.wz == 0.0) {
    fprintf(stderr, "Width cannot be zero.\n");
    exit(1);
  }

  if(wp_params.kx == 0.0 || wp_params.ky == 0.0 || wp_params.kz == 0.0) {
    fprintf(stderr, "Force constant cannot be zero.\n");
    exit(1);
  }
  
  /* Set spatial grid step length based on number of grid points */
  step = 0.4 / (((REAL) n) / 16.0);
  
  fprintf(stderr, "Grid (" FMT_I "X" FMT_I "X" FMT_I ")\n", n, n, n);
  
  /* potential parameters */
  lx = ((REAL) n) * step;
  potential_params.kx = lx * 2.0;
  potential_params.ky = lx * 2.0;
  potential_params.kz = lx * 2.0;
  
  /* Initialize OpenMP threads */
  grid_threads_init(threads);
  
  /* allocate memory (mass = 1.0) */
  gwf = grid_wf_alloc(n, n, n, step, 1.0, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "WF");
  potential = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "potential");
  workspace = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "workspace");
  workspace2 = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "workspace2");
  rworkspace = rgrid_alloc(n, n, n, step, RGRID_PERIODIC_BOUNDARY, 0, "rworkspace");
  sq_grad_pot = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "sq_grad_pot");
  
  /* Initialize wave function */
  grid_wf_map(gwf, wavepacket, &wp_params);
  grid_wf_normalize(gwf);
  
  /* Map potential */
  cgrid_smooth_map(potential, harmonic, &potential_params, 1);
  
  /* Propagate */
  time = time_step;
  for(l = 0; l < iterations; l++) {
    /* Propagate one time step */
    grid_wf_square_of_potential_gradient(sq_grad_pot, potential, workspace, workspace2);
    grid_wf_propagate(gwf, potential, sq_grad_pot, time, workspace);
    /* Write |psi|^2 to output-* files */
    grid_wf_density(gwf, rworkspace);
    sprintf(fname, "output-" FMT_I, l);
    rgrid_write_grid(fname, rworkspace);
  }

  /* Release resources */
  grid_wf_free(gwf);
  cgrid_free(workspace);
  cgrid_free(workspace2);
  rgrid_free(rworkspace);
  cgrid_free(sq_grad_pot);
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
  
  x -= xc;
  y -= yc;
  z -= zc;

  return CEXP(-x / wx * x / wx + I * kx * x - y / wy * y / wy + I * ky * y - z / wz * z / wz + I * kz * z);
}

/* Function for harmonic potential */
REAL complex harmonic(void *arg, REAL x, REAL y, REAL z) {

  pparams params = *((pparams *) arg);

  return 0.5 * (params.kx * params.kx * x * x + params.ky * params.ky * y * y + params.kz * params.kz * z * z);
}
