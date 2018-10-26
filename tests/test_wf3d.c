/*
 * Test program for 3D
 *
 * Propagate wavepacket in 3D.
 *
 * Try for example:
 * ./test_wf3d 1 128 0.1 200 8.0 8.0 8.0 0.25 0.25 0.25 -2.0 -2.0 -2.0 > test.dat
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <omp.h>

REAL complex wavepacket(void *arg, REAL x, REAL y, REAL z);
REAL complex harmonic(void *arg, REAL x, REAL y, REAL z);
REAL complex dip(void *arg, REAL x, REAL y, REAL z);

typedef struct wparams_struct {
  REAL kx, ky, kz;
  REAL wx, wy, wz;
  REAL xc, yc, zc;
} wparams;

typedef struct pparams_struct {
  REAL kx, ky, kz;
  REAL delta;
} pparams;

REAL complex external_potential(void *arg, REAL x, REAL y, REAL z) {

  return harmonic(arg, x, y, z);
}

int main(int argc, char *argv[]) {

  INT i, l, n, iterations, threads;
  REAL x, step, lx;
  REAL time_step;
  REAL complex time;
  
  wf *gwf = NULL;
  cgrid *potential = NULL;
  cgrid *workspace = NULL;
  cgrid *workspace2 = NULL;
  cgrid *sq_grad_pot = NULL;
  rgrid *rworkspace = NULL;
  
  pparams potential_params;
  wparams wp_params;
  
  /* parameters */
  if (argc != 14) {
    fprintf(stderr, "Usage: %s <threads> <points/axis> <time_step> <iterations> <kx> <ky> <kz> <wx> <wy> <wz> <xc> <yc> <zc>\n", argv[0]);
    return -1;
  }
  
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
    fprintf(stderr, "force constant cannot be zero.\n");
    exit(1);
  }
  
  step = 0.4 / (((REAL) n) / 16.0);
  lx = ((REAL) n) * step;
  
  fprintf(stderr, "Grid (" FMT_I "X" FMT_I "X" FMT_I ")\n", n, n, n);
  
  /* potential parameters */
  potential_params.kx = lx * 2.0;
  potential_params.ky = lx * 2.0;
  potential_params.kz = lx * 2.0;
  potential_params.delta = 1;
  
  grid_threads_init(threads);
  
  /* allocate memory (mass = 1.0) */
  gwf = grid_wf_alloc(n, n, n, step, 1.0, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "WF");
  //gwf = grid_wf_alloc(n, n, n, step, 1.0, WF_NEUMANN_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "WF");
  potential = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "potential");
  workspace = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "workspace");
  workspace2 = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "workspace2");
  rworkspace = rgrid_alloc(n, n, n, step, RGRID_PERIODIC_BOUNDARY, 0, "rworkspace");
  sq_grad_pot = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "sq_grad_pot");
  
  /* initialize wave function */
  grid_wf_map(gwf, wavepacket, &wp_params);
  grid_wf_normalize(gwf);
  
  /* map potential */
  cgrid_smooth_map(potential, external_potential, &potential_params, 1);
  
  /* solve */
  time = time_step;
  for(l = 0; l < iterations; l++) {
    grid_wf_square_of_potential_gradient(sq_grad_pot, potential, workspace, workspace2);
    grid_wf_propagate(gwf, potential, sq_grad_pot, time, workspace, workspace2);
#if 1
    grid_wf_density(gwf, rworkspace);
    for (i = 0; i < n; i++) {
      x = ((REAL) (i - n/2)) * step;
      printf(FMT_R " " FMT_R " " FMT_R "\n", time_step * (REAL) l, x, rgrid_value_at_index(rworkspace, i, n/2, n/2));
    }
    printf("\n");
    fflush(stdout);
#endif
  }

  /* release resources */
  grid_wf_free(gwf);
  cgrid_free(workspace);
  cgrid_free(workspace2);
  rgrid_free(rworkspace);
  cgrid_free(sq_grad_pot);
  cgrid_free(potential);
  
  return 0;
}

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

  return CEXP(-x/wx*x/wx + I * kx * x
	      -y/wy*y/wy + I * ky * y
	      -z/wz*z/wz + I * kz * z);
}

REAL complex harmonic(void *arg, REAL x, REAL y, REAL z) {

  pparams params = *((pparams *) arg);

  return 0.5 * params.delta * (
			       params.kx * params.kx * x * x
			       + params.ky * params.ky * y * y
			       + params.kz * params.kz * z * z
			       );
    
}

REAL complex dip(void *arg, REAL x, REAL y, REAL z) {

  REAL pot;
  pparams params = *((pparams *) arg);

  x = 2.0 * M_PI * x / (2.0 * params.kx); 
  y = 2.0 * M_PI * y / (2.0 * params.ky); 
  z = 2.0 * M_PI * z / (2.0 * params.kz); 

  pot = params.delta * 
    (1.0 - COS(x) * COS(x))
    * (1.0 - COS(y) * COS(y))
    * (1.0 - COS(z) * COS(z));

  return pot;
}
