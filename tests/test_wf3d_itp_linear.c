/*
 * Test program for itp_linear routines.
 *
 * Try for example:
 * ./test_wf3d_itp_linear 1 32 5 1 1.0 1e-4 1000 > test.dat
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

REAL complex random_wf(void *arg, REAL x, REAL y, REAL z);

typedef struct pparams_struct {
  REAL kx, ky, kz;
  REAL delta;
} pparams;

REAL complex harmonic(void *arg, REAL x, REAL y, REAL z);
REAL complex dip(void *arg, REAL x, REAL y, REAL z);

REAL complex external_potential(void *arg, REAL x, REAL y, REAL z) {

  return harmonic(arg,x,y,z);
}

int main(int argc, char *argv[]) {

  INT i, n, iterations, states, virtuals, threads;
  INT riterations;
  REAL step, lx, threshold, rtau;
  REAL tau, erms;
  wf **gwf = 0;
  cgrid *potential = 0;
  cgrid *workspace = 0;
  pparams potential_params;
  
  /* parameters */
  if (argc < 8) {
    fprintf(stderr, "Usage: %s <threads> <points/axis> <states> <virtuals> <time_step> <threshold> <iterations>\n", argv[0]);
    return 0;
  }
  
  threads = atoi(argv[1]);
  n = atoi(argv[2]);
  states = atoi(argv[3]);
  virtuals = atoi(argv[4]);
  tau = atof(argv[5]);
  threshold = atof(argv[6]);
  iterations = atol(argv[7]);
  
  step = 0.4 / (((REAL) n) / 16.0);
  lx = ((REAL) n) * step;
  
  fprintf(stderr, "Grid (" FMT_I "x" FMT_I "x" FMT_I ")\n", n, n, n);
  
  /* potential parameters */
  potential_params.kx = lx/3.0;
  potential_params.ky = lx/3.0;
  potential_params.kz = lx/3.0;
  potential_params.delta = 10;
    
  grid_threads_init(threads);
  
  /* allocate memory */
  gwf = (wf **) malloc(((size_t) states) * sizeof(wf *));
  for(i = 0; i < states; i++)
    gwf[i] = grid_wf_alloc(n, n, n, step, 1.0, WF_NEUMANN_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "WFs");
  potential = cgrid_alloc(n, n, n, step, CGRID_NEUMANN_BOUNDARY, 0, "potential");
  
  /* initialize wave function */
  for(i = 0; i < states; i++) {
    grid_wf_map(gwf[i], random_wf, 0);
    grid_wf_normalize(gwf[i]);
  }
  
  /* map potential */
  cgrid_smooth_map(potential, external_potential, &potential_params, 1);
  
  /* solve */
  erms = grid_itp_linear(gwf, states, virtuals, potential, tau, threshold, iterations, &rtau, &riterations);
  fprintf(stderr, "RMS of error = " FMT_R "\n", erms); 
  
  /* print energies */
  workspace = cgrid_alloc(n, n, n, step, CGRID_NEUMANN_BOUNDARY, 0, "workspace");
  
#if 1
  for(i = 0; i < states; i++)
    fprintf(stderr, " " FMT_R " ", grid_wf_energy(gwf[i], potential, workspace));
  fprintf(stderr, "\n");
#if 1
  /* print wave function */
  for(i = 0; i < states; i++)
    grid_wf_print(gwf[i], stdout);
#endif
#endif
    
  /* release resources */
  for(i = 0; i < states; i++)
    grid_wf_free(gwf[i]);
  cgrid_free(workspace);
  cgrid_free(potential);
  
  free(gwf);
  
  return 0;
}


REAL complex random_wf(void *arg, REAL x, REAL y, REAL z) {

  return (REAL) drand48();
}

REAL complex harmonic(void *arg, REAL x, REAL y, REAL z) {

  pparams params = *((pparams *) arg);

  return 0.5 * params.delta * (params.kx*params.kx * x*x + 
			       params.ky*params.ky * y*y + 
			       params.kz*params.kz * z*z);
}

REAL complex dip(void *arg, REAL x, REAL y, REAL z) {

  REAL pot;
  pparams params = *((pparams *) arg);

  x = 2.0 * M_PI * x / (2.0 * params.kx); 
  y = 2.0 * M_PI * y / (2.0 * params.ky); 
  z = 2.0 * M_PI * z / (2.0 * params.kz); 
  pot = params.delta * (1.0 - COS(x) * COS(x) * COS(y) * COS(y) * COS(z) * COS(z));
  return pot;
}
