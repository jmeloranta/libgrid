/*
 * Test program for itp_nonlinear routines.
 *
 * Try for example:
 * ./test_wf3d_itp_nonlinear 1 32 1 10000 1e-4 1000 > test.dat
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>

REAL complex random_wf(void *arg, REAL x, REAL y, REAL z);

typedef struct lpparams_struct {
  REAL kx, ky, kz;
  REAL delta;
} lpparams;

typedef struct gpparams_struct {
  REAL mu, lambda, particles;
} gpparams;

typedef struct pparams_struct {
  lpparams linear;
  gpparams gp;
} pparams;

REAL complex harmonic(void *arg, REAL x, REAL y, REAL z);
REAL complex dip(void *arg, REAL x, REAL y, REAL z);
REAL complex external_potential(void *arg, REAL x, REAL y, REAL z) {

  return dip(arg, x, y, z);
}

void calculate_potentials(cgrid **potential, void *arg, wf **gwf, INT states);

int main(int argc, char *argv[]) {

  INT i, n, iterations, states = 1, virtuals = 0, threads, particles;
  INT riterations;
  REAL step, lx, threshold;
  REAL tau, erms, rtau;
  wf **gwf = 0;
  cgrid *workspace = 0, **potential = 0;
  pparams potential_params;
  
  /* parameters */
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <threads> <points/axis> <particles> <time_step> <threshold> <iterations>\n", argv[0]);
    return 0;
  }
  
  threads = atoi(argv[1]);
  n = atoi(argv[2]);
  particles = atoi(argv[3]);
  tau = atof(argv[4]);
  threshold = atof(argv[5]);
  iterations = atol(argv[6]);
  
  step = 1.6 / (((REAL) n) / 16.0);
  lx = ((REAL) n) * step;
  
  fprintf(stderr, "Grid (%ldx%ldx%ld)\n", n, n, n);
  
  /* potential parameters */
  potential_params.linear.kx = lx;
  potential_params.linear.ky = lx;
  potential_params.linear.kz = lx;
  potential_params.linear.delta = 100.0 / GRID_AUTOK;
  potential_params.gp.mu = 7.0 / GRID_AUTOK;
  potential_params.gp.lambda = potential_params.gp.mu / (GRID_AUTOFS * (GRID_AUTOANG*GRID_AUTOANG*GRID_AUTOANG)); /* mu / rho_0 */
  potential_params.gp.mu = 0.0 * 7.0 / GRID_AUTOK;
  potential_params.gp.particles = (REAL) particles;
  
  grid_threads_init(threads);
  
  /* allocate memory */
  gwf = (wf **) malloc(((size_t) states) * sizeof(wf *));
  potential = (cgrid **) malloc(((size_t) states) * sizeof(cgrid *));
  for(i = 0; i < states; i++) {
    gwf[i] = grid_wf_alloc(n, n, n, step, 4.0026 / GRID_AUTOAMU /*He*/, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "WFs");
    potential[i] = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "potentials");
  }
  workspace = cgrid_alloc(n, n, n, step, CGRID_PERIODIC_BOUNDARY, 0, "workspace");
  
  /* initialize wave function */
  for(i = 0; i < states; i++) {
    grid_wf_map(gwf[i], random_wf, 0);
    grid_wf_normalize(gwf[i]);
  }
  
  /* solve */
  erms = grid_itp_nonlinear(gwf, states, virtuals, calculate_potentials, &potential_params, tau, threshold, iterations, &rtau, &riterations);
  fprintf(stderr, "RMS of error = %le\n", erms);   

  calculate_potentials(potential, &potential_params, gwf, states);
  for (i = 0; i < states; i++)
    fprintf(stderr, " %24.16lf ", grid_wf_energy(gwf[i], potential[i], workspace));
  fprintf(stderr, "\n");

#if 0
  /* print wfs */  
  for(i = 0; i < states; i++)
    grid_wf_print(gwf[i], stdout);  
#endif
  
  /* release resources */
  for(i = 0; i < states; i++)
    grid_wf_free(gwf[i]);
  cgrid_free(workspace);
  
  free(gwf);
  
  return 0;
}

REAL complex random_wf(void *arg, REAL x, REAL y, REAL z) {

  return drand48();
}

REAL complex harmonic(void *arg, REAL x, REAL y, REAL z) {

  lpparams params = *((lpparams *) arg);

  return 0.5 * params.delta * (params.kx*params.kx * x*x + 
			       params.ky*params.ky * y*y + 
			       params.kz*params.kz * z*z);
}

REAL complex dip(void *arg, REAL x, REAL y, REAL z) {

  REAL pot;
  lpparams params = *((lpparams *) arg);

  x = 2.0 * M_PI * x / (2.0 * params.kx); 
  y = 2.0 * M_PI * y / (2.0 * params.ky); 
  z = 2.0 * M_PI * z / (2.0 * params.kz); 
  pot = params.delta * (1.0 - COS(x)*COS(x) * COS(y)*COS(y) * COS(z)*COS(z));
  return pot;
}

void calculate_potentials(cgrid **potential, void *arg, wf **gwf, INT states) {

  INT i;
  pparams *params = (pparams *) arg; 
  cgrid *tmp = cgrid_alloc(potential[0]->nx, potential[0]->ny, potential[0]->nz, potential[0]->step, CGRID_PERIODIC_BOUNDARY, 0, "tmp");
  
  for(i = 0; i < states; i++) {
    cgrid_conjugate_product(potential[i], gwf[i]->grid, gwf[i]->grid);
    cgrid_multiply_and_add(potential[i], 
			    params->gp.particles * params->gp.lambda, 
			    - params->gp.mu);
    cgrid_map(tmp, external_potential, &params->linear);
    cgrid_sum(potential[i], potential[i], tmp);
  }
  
  cgrid_free(tmp);
}
