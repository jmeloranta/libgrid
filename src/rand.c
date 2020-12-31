/*
 * Random number routines.
 *
 * Based on drand48() - This is NOT thread safe, so do not use these functions
 * inside OMP parallel regions! Also these do not execute in parallel on CPU-based systems.
 *
 * The CUDA variants are in cuRAND libary (see rand-cuda.c and rand-cuda2.cu).
 *
 */

#include "grid.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define EXPORT

static char init = 0;

/* 
 * @FUNC{grid_random_seed, "Set random number generator seed"}
 * @DESC{"Initialize the random number generator"}
 * @ARG1{INT seed, "Set value for the radom number generator. 
                    Set to zero to take the seed from current wallclock time.
                    This is not OpenMP thread safe as it uses drand48().
                    Also if cuda_status is off, when this is called,
                    curand initialization is skipped"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_random_seed(INT seed) {

  if(!seed) srand48(time(0));
  else srand48((long int) seed);

#ifdef USE_CUDA
  if(cuda_status()) grid_cuda_random_seed(seed);
#endif

  init = 1;
}

/*
 * @FUNC{rgrid_random_uniform, "Add uniformly distributed random numbers to real grid"}
 * @DESC{"Add uniform random numbers to grid between -scale and scale"}
 * @ARG1{rgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL scale, "Numbers are produced between -scale and scale"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_random_uniform(rgrid *grid, REAL scale) {

  INT i, j, k;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_random_uniform(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx; i++)
    for(j = 0; j < grid->ny; j++)
      for(k = 0; k < grid->nz; k++)
        grid->value[(i * grid->ny + j) * grid->nz2 + k] += 2.0 * scale * (((REAL) drand48()) - 0.5);
}

/* 
 * @FUNC{rgrid_random_normal, "Add normal distributed random numbers to real grid"}
 * @DESC{"Add normal random numbers to grid between -scale and scale"}
 * @ARG1{rgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL scale, "Width of the distribution -scale, scale"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_random_normal(rgrid *grid, REAL scale) {

  REAL v1, rsq, fac, val;
  static REAL v2;
  static char flag = 1;
  INT i, j, k;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_random_normal(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx; i++)
    for(j = 0; j < grid->ny; j++)
      for(k = 0; k < grid->nz; k++) {
        if(flag) { // Produces two random numbers, unit variance
          do {
            v1 = 2.0 * (((REAL) drand48()) - 0.5);
            v2 = 2.0 * (((REAL) drand48()) - 0.5);
            rsq = v1 * v1 + v2 * v2;
          } while(rsq == 0.0 || rsq > 1.0);
          fac = SQRT(-2.0 * LOG(rsq) / rsq);
          v1 *= fac;
          v2 *= fac;  
          flag = 0;
          val = v1;
        } else {
          flag = 1;
          val = v2;
        }
        grid->value[(i * grid->ny + j) * grid->nz2 + k] += val * scale;
  }
}

/*
 * @FUNC{cgrid_random_uniform, "Add uniformly distributed random numbers to complex grid"}
 * @DESC{"Add unform random numbers to complex grid"}
 * @ARG1{cgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL complex scale, "Scaling factor for random numbers (-scale to scale). Note that this is complex number allowing to scale each component separately"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_random_uniform(cgrid *grid, REAL complex scale) {

  INT i;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_random_uniform(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++)
    grid->value[i] += 2.0 * CREAL(scale) * (((REAL) drand48()) - 0.5) + I * 2.0 * CIMAG(scale) * (((REAL) drand48()) - 0.5);
}

/*
 * @FUNC{cgrid_random_uniform_sp, "Add uniformly distributed random numbers to complex grid (random amplitude and phase)"}
 * @DESC{"Add unform random numbers to complex grid. This assigns random complex vector length and phase, which is different from assigning random numbers to real and imaginary parts separately"}
 * @ARG1{cgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL scale, "Scaling factor for random numbers. This scales the length of the complex vector"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_random_uniform_sp(cgrid *grid, REAL scale) {

  INT i;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_random_uniform_sp(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++)
    grid->value[i] += scale * ((REAL) drand48()) * CEXP(I * 2.0 * M_PI * ((REAL) drand48()));
}

/* 
 * @FUNC{cgrid_random_normal, "Add normal distributed random numbers to complex grid"}
 * @DESC{"Add normal random numbers to complex grid"}
 * @ARG1{cgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL complex scale, "Scaling factor for random numbers (-scale to scale). Note that this is complex number allowing to scale each component separately"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_random_normal(cgrid *grid, REAL complex scale) {

  REAL v1, rsq, fac;
  static REAL v2;
  INT i;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_random_normal(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++) {
    do {
      v1 = 2.0 * (((REAL) drand48()) - 0.5);
      v2 = 2.0 * (((REAL) drand48()) - 0.5);
      rsq = v1 * v1 + v2 * v2;
    } while(rsq == 0.0 || rsq > 1.0);
    fac = SQRT(-2.0 * LOG(rsq) / rsq);
    v1 *= fac;
    v2 *= fac;  
    grid->value[i] += CREAL(scale) * v1 + I * CIMAG(scale) * v2;
  }
}

/* 
 * @FUNC{cgrid_random_normal_sp, "Add normal distributed random numbers to complex grid (gaussian amplitude and uniform phase)"}
 * @DESC{"Add normal random numbers to complex grid. This assigns random complex vector length from gaussian distribution and phase from uniform distribution"}
 * @ARG1{cgrid *grid, "Grid where the random numbers are added to"}
 * @ARG2{REAL scale, "Scaling factor for random numbers. This scales the length of the complex vector"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_random_normal_sp(cgrid *grid, REAL scale) {

  REAL v1, rsq, fac;
  static REAL v2;
  INT i;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_random_normal_sp(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++) {
    do {
      v1 = 2.0 * (((REAL) drand48()) - 0.5);
      v2 = 2.0 * (((REAL) drand48()) - 0.5);
      rsq = v1 * v1 + v2 * v2;
    } while(rsq == 0.0 || rsq > 1.0);
    fac = SQRT(-2.0 * LOG(rsq) / rsq);
    v1 *= fac;
    v2 *= fac;  
    // v2 is not needed
    grid->value[i] += scale * v1 * CEXP(I * M_PI * ((REAL) drand48()));  // Only between 0 and Pi as v1 is +-
  }
}
