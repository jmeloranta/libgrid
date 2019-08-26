/*
 * Random number routines.
 *
 * Based on drand48() - This is NOT thread safe, so do not use these functions
 * inside OMP parallel regions!
 *
 */

#include <math.h>
#include <stdlib.h>
#include <time.h>
#define EXPORT

static char init = 0;

/* 
 * Initialize the random number generator.
 *
 * seed = See value for the radom number generator (INT).
 *        Set to zero to take the seed from current wallclock time.
 *
 * NOTE: This is not thread safe.
 *
 */

void grid_random_seed(INT seed) {

  if(!seed) srand48(time(0));
  else srand48((long int) seed);
  init = 1;
}

/*
 * Calculate random number between -1 and +1 (uniform distribution).
 *
 * At present this is just a wrapper for drand48().
 *
 * Returns the random number.
 *
 * NOTE: This is not thread safe.
 *
 */

EXPORT REAL grid_random() {

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }
  return 2.0 * (((REAL) drand48()) - 0.5);
}

/* 
 * Calculate random number between -1 and +1 (normal distribution) (from Numerical Recipes).
 * 
 * This uses grid_random() (which by default uses srand48() / drand48()).
 * 
 * Returns the random number (REAL).
 *
 * NOTE: This is not thread safe.
 *
 */

EXPORT REAL grid_random_normal() {

  REAL v1, rsq, fac;
  static REAL v2;
  static char flag = 1;

  if(!init) {
    grid_random_seed(0);
    init = 1;
  }
  if(flag) {
    do {
      v1 = grid_random();
      v2 = grid_random();
      rsq = v1 * v1 + v2 * v2;
    } while(rsq == 0.0 || rsq > 1.0);
    fac = SQRT(-2.0 * LOG(rsq) / rsq);
    v1 *= fac;
    v2 *= fac;  
    flag = 0;
    return v1;
  } else {
    flag = 1;
    return v2;
  }
}
