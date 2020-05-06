#ifndef __CGRIDPRIVATE__
#define  __CGRIDPRIVATE__

typedef struct sShiftParametersc_struct {
  REAL x, y, z;
  cgrid *grid;
} sShiftParametersc;

/* linear_weight */
static inline REAL linear_weight(REAL x) {

  if (x < 0.0) x = -x;
  return (1.0 - x);
}

static inline REAL csqnorm(REAL complex x) {

  return CREAL(x) * CREAL(x) + CIMAG(x) * CIMAG(x);
}

/*
 * linearly weighted integral
 *
 * in 1D:
 *      x-step        x         x+step
 *                    1
 * ns = 1 |-----------|-----------| 
 *             1/2         1/2
 * ns = 2 |-----+-----|-----+-----|
 *           1/3     1/3     1/3
 * ns = 3 |---+-------|-------+---| 
 *
 */

static inline REAL complex linearly_weighted_integralc(REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, REAL x, REAL y, REAL z, REAL step, INT ns) {

  INT i, j, k;
  REAL xs, ys, zs;
  REAL w, wx, wy, wz;
  REAL substep = 2.0 * step / (REAL) ns;
  REAL complex sum;
  REAL wsum;
  
  sum = 0.0;
  wsum = 0.0;
  
  for(i = 0; i < ns; i++) {
    xs = -step + (((REAL) i) + 0.5) * substep;
    wx = linear_weight(xs / step);
    
    for(j = 0; j < ns; j++) {
      ys = -step + (((REAL) j) + 0.5) * substep;
      wy = linear_weight(ys / step);
      
      for(k = 0; k < ns; k++) {
        zs = -step + (((REAL) k) + 0.5) * substep;
        wz = linear_weight(zs / step);
        
        w = wx * wy * wz;
	
	sum += w * func(farg, x + xs, y + ys, z + zs);
        wsum += w;
      }
    }
  }
  
  return sum / wsum;
}

/* Shift a complex grid function */

static inline REAL complex shift_cgrid(void *arg, REAL x, REAL y, REAL z) {

  sShiftParametersc *params = (sShiftParametersc *) arg;

  return cgrid_value(params->grid, x - params->x, y - params->y, z - params->z);
}

/*
 * Integer version of pow().
 *
 */

static inline REAL ipow(REAL x, INT n) {

  INT ii, sig;
  REAL value = 1.0;

  if(n == 0) return 1.0;
  sig = (n < 0) ? -1:1;
  n = ABS(n);
  switch(n) {
    case 1:      
      break;
    case 2:
      x *= x;
      break;
    case 3:
      x *= x * x;
      break;
    default:
      for(ii = 0; ii < n; ii++)
        value *= x;
      x = value;
  }
  if(sig == -1) x = 1.0 / x;
  return x;
}

#endif /*  __CGRIDPRIVATE__ */
