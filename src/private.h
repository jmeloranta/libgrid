#ifndef __GRIDPRIVATE__
#define  __GRIDPRIVATE__

typedef struct sShiftParametersc_struct {
  REAL x, y, z;
  cgrid *grid;
} sShiftParametersc;

typedef struct sShiftParametersr_struct {
  REAL x, y, z;
  rgrid *grid;
} sShiftParametersr;

/* linear_weight */
static inline REAL linear_weight(REAL x) {

  if (x < 0.0) x = -x;
  return (1.0 - x);
}

static inline REAL sqnorm(REAL complex x) {

  return CREAL(x) * CREAL(x) + CIMAG(x) * CIMAG(x);
}

static inline REAL rms(REAL *data, INT n) {

  INT i; 
  REAL sum = 0;

  for(i = 0; i < n; i++)
    sum += data[i] * data[i];
  return SQRT(sum / (REAL) n);
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

static inline REAL linearly_weighted_integralr(REAL (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, REAL x, REAL y, REAL z, REAL step, INT ns) {

  INT i, j, k;
  REAL xs, ys, zs;
  REAL w, wx, wy, wz;
  REAL substep = 2.0 * step / (REAL) ns;
  REAL sum;
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

/* Shift a real grid function */

static inline REAL shift_rgrid(void *arg, REAL x, REAL y, REAL z) {

  sShiftParametersr *params = (sShiftParametersr *) arg;

  return rgrid_value(params->grid, x - params->x, y - params->y, z - params->z);
}

/*
 * Subroutine for rotating grid around z axis.
 *
 */

static REAL rgrid_value_rotate_z(void *arg, REAL x, REAL y, REAL z) {

  /* Unpack the values in arg */ 
  rgrid *grid = ((rotation *) arg)->rgrid;
  REAL sth = ((rotation *) arg)->sinth, cth = ((rotation *) arg)->costh, xp, yp;

  xp = -y * sth + x * cth; 
  yp =  y * cth + x * sth;

  return rgrid_value(grid, xp, yp, z);
}

#endif /*  __GRIDPRIVATE__ */
