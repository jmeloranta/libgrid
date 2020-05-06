#ifndef __RGRIDPRIVATE__
#define  __RGRIDPRIVATE__

typedef struct sShiftParametersr_struct {
  REAL x, y, z;
  rgrid *grid;
} sShiftParametersr;

#ifdef USE_CUDA
static char rgrid_bc_conv(rgrid *grid) {

  if(grid->value_outside == RGRID_DIRICHLET_BOUNDARY) return 0;
  else if(grid->value_outside == RGRID_NEUMANN_BOUNDARY) return 1;
  else if(grid->value_outside == RGRID_PERIODIC_BOUNDARY) return 2;
  else {
    fprintf(stderr, "libgrid(cuda): Incompatible boundary condition.\n");
    abort();
  }
}
#endif

/*
 * Local subroutine for rotating grid around z axis.
 *
 */

static REAL rgrid_value_rotate_z(void *arg, REAL x, REAL y, REAL z) {

  /* Unpack the values in arg */ 
  rgrid *grid = ((grid_rotation *) arg)->rgrid;
  REAL sth = ((grid_rotation *) arg)->sinth, cth = ((grid_rotation *) arg)->costh, xp, yp;

  xp = -y * sth + x * cth; 
  yp =  y * cth + x * sth;

  return rgrid_value(grid, xp, yp, z);
}

/* linear_weight */
static inline REAL linear_weight(REAL x) {

  if (x < 0.0) x = -x;
  return (1.0 - x);
}

static inline REAL rsqnorm(REAL complex x) {

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

/* Shift a real grid function */

static inline REAL shift_rgrid(void *arg, REAL x, REAL y, REAL z) {

  sShiftParametersr *params = (sShiftParametersr *) arg;

  return rgrid_value(params->grid, x - params->x, y - params->y, z - params->z);
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

#endif /*  __RGRIDPRIVATE__ */
