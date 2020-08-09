/*
 * Interpolation functions.
 *
 */

#include "grid.h"

/*
 * @FUNC{grid_polynomial_interpolate, "Polynomial interpolation"}
 * @DESC{"Polynomial interpolation routine"}
 * @ARG1{REAL *xa, "Array of x values"}
 * @ARG2{REAL *ya, "Array of y values"}
 * @ARG3{INT n, "Number of points for (x,y) arrays"}
 * @ARG4{REAL x, "Point at which the approximation is obtained"}
 * @ARG5{REAL *dy, "Error estimate"}
 * @RVAL{REAL, "Returns the interpolated value"}
 *
 * Source: Numerical Recipes (indexing from zero).
 *
 */

EXPORT REAL grid_polynomial_interpolate(REAL *xa, REAL *ya, INT n, REAL x, REAL *dy) {

  INT i, m, ns = 1;
  REAL den, dif, dift, ho, hp, w, *c, *d, y;

  /* Indexing: preserve Num. Recp's messed up indexing and just shift xa and ya accordingly */
  dif = FABS(x - xa[0]);
  if(!(c = (REAL *) malloc(sizeof(REAL) * (size_t) (n+1))) || !(d = (REAL *) malloc(sizeof(REAL) * (size_t) (n+1)))) {
    fprintf(stderr, "libgrid: Out of memory in grid_polynomial_interpolate().\n");
    abort();
  }
  
  for (i = 1; i <= n; i++) {
    if((dift = FABS(x - xa[i-1])) < dif) {
      ns = i;
      dif = dift;
    }
    c[i] = ya[i-1];
    d[i] = ya[i-1];
  }
  y = ya[ns-1]; ns--;
  for (m = 1; m < n; m++) {
    for (i = 1; i <= n - m; i++) {
      ho = xa[i-1] - x;
      hp = xa[i + m - 1] - x;
      w = c[i+1] - d[i];
      if((den = ho - hp) == 0.0) {
	fprintf(stderr, "libgrid: Polynomial interpolation failed.\n");
	abort();
      }
      den = w / den;
      d[i] = hp * den;
      c[i] = ho * den;
    }
    y += (*dy = (2 * ns < (n - m) ? c[ns+1]:d[ns--]));
  }
  free(c);
  free(d);
  return y;
}

/*
 * @FUNC{grid_spline_ypp, "Spline 2nd derivate auxiliary function"}
 * @DESC{"Second derivative for spline interpolation. To be used with grid_spline_interpolate().
         Must be called before calling grid_spline_interpolate"}
 * @ARG1{REAL *x, "Array of x values"}
 * @ARG2{REAL *y, "Array of y values"}
 * @ARG3{INT n, "Number of points for (x,y) arrays"}
 * @ARG4{REAL yp1, "First derivative at x[1]"}
 * @ARG5{REAL ypn, "First derivative at x[n+1]"}
 * @ARG6{REAL *y2, "Second derivative computed"}
 * @RVAL{void, "No return value"}
 *
 * Source: Numerical Recipes (indexing from zero).
 *
 */

EXPORT void grid_spline_ypp(REAL *x, REAL *y, INT n, REAL yp1, REAL ypn, REAL *y2) {

  INT i, k;
  REAL p, qn, sig, un, *u;

  if(!(u = (REAL *) malloc(sizeof(REAL) * (size_t) n))) {
    fprintf(stderr, "libgrid: Out of memory in grid_spline_ypp().\n");
    abort();
  }

  if(yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }

  for(i = 1; i <= n-2; i++) { 
    sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);
    p = sig * y2[i-1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1]);
    u[i] = (6.0 * u[i] / (x[i+1] - x[i-1]) - sig * u[i-1]) / p;
  }
  if(ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0 / (x[n] - x[n-1])) * (ypn - (y[n] - y[n-1]) / (x[n] - x[n-1]));
  }
  y2[n-1] = (un - qn * u[n-2]) / (qn * y2[n-2] + 1.0);
  for(k = n - 2; k >= 0; k--)
    y2[k] = y2[k] * y2[k+1] + u[k];

 free(u);
}

/*
 * @FUNC{grid_spline_interpolate, "Spline interpolation"}
 * @DESC{"Spline interpolation routine."}
 * @ARG1{REAL *xa, "Array of x values"}
 * @ARG2{REAL *ya, "Array of y values"}
 * @ARG3{REAL *y2a, "Array of second derivatives of y values. Computed by grid_spline_ypp()"}
 * @ARG4{INT n, "Number of points for (x,y) arrays"}
 * @ARG5{REAL x, "Point at which the approximation is obtained"}
 * @RVAL{REAL, "Returns the interpolated value"}
 *
 * Source: Numerical Recipes (indexing from zero).
 *
 */

EXPORT REAL grid_spline_interpolate(REAL *xa, REAL *ya, REAL *y2a, INT n, REAL x) {

  INT klo, khi, k;
  REAL h, b, a;

  klo = 0;
  khi = n - 1;
  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if(xa[k] > x) khi = k;
    else klo = k;
  }

  h = xa[khi] - xa[klo];
  if(h == 0.0) {
    fprintf(stderr, "libgrid: Bad x input values for grid_spline_interpolation() (repeated value).\n");
    abort();
  }

  a = (xa[khi] - x) / h;
  b = (x - xa[klo]) / h;
  return a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
}

