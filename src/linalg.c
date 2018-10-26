/*
 * Linear algebra routines.
 *
 */

#include "grid.h"
#include "private.h"

/*
 * Solve tridiagonal matrix equation A x = b using the Thomas algorithm.
 * This is not stable in general but works for diagonally dominant systems.
 *
 * n = dimensionality (number of equations; INT).
 * a = subdiagonal (preserved) (indexed 1 ... n-1). 0 unused (REAL complex *).
 * b = diagonal (overwritten) (REAL complex *).
 * c = supdiagonal (preserved) (indexed 0 ... n-2). n-1 unused (REAL complex *).
 * v = right hand side vector (overwritten) (REAL complex *).
 * x = solution on exit (REAL complex *).
 * str = stride for writing the output (INT).
 *
 * Note: a and x may be the same array.
 *
 * Status: Compared against Mathematica.
 *
 */

EXPORT inline void grid_solve_tridiagonal_system(INT n, REAL complex *a, REAL complex *b, REAL complex *c, REAL complex *v, REAL complex *x, INT str) {

  REAL complex m;
  INT i;

  for (i = 1; i < n; i++) {
    m = a[i] / b[i-1];
    b[i] -= m * c[i-1];
    v[i] -= m * v[i-1];
  }
 
  /* a not used anymore (it is possible to have a = x) */
  x[str*(n-1)] = v[n-1] / b[n-1];
 
  for (i = n - 2; i >= 0; --i)
    x[i*str] = (v[i] - c[i] * x[str*(i+1)]) / b[i];
}

/*
 * Solve tridiagonal matrix equation A x = b using the Thomas algorithm.
 * This is not stable in general but works for diagonally dominant systems.
 *
 * Sub and sup diagonals are preset for CN with NBC: 1 (+ c2/kx0 term) except at the ends 2 (c2/kx0 does not contribute).
 *
 * n = dimensionality (number of equations; INT).
 * b = diagonal elements (overwritten) (REAL complex *).
 * v = right hand side vector (overwritten) (REAL complex *).
 * x = solution on exit (REAL complex *).
 * c2 = Off diagonal parameter (REAL complex; input). Sub = c2 - 1 (or c2 at rh boundary); Sup = c2 + 1 (or c2 at lh boundary).
 * str = solution may go to a non-continous array (INT; input).
 *
 * Status: Compared against Mathematica.
 *
 */

EXPORT inline void grid_solve_tridiagonal_system2(INT n, REAL complex *b, REAL complex *v, REAL complex *x, REAL complex c2, INT str) {

  REAL complex m, a, c, cc2p, cc2m;
  INT i, ii;

// (dl) & c (du) = 1 +/- c2; computed on the fly to save memory.
// we use b as the diagonal elements (input)

  cc2p = 1.0 + c2;
  cc2m = 1.0 - c2;
  // a sub and c is sup
  for (i = 1; i < n; i++) {
    if(i == n-1) a = 2.0;
    else a = cc2m;  // a[i]
    if(i == 1) c = 2.0;  // c[i-1]
    else c = cc2p;

    m = a / b[i-1];

    b[i] -= m * c;
    v[i] -= m * v[i-1];
  }
 
  x[str*(n-1)] = v[n-1] / b[n-1];
 
  for (i = n - 2, ii = i * str; i >= 0; --i, ii -= str) {
    if(i == 0) c = 2.0;
    else c = cc2p;  // c[i]
    x[ii] = (v[i] - c * x[ii + str]) / b[i];
  }
}

/*
 * Solve tridiagonal matrix equation A x = b using the Thomas algorithm.
 * This is not stable in general but works for diagonally dominant systems.
 *
 * Sub and sup diagonals are preset for CN. This is to be used with periodic BC (from ..._cyclic2 below).
 *
 * n = dimensionality (number of equations; INT).
 * b = diagonal elements (overwritten) (REAL complex *).
 * v = right hand side vector (overwritten) (REAL complex *).
 * x = solution on exit (REAL complex *).
 * c2 = Off diagonal parameter (REAL complex; input). Subdiagonal = c2 - 1; Supdiagonal = c2 + 1.
 * str = solution may go to a non-continous array (INT; input).
 *
 * Status: Compared against Mathematica.
 *
 */

EXPORT inline void grid_solve_tridiagonal_system3(INT n, REAL complex *b, REAL complex *v, REAL complex *x, REAL complex c2, INT str) {

  REAL complex m, a, c, cc2p, cc2m;
  INT i, ii;

// (dl) & c (du) = 1 +/- c2; computed on the fly to save memory.
// we use b as the diagonal elements (input)

  cc2p = 1.0 + c2;
  cc2m = 1.0 - c2;
  // a sub and c is sup
  for (i = 1; i < n; i++) {
    a = cc2m;  // a[i]
    c = cc2p;  // c[i-1]

    m = a / b[i-1];

    b[i] -= m * c;
    v[i] -= m * v[i-1];
  }
 
  x[str*(n-1)] = v[n-1] / b[n-1];
 
  for (i = n - 2, ii = i * str; i >= 0; --i, ii -= str) {
    c = cc2p;  // c[i]
    x[ii] = (v[i] - c * x[ii + str]) / b[i];
  }
}

/*
 * Solve tridiagonal matrix equation A x = b (Sherman-Morrison).
 * Alpha and beta specify the extreme non-zero elements in A.
 * (this arises from finite diff. and periodic boundary conds)
 *
 * n = dimensionality (number of equations) (INT).
 * a = subdiagonal (indexed 1 ... n-1). 0 unused (REAL complex *).
 * b = diagonal (overwritten) (REAL complex *).
 * c = supdiagonal (indexed 0 ... n-2). n-1 unused (REAL complex *).
 * v = right hand side vector (overwritten) (REAL complex *).
 * alpha = left corner matrix element for the last row of A (REAL complex).
 * beta  = right corner matrix element for the first rotw of A (REAL complex).
 * x = solution on exit (REAL complex *).
 * str = solution x may go to a non-continous array (INT; input).
 * bb = worksapce1 (0, ..., n) (REAL complex).
 *
 * Notes:  - with finite difference, use alpha = beta = 1.0 for periodic boundary
 *           and -1.0 for both if anti periodc.
 *
 * Status: Compared against Mathematica.
 *
 */

EXPORT void grid_solve_tridiagonal_system_cyclic(INT n, REAL complex *a, REAL complex *b, REAL complex *c, REAL complex *v, REAL complex alpha, REAL complex beta, REAL complex *x, INT str, REAL complex *bb) {

  REAL complex gamma, fact;
  INT i;

  gamma = -b[0];

  b[0] = bb[0] = b[0] - gamma;
  b[n-1] = bb[n-1] = b[n-1] - alpha * beta / gamma;
  for (i = 1; i < n-1; i++) bb[i] = b[i];

  grid_solve_tridiagonal_system(n, a, bb, c, v, x, str); /* both bb and v overwritten */

  v[0] = gamma;
  v[n-1] = alpha;

  for (i = 1; i < n-1; i++) v[i] = 0.0;

  grid_solve_tridiagonal_system(n, a, b, c, v, a, 1); /* note: a and x may be the same arrays */

  fact = (x[0] + beta * x[str*(n-1)] / gamma) / (1.0 + a[0] + beta * a[n-1] / gamma);
  for(i = 0; i < n; i++)
    x[str*i] -= fact * a[i];
}

/*
 * Solve tridiagonal matrix equation A x = b (Sherman-Morrison).
 * Special version for periodic boundaries and difference from 1st and 2nd derivatives.
 * 
 * Special version of the above:
 * Both sup and subdiagonal cyclic elements are 1.0 (alpha = beta = 1.0 + c2)
 *
 * n = dimensionality (number of equations) (INT).
 * b = diagonal (overwritten) (REAL complex *).
 * v = right hand side vector (overwritten) (REAL complex *).
 * x = solution on exit (REAL complex *).
 * c2 = Off diagonal parameter (REAL complex; input).
 * str = solution x may go to a non-continous array (INT; input).
 * bb = worksapce1 (0, ..., n) (REAL complex).
 *
 * Status: Compared against Mathematica.
 *
 */

EXPORT void grid_solve_tridiagonal_system_cyclic2(INT n, REAL complex *b, REAL complex *v, REAL complex *x, REAL complex c2, INT str, REAL complex *bb) {

  REAL complex gamma, fact, alpha, beta; // beta = top rh corner element, alpha = bottom lh corner element
  INT i;

  alpha = 1.0 + c2;  // Laplacian contrib + c2 * forward term from first derivative (finite difference)
  beta = 1.0 - c2;   // Laplacian contrib - c2 * backward term from first derivative (finite difference)
  gamma = -b[0];

  b[0] = bb[0] = b[0] - gamma;
  b[n-1] = bb[n-1] = b[n-1] - alpha * beta / gamma;
  for (i = 1; i < n-1; i++) bb[i] = b[i];

  grid_solve_tridiagonal_system3(n, bb, v, x, c2, str); /* both bb and v overwritten */

  v[0] = gamma;
  v[n-1] = alpha;
  for (i = 1; i < n-1; i++) v[i] = 0.0;

  grid_solve_tridiagonal_system3(n, b, v, bb, c2, 1);

  fact = (x[0] + beta * x[str*(n-1)] / gamma) / (1.0 + bb[0] + beta * bb[n-1] / gamma);
  for(i = 0; i < n; i++)
    x[str*i] -= fact * bb[i];
}
