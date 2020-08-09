/*
 * Linear algebra routines.
 *
 */

#include "grid.h"

/*
 * @FUNC{grid_solve_tridiagonal_system, "Solve tridiagonal linear system"}
 * @DESC{"Solve tridiagonal matrix equation $A x = b$ using the Thomas algorithm.
          This is not stable in general but works for diagonally dominant systems.
          Note that A and x may be the same array"}
 * @ARG1{INT n, "Dimensionality (number of equations)"}
 * @ARG2{REAL complex *a, "Subdiagonal (preserved) (indexed 1 ... n-1). 0 unused"}
 * @ARG3{REAL complex *b, "Diagonal (overwritten)"}
 * @ARG4{REAL complex *c, "Supdiagonal (preserved) (indexed 0 ... n-2). n-1 unused"}
 * @ARG5{REAL complex *v, "Right hand side vector (overwritten)"}
 * @ARG6{REAL complex *x, "Solution on exit"}
 * @ARG7{INT str, "Stride for writing the output"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_solve_tridiagonal_system2, "Solve tridiagonal linear system (Neumann Crank-Nicolson)"}
 * @DESC{"Solve tridiagonal matrix equation $A x = b$ using the Thomas algorithm.
          This is not stable in general but works for diagonally dominant systems.
          Sub and sup diagonals are preset for CN with Neumann BC: 1 (+ c2/kx0 term) except 
          at the ends 2 (c2/kx0 does not contribute)"}
 * @ARG1{INT n, "Dimensionality (number of equations)"}
 * @ARG2{REAL complex *b, "Diagonal elements (overwritten)"}
 * @ARG3{REAL complex *v, "Right hand side vector (overwritten)"}
 * @ARG4{REAL complex *x, "Solution on exit"}
 * @ARG5{REAL complex c2, "Off diagonal parameter. Sub = 1 - c2 (or c2 at rh boundary); Sup = 1 + c2 (or c2 at lh boundary)"}
 * @ARG6{INT str, "Solution may go to a non-continuous array (stride)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_solve_tridiagonal_system3, "Solve tridiagonal linear system (periodic Crank-Nicolson)"}
 * @DESC{"Solve tridiagonal matrix equation $A x = b$ using the Thomas algorithm.
          This is not stable in general but works for diagonally dominant systems.
          Sub and sup diagonals are preset for CN. This is to be used with periodic BC"}
 * @ARG1{INT n, "Dimensionality (number of equations)"}
 * @ARG2{REAL complex *b, "Diagonal elements (overwritten)"}
 * @ARG3{REAL complex *v, "Right hand side vector (overwritten)"}
 * @ARG4{REAL complex *x, "Solution on exit"}
 * @ARG5{REAL complex c2, "Off diagonal parameter. Subdiagonal = c2 - 1; Supdiagonal = c2 + 1"}
 * @ARG6{INT str, "Solution may go to a non-continuous array (stride)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_solve_tridiagonal_system_cyclic, "Solve linear tridiagonal system (Sherman-Morrison)"}
 * @DESC{"Solve tridiagonal matrix equation $A x = b$ (Sherman-Morrison).
          Alpha and beta specify the extreme non-zero elements in A.
          (this arises from finite diff. and periodic boundary conds)
          Note that with finite difference, use alpha = beta = 1.0 for periodic boundary
          and -1.0 for both if anti periodc"}
 * @ARG1{INT n, "Dimensionality (number of equations)"}
 * @ARG2{REAL complex *a, "Subdiagonal (indexed 1 ... n-1). 0 unused"}
 * @ARG3{REAL complex *b, "Diagonal (overwritten)"}
 * @ARG4{REAL complex *c, "Supdiagonal (indexed 0 ... n-2). n-1 unused"}
 * @ARG5{REAL complex *v, "Right hand side vector (overwritten)"}
 * @ARG6{REAL complex alpha, "Left corner matrix element for the last row of A"}
 * @ARG7{REAL beta, "Right corner matrix element for the first row of A"}
 * @ARG8{REAL complex *x, "Solution on exit"}
 * @ARG9{INT str, "Solution may go into a non-continuous array (stride)"}
 * @ARG10{REAL complex *bb, "Worksapce vector of dimension n"}
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
 * @FUNC{grid_solve_tridiagonal_system_cyclic2, "Solve linear tridiagonal system (Sherman-Morrison special)"}
 * @DESC{"Solve tridiagonal matrix equation $A x = b$ (Sherman-Morrison).
          Special version for periodic boundaries and difference from 1st and 2nd derivatives.
          Like grid_solve_tridiagonal_system_cyclic() but:
          both sup and subdiagonal cyclic elements are 1.0 (alpha = beta = 1.0 + c2)"}
 * @ARG1{INT n, "Dimensionality (number of equations)"}
 * @ARG2{REAL complex *b, "Diagonal (overwritten)"}
 * @ARG3{REAL complex *v, "Right hand side vector (overwritten)"}
 * @ARG4{REAL complex *x, "Solution on exit"}
 * @ARG5{REAL complex c2, "Off diagonal parameter"}
 * @ARG6{INT str, "Solution may go to a non-continuous array (stride)"}
 * @ARG7{REAL complex *bb, "Worksapce1 (0, ..., n)"}
 * @RVAL{void, "No return value"}
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
