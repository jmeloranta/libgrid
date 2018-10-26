#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

#define N 5

int main(int argc, char **argv) {

  REAL complex b[N], v[N], x[N];
  INT i;

  for (i = 0; i < N; i++) {
    b[i] = 5.0; // diagonal
    v[i] = i;   // right hand side
  }
#if 1
  grid_solve_tridiagonal_system3(N, b, v, x, 2.0, 1);  // sup diag = 1.0 + 2.0 = 3.0, sub diag = 1.0 - 2.0 = -1.0
#else
  grid_solve_tridiagonal_system2(N, b, v, x, 2.0, 1);  // sup diag = 1.0 + 2.0 = 3.0, sub diag = 1.0 - 2.0 = -1.0
                                                       // except first element in sup = 2.0 and last in sub = 2.0.
#endif
  for (i = 0; i < N; i++)
    printf("(" FMT_R "," FMT_R ")\n", CREAL(x[i]), CIMAG(x[i]));
  return 0;
}
