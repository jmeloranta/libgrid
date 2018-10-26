#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

#define N 5

int main(int argc, char **argv) {

  REAL complex b[N], v[N], x[N], wrk[N];
  INT i;

  for (i = 0; i < N; i++) {
    b[i] = 5.0; // diagonal
    v[i] = i;   // right hand side
  }
  grid_solve_tridiagonal_system_cyclic2(N, b, v, x, 2.0, 1, wrk);  // sup diag = 1.0 + 2.0 = 3.0, sub diag = 1.0 - 2.0 = -1.0
                                                       // beta = 1.0 - 2.0 = -1.0 (top rh corner) and alpha = 1.0 + 2.0 = 3.0 (bottom lh corner)
  for (i = 0; i < N; i++)
    printf("(" FMT_R "," FMT_R ")\n", CREAL(x[i]), CIMAG(x[i]));
  return 0;
}
