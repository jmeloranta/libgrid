#include <stdio.h>
#include <math.h>
#include <grid/grid.h>

#define N 5

int main(int argc, char **argv) {

  REAL complex a[N], b[N], c[N], v[N], x[N], wrk[N], alpha, beta;
  INT i;

  for (i = 0; i < N; i++) {
    a[i] = 1.0; // subdiagonal
    b[i] = 5.0; // diagonal
    c[i] = 2.0; // supdiagonal
    v[i] = i;   // right hand side
  }
  alpha = 1.0; beta = 1.0;
  grid_solve_tridiagonal_system_cyclic(N, a, b, c, v, alpha, beta, x, 1, wrk);
  for (i = 0; i < N; i++)
    printf("(" FMT_R "," FMT_R ")\n", CREAL(x[i]), CIMAG(x[i]));
  return 0;
}
