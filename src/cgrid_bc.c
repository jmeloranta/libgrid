/*
 * Complex grid (cgrid) boundary conditions.
 *
 * Laset reviewed: 26 Sep 2019.
 *
 */

#include "grid.h"

/*
 * Dirichlet (or constant boundary value) boundary condition.
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 *
 */

EXPORT REAL complex cgrid_value_outside_dirichlet(cgrid *grid, INT i, INT j, INT k) {

  return *((REAL complex *) grid->outside_params_ptr);
}

/*
 * Neumann boundary condition.
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: For efficiency, this does not roll over if the indices are too large.
 *
 */

EXPORT REAL complex cgrid_value_outside_neumann(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  /* Symmetry points 0 and nx-1; finite difference first derivative: f(x + h) = f(x - h) */
  if (i < 0) i = -i;
  if (i > nx-1) i = 2 * nx - i - 2;

  /* Symmetry points 0 and ny-1; finite difference first derivative: f(y + h) = f(y - h) */
  if (j < 0) j = -j;
  if (j > ny-1) j = 2 * ny - j - 2;

  /* Symmetry points 0 and nz-1; finite difference first derivative: f(z + h) = f(z - h) */
  if (k < 0) k = -k;
  if (k > nz-1) k = 2 * nz - k - 2;

  return cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT fully periodic boundary condition. The symmetry points are 0 and N.
 * This is to be used with regular FFT.
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 */

EXPORT REAL complex cgrid_value_outside_periodic(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  
  /* periodic points 0 and nx */
  i %= nx;
  if (i < 0) i = nx + i;

  /* periodic points 0 and ny */
  j %= ny;
  if (j < 0) j = ny + j;

  /* periodic points 0 and nz */
  k %= nz;
  if (k < 0) k = nz + k;
  
  return cgrid_value_at_index(grid, i, j, k);
}
