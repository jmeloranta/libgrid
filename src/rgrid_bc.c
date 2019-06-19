/* 
 * Real grid (rgrid) boundary conditions.
 *
 */

#include "grid.h"
#include "private.h"

/*
 * Dirichlet (or constant boundary value) boundary condition.
 *
 * grid = grid to be evaluated (rgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 *
 */

EXPORT REAL rgrid_value_outside_dirichlet(rgrid *grid, INT i, INT j, INT k) {

  return *((REAL *) grid->outside_params_ptr);
}

/*
 * Neumann boundary condition.
 *
 * grid = grid to be evaluated (rgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 */

EXPORT REAL rgrid_value_outside_neumann(rgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  if (i < 0) i = -i;
  if (i > nx-1) i = 2 * nx - i - 2;
  
  if (j < 0) j = -j;
  if (j > ny-1) j = 2 * ny - j - 2;
  
  if (k < 0) k = -k;
  if (k > nz-1) k = 2 * nz - k - 2;

  return rgrid_value_at_index(grid, i, j, k);
}

/*
 * Periodic boundary condition.
 *
 * grid = grid to be evaluated (rgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 */

EXPORT REAL rgrid_value_outside_periodic(rgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  i %= nx;
  if (i < 0) i = nx + i;
  j %= ny;
  if (j < 0) j = ny + j;
  k %= nz;
  if (k < 0) k = nz + k;
  
  return rgrid_value_at_index(grid, i, j, k);
}
