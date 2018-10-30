/* 
 * rgrid boundary conditions.
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

EXPORT REAL rgrid_value_outside_constantdirichlet(rgrid *grid, INT i, INT j, INT k) {

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
 * TODO:
 * The symmetry point are i=0 and i=nx-1 for consistency with the FFT plan FFTW_REDFT00. 
 * If one wants to use REDFT01 the symmetry points are i=-0.5 and i=nx-0.5 
 * Do we want to have nd - i - 1 (FFT compatibility)?
 *
 */

EXPORT REAL rgrid_value_outside_neumann(rgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nd;

  nd = nx * 2;
  if (i < 0) i = -i;
  if (i >= nd) i %= nd;
  if (i >= nx) i = nd - i;
  
  nd = ny * 2;
  if (j < 0) j = -j;
  if (j >= nd) j %= nd;
  if (j >= ny) j = nd - j;
  
  nd = nz * 2;
  if (k < 0) k = -k;
  if (k >= nd) k %= nd;
  if (k >= nz) k = nd - k;

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

EXPORT REAL rgrid_value_outside(rgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  i %= nx;
  if (i < 0) i = nx + i;
  j %= ny;
  if (j < 0) j = ny + j;
  k %= nz;
  if (k < 0) k = nz + k;
  
  return rgrid_value_at_index(grid, i, j, k);
}

/*
 * Vortex boundary condition.
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 *
 * NOTE: Not tested and possibly out of date.
 *
 */

EXPORT REAL rgrid_value_outside_vortex(rgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  i %= nx;
  if (i < 0) i = nx + i;
  j %= ny;
  if (j < 0) j = ny + j;
  k %= nz;
  if (k < 0) k = nz + k;
  
  return rgrid_value_at_index(grid, i, j, k);
}
