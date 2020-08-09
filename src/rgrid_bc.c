/* 
 * Real grid (rgrid) boundary conditions.
 *
 */

#include "grid.h"

/*
 * @FUNC{rgrid_value_outside_dirichlet, "Dirichlet boundary condition (real)"}
 * @DESC{"Dirichlet (or constant boundary value) boundary condition for real grid"}
 * @ARG1{rgrid *grid, "Grid to be evaluated"}
 * @ARG2{INT i, "Grid index (x)"}
 * @ARG3{INT j, "Grid index (y)"}
 * @ARG4{INT k, "Grid index (z)"}
 * @RVAL{REAL, "Returns grid value subject to the boundary condition"}
 *
 */

EXPORT REAL rgrid_value_outside_dirichlet(rgrid *grid, INT i, INT j, INT k) {

  return *((REAL *) grid->outside_params_ptr);
}

/*
 * @FUNC{rgrid_value_outside_neumann, "Neumann boundary condition (real)"}
 * @DESC{"Neumann boundary condition for real grid"}
 * @ARG1{rgrid *grid, "Grid to be evaluated"}
 * @ARG2{INT i, "Grid index (x)"}
 * @ARG3{INT j, "Grid index (y)"}
 * @ARG4{INT k, "Grid index (z)"}
 * @RVAL{REAL, "Returns grid value subject to the boundary condition"}
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
 * @FUNC{rgrid_value_outside_periodic, "Periodic boundary condition (real)"}
 * @DESC{"Periodic boundary condition for real grid"}
 * @ARG1{rgrid *grid, "Grid to be evaluated"}
 * @ARG2{INT i, "Grid index (x)"}
 * @ARG3{INT j, "Grid index (y)"}
 * @ARG4{INT k, "Grid index (z)"}
 * @RVAL{REAL, "Returns grid value subject to the boundary condition"}
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
