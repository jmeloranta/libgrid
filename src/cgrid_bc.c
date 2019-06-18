/*
 * Complex grid (cgrid) boundary conditions.
 *
 */

#include "grid.h"
#include "private.h"

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
 * TODO:
 * The symmetry point are i=0 and i=nx-1 for consistency with the FFT plan FFTW_REDFT00. 
 * If one wants to use REDFT01 the symmetry points are i=-0.5 and i=nx-0.5 
 * Do we want to have nd - i - 1 (FFT compatibility)?
 *
 */

EXPORT REAL complex cgrid_value_outside_neumann(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nd;

  nd = nx * 2;
  if (i < 0) i  = -i;
  if (i >= nd) i %= nd;
  if (i >= nx) i  = nd - i - 1;
  
  nd = ny * 2;
  if (j < 0) j  = -j;
  if (j >= nd) j %= nd;
  if (j >= ny) j  = nd - j - 1;
  
  nd = nz * 2;
  if (k < 0) k  = -k;
  if (k >= nd) k %= nd;
  if (k >= nz) k  = nd - k - 1;

  return cgrid_value_at_index(grid, i, j, k);
}

/*
 * Periodic boundary condition.
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
  
  i %= nx;
  if (i < 0) i = nx + i;
  j %= ny;
  if (j < 0) j = ny + j;
  k %= nz;
  if (k < 0) k = nz + k;
  
  return cgrid_value_at_index(grid, i, j, k);
}

/*
 * Vortex boundary condition (x).
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

EXPORT REAL complex cgrid_value_outside_vortex_x(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nd;

  /* In vortex direction, map i to {0,nx-1}. This does not introduce any phase */
  nd = nx * 2;
  if (i < 0) i  = -i - 1;
  if (i >= nd) i %= nd;
  if (i >= nx) i  = nd -1 - i;
  
  /* First, map j,k to {0,2*nx-1} range
   * (this does not introduce any phase)
   */
  nd = ny * 2;
  if (j < 0) j = j % nd + nd;
  if (j >= nd) j %= nd;

  nd = nz * 2;
  if (k < 0 ) k = k % nd + nd;
  if (k >= nd) k %= nd;

  /* Then, if j has to be mapped to {0,nx-1} return -phi*
   *       if k has to be mapped to {0,ny-1} return +phi*
   *       if both have to be mapped return -phi
   */
  if(j >= ny) {
    j = 2 * ny - 1 - j;
    if(k >= nz) {
      k = nd - 1 - k;
      return -cgrid_value_at_index(grid, i, j, k);
    } else
       return -CONJ(cgrid_value_at_index(grid, i, j, k));
  } else {
    if(k >= nz) {
      k = nd - 1 - k;
      return CONJ(cgrid_value_at_index(grid, i, j, k));
    } else return cgrid_value_at_index(grid, i, j, k);
  }
}

/*
 * Vortex boundary condition (y).
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

EXPORT REAL complex cgrid_value_outside_vortex_y(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nd;

  /* In vortex direction, map i to {0,nx-1}. This does not introduce any phase */
  nd = ny * 2;
  if (j < 0) j = -j - 1;
  if (j >= nd) j %= nd;
  if (j >= ny) j = nd - 1 - j;

  /* First, map i,j to {0,2*nx-1} range
   * (this does not introduce any phase)
   */
  nd = nz * 2;
  if (k < 0) k = k % nd + nd;
  if (k >= nd) k %= nd;

  nd = nx * 2;
  if (i < 0 ) i = i % nd + nd;
  if (i >= nd) i %= nd;

  /* Then, if i has to be mapped to {0,nx-1} return -phi*
   *       if j has to be mapped to {0,ny-1} return +phi*
   *       if both have to be mapped return -phi
   */
  if(k >= nz) {
    k = 2*nz - 1 - k;
    if(i >= nx) {
      i = nd - 1 - i;
      return -cgrid_value_at_index(grid, i, j, k);
    } else return -CONJ(cgrid_value_at_index(grid, i, j, k));
  } else {
    if(i >= nx) {
      i = nd - 1 - i;
      return CONJ(cgrid_value_at_index(grid, i, j, k));
    } else return cgrid_value_at_index(grid, i, j, k);
  }
}

/*
 * Vortex boundary condition (z).
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

EXPORT REAL complex cgrid_value_outside_vortex_z(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nd;

  /* In vortex direction, map k to {0,nx-1}. This does not introduce any phase */
  nd = nz * 2;
  if (k < 0) k = -k - 1;
  if (k >= nd) k %= nd;
  if (k >= nz) k = nd - 1 - k;
  
  /* First, map i,j to {0,2*nx-1} range
   * (this does not introduce any phase)
   */
  nd = nx * 2;
  if (i < 0) i = i%nd + nd;
  if (i >= nd) i %= nd;

  nd = ny * 2;
  if (j < 0 ) j = j%nd + nd;
  if (j >= nd) j %= nd;

  /* Then, if i has to be mapped to {0,nx-1} return -phi*
   *       if j has to be mapped to {0,ny-1} return +phi*
   *       if both have to be mapped return -phi
   */
  if(i >= nx) {
    i = 2 * nx - 1 - i;
    if(j >= ny) {
      j = nd - 1 - j;
      return -cgrid_value_at_index(grid, i, j, k);
    } else return -CONJ(cgrid_value_at_index(grid, i, j, k));
  } else {
    if(j >= ny) {
      j = nd - 1 - j;
      return CONJ(cgrid_value_at_index(grid, i, j, k));
    } else return cgrid_value_at_index(grid, i, j, k);
  }
}
