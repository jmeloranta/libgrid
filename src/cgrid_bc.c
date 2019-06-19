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

/*
 * For the following routines, see section 2.5.2 (Real even/odd DFTs (cosine/sine transforms)) in FFTW manual.
 * FFTW_REDFT10 etc.
 *
 * These have been only implemented for CPUs (not GPUs).
 *
 */

/*
 * FFT fully even boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_eee(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  
  /* Even with respect to -0.5 and N - 0.5 */
  if (i < 0) i = -i - 1;
  if (i > nx-1) i = i - nx - 1;

  /* Even with respect to -0.5 and N - 0.5 */
  if (j < 0) j = -j - 1;
  if (j > ny-1) j = j - ny - 1;

  /* Even with respect to -0.5 and N - 0.5 */
  if (k < 0) k = -k - 1;
  if (k > nz-1) k = k - nz - 1;

  return cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT odd(x)/even(y)/even(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_oee(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Odd with respect to -0.5 and N - 0.5 */
  if (i < 0) {i = -i - 1; m *= -1.0;}
  if (i > nx-1) {i = i - nx - 1; m *= -1.0;}

  /* Even with respect to -0.5 and N - 0.5 */
  if (j < 0) j = -j - 1;
  if (j > ny-1) j = j - ny - 1;

  /* Even with respect to -0.5 and N - 0.5 */
  if (k < 0) k = -k - 1;
  if (k > nz-1) k = k - nz - 1;

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT even(x)/odd(y)/even(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_eoe(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Even with respect to -0.5 and N - 0.5 */
  if (i < 0) i = -i - 1;
  if (i > nx-1) i = i - nx - 1;

  /* Odd with respect to -0.5 and N - 0.5 */
  if (j < 0) {j = -j - 1; m *= -1.0;}
  if (j > ny-1) {j = j - ny - 1; m *= -1.0;}

  /* Even with respect to -0.5 and N - 0.5 */
  if (k < 0) k = -k - 1;
  if (k > nz-1) k = k - nz - 1;

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT even(x)/even(y)/odd(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_eeo(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Even with respect to -0.5 and N - 0.5 */
  if (i < 0) i = -i - 1;
  if (i > nx-1) i = i - nx - 1;

  /* Even with respect to -0.5 and N - 0.5 */
  if (j < 0) j = -j - 1;
  if (j > ny-1) j = j - ny - 1;

  /* Odd with respect to -0.5 and N - 0.5 */
  if (k < 0) {k = -k - 1; m *= -1.0;}
  if (k > nz-1) {k = k - nz - 1; m *= -1.0;}

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT odd(x)/odd(y)/even(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_ooe(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Odd with respect to -0.5 and N - 0.5 */
  if (i < 0) {i = -i - 1; m *= -1.0;}
  if (i > nx-1) {i = i - nx - 1; m *= -1.0;}

  /* Odd with respect to -0.5 and N - 0.5 */
  if (j < 0) {j = -j - 1; m *= -1.0;}
  if (j > ny-1) {j = j - ny - 1; m *= -1.0;}

  /* Even with respect to -0.5 and N - 0.5 */
  if (k < 0) k = -k - 1;
  if (k > nz-1) k = k - nz - 1;

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT even(x)/odd(y)/odd(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_eoo(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Even with respect to -0.5 and N - 0.5 */
  if (i < 0) i = -i - 1;
  if (i > nx-1) i = i - nx - 1;

  /* Odd with respect to -0.5 and N - 0.5 */
  if (j < 0) {j = -j - 1; m *= -1.0;}
  if (j > ny-1) {j = j - ny - 1; m *= -1.0;}

  /* Odd with respect to -0.5 and N - 0.5 */
  if (k < 0) {k = -k - 1; m *= -1.0;}
  if (k > nz-1) {k = k - nz - 1; m *= -1.0;}

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT odd(x)/even(y)/odd(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_oeo(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Odd with respect to -0.5 and N - 0.5 */
  if (i < 0) {i = -i - 1; m *= -1.0;}
  if (i > nx-1) {i = i - nx - 1; m *= -1.0;}

  /* Even with respect to -0.5 and N - 0.5 */
  if (j < 0) j = -j - 1;
  if (j > ny-1) j = j - ny - 1;

  /* Odd with respect to -0.5 and N - 0.5 */
  if (k < 0) {k = -k - 1; m *= -1.0;}
  if (k > nz-1) {k = k - nz - 1; m *= -1.0;}

  return m * cgrid_value_at_index(grid, i, j, k);
}

/*
 * FFT odd(x)/odd(y)/odd(z) boundary condition. The symmetry points are -0.5 and N - 0.5.
 * This is to be used with fast sin/cos transforms (special boundary conditions).
 *
 * grid = grid to be evaluated (cgrid *; input).
 * i    = grid index (INT; input).
 * j    = grid index (INT; input).
 * k    = grid index (INT; input).
 *
 * Returns grid value subject to the boundary condition.
 * 
 * NOTE: Does not roll over properly with respect to indices.
 *
 */

EXPORT REAL complex cgrid_value_outside_fft_ooo(cgrid *grid, INT i, INT j, INT k) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL m = 1.0;
  
  /* Odd with respect to -0.5 and N - 0.5 */
  if (i < 0) {i = -i - 1; m *= -1.0;}
  if (i > nx-1) {i = i - nx - 1; m *= -1.0;}

  /* Odd with respect to -0.5 and N - 0.5 */
  if (j < 0) {j = -j - 1; m *= 1.0;}
  if (j > ny-1) {j = j - ny - 1; m *= -1.0;}

  /* Odd with respect to -0.5 and N - 0.5 */
  if (k < 0) {k = -k - 1; m *= -1.0;}
  if (k > nz-1) {k = k - nz - 1; m *= -1.0;}

  return m * cgrid_value_at_index(grid, i, j, k);
}
