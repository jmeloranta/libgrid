/*
 * Routines for mixed complex/real grids.
 *
 * Nx is major index and Nz is minor index (varies most rapidly).
 *
 * For 2-D grids use: (1, NY, NZ)
 * For 1-D grids use: (1, 1, NZ)
 *
 * Note that due to FFT, the last index dimension for real grids is 2 * (nz / 2 + 1) rather than just nz.
 *
 */

#include "grid.h"
#include "private.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

/*
 * Peek at a grid file to get the grid dimensions.
 *
 * fp   = File pointer for operation (FILE *; input).
 * nx   = # of points along x (INT *; input).
 * ny   = # of points along y (INT *; input).
 * nz   = # of points along z (INT *; input).
 * step = spatial step length (REAL *; input).
 *
 * No return value.
 *
 * Notes: - This works for both real and complex grids and hence it is just called
 *          grid_read_peek().
 *        - This rewinds the fp so that Xgrid_read() can be called directly
 *          after this.
 *
 */

EXPORT void grid_read_peek(FILE *fp, INT *nx, INT *ny, INT *nz, REAL *step) {

  fread(nx, sizeof(INT), 1, fp);
  fread(ny, sizeof(INT), 1, fp);
  fread(nz, sizeof(INT), 1, fp);
  fread(step, sizeof(REAL), 1, fp);
  rewind(fp);
}

/*
 * Copy a real grid to a complex grid (to real part).  Note that this zeroes the imaginary part.
 *
 * dest   = destination grid (cgrid *; output).
 * source = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_real_to_complex_re(cgrid *dest, rgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz = source->nz2;
  REAL *src = source->value;
  REAL complex *dst = dest->value;
  
  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  dest->step = source->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_real_to_complex_re(dest, source)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz + k] = (REAL complex) src[ijnz2 + k];
  }
}

/*
 * Copy a real grid to a complex grid (to imaginary part). Note that this zeroes the real part.
 *
 * dest   = destination grid (cgrid *; output).
 * source = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_real_to_complex_im(cgrid *dest, rgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz = source->nz2;
  REAL *src = source->value;
  REAL complex *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_real_to_complex_im(dest, source)) return;
#endif

  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  dest->step = source->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz + k] = I * (REAL complex) src[ijnz2 + k];
  }
}

/*
 * Add a real grid to a complex grid (to real part).
 *
 * dest   = destination grid (cgrid *; output).
 * source = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_add_real_to_complex_re(cgrid *dest, rgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz = source->nz2;
  REAL *src = source->value;
  REAL complex *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_add_real_to_complex_re(dest, source)) return;
#endif

  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  dest->step = source->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz + k] += (REAL complex) src[ijnz2 + k];
  }
}

/*
 * Add a real grid to a complex grid (to imaginary part).
 *
 * dest   = destination grid (cgrid *; output).
 * source = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_add_real_to_complex_im(cgrid *dest, rgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz = source->nz2;
  REAL *src = source->value;
  REAL complex *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_add_real_to_complex_im(dest, source)) return;
#endif

  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  dest->step = source->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz + k] += I * (REAL complex) src[ijnz2 + k];
  }
}

/*
 * Multiply real grid with square norm of complex grid: dest = src1 * |src2|^2
 *
 * dst   = Destination grid (rgrid *; output).
 * src1  = Source grid 1 (rgrid *; input).
 * src2  = Source grid 2 (cgrid *; input).
 * 
 * No return value.
 *
 */

EXPORT void grid_product_norm(rgrid *dst, rgrid *src1, cgrid *src2) {
  
  INT ij, k, nz = dst->nz, nxy = dst->nx * dst->ny, ijnz, ijnz2, nzz = dst->nz2;
  REAL *s1value = src1->value, *dvalue = dst->value;
  REAL complex *s2value = src2->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_product_norm(dst, src1, src2)) return;
#endif

  dst->nx = src1->nx;
  dst->ny = src1->ny;
  dst->nz = src1->nz;
  dst->step = src1->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,s1value,s2value) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz2 + k] = s1value[ijnz2 + k] * sqnorm(s2value[ijnz + k]);
  }
}

/*
 * Divide real grid with square norm of complex grid: dest = src1 / (|src2|^2 + eps)
 *
 * dst   = Destination grid (rgrid *; output).
 * src1  = Source grid 1 (rgrid *; input).
 * src2  = Source grid 2 (cgrid *; input).
 * eps   = Epsilon to add when dividing (REAL; input).
 * 
 * No return value.
 *
 */

EXPORT void grid_division_norm(rgrid *dst, rgrid *src1, cgrid *src2, REAL eps) {
  
  INT ij, k, nz = dst->nz, nxy = dst->nx * dst->ny, ijnz, ijnz2, nzz = dst->nz2;
  REAL *s1value = src1->value, *dvalue = dst->value;
  REAL complex *s2value = src2->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_division_norm(dst, src1, src2, eps)) return;
#endif

  dst->nx = src1->nx;
  dst->ny = src1->ny;
  dst->nz = src1->nz;
  dst->step = src1->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,s1value,s2value,eps) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz2 + k] = s1value[ijnz2 + k] / (sqnorm(s2value[ijnz + k]) + eps);
  }
}

/*
 * Product of a real grid with a complex grid: dest(complex) = dest(complex) * source(real)
 *
 * dst  = destination grid (cgrid *; output).
 * ssrc = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_product_complex_with_real(cgrid *dst, rgrid *src) {
  
  INT ij, k, nz = src->nz, nxy = src->nx * src->ny, ijnz, ijnz2, nzz = src->nz2;
  REAL *svalue = src->value;
  REAL complex *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_product_complex_with_real(dst, src)) return;
#endif

  dst->nx = src->nx;
  dst->ny = src->ny;
  dst->nz = src->nz;
  dst->step = src->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,svalue) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] *= (REAL complex) svalue[ijnz2 + k];
  }
}

/*
 * Copy imaginary part of a complex grid to a real grid.
 *
 * dest   = destination grid (rgrid *; output).
 * source = source grid (cgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_complex_im_to_real(rgrid *dest, cgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz;
  REAL complex *src = source->value;
  REAL *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_complex_im_to_real(dest, source)) return;
#endif

  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  nzz = dest->nz2 = 2 * (source->nz / 2 + 1);
  dest->step = source->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz2 + k] = CIMAG(src[ijnz + k]);
  }
}

/*
 * Copy real part of a complex grid to a real grid.
 *
 * dest   = destination grid (rgrid *; output).
 * source = source grid (cgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_complex_re_to_real(rgrid *dest, cgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz;
  REAL complex *src = source->value;
  REAL *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_complex_re_to_real(dest, source)) return;
#endif

  dest->nx = source->nx;
  dest->ny = source->ny;
  dest->nz = source->nz;
  nzz = dest->nz2 = 2 * (source->nz / 2 + 1);
  dest->step = source->step;
  
#pragma omp parallel for firstprivate(nxy,nz,nzz,dst,src) private(ij,ijnz,ijnz2,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    ijnz2 = ij * nzz;
    for(k = 0; k < nz; k++)
      dst[ijnz2 + k] = CREAL(src[ijnz + k]);
  }
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int opgrid |dgrid|^2).
 *
 * dgrid  = grid giving the probability (cgrid *; input).
 * opgrid = grid to be averaged (rgrid *; input).
 *
 * Returns the average value (REAL *).
 *
 */

EXPORT REAL grid_grid_expectation_value(cgrid *dgrid, rgrid *opgrid) {

  INT i, j, k, nx = dgrid->nx, ny = dgrid->ny, nz = dgrid->nz;
  REAL sum, step = dgrid->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_grid_expectation_value(dgrid, opgrid, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(nx,ny,nz,dgrid,opgrid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += sqnorm(cgrid_value_at_index(dgrid, i, j, k)) * rgrid_value_at_index(opgrid, i, j, k);

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;

  return sum;
}
