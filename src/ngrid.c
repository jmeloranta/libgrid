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
 * Product of a real grid with a complex grid: dest(complex) = dest(complex) * source(real)
 *
 * dest   = destination grid (cgrid *; output).
 * source = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_product_complex_with_real(cgrid *dest, rgrid *source) {
  
  INT ij, k, nz = source->nz, nxy = source->nx * source->ny, ijnz, ijnz2, nzz = source->nz2;
  REAL *src = source->value;
  REAL complex *dst = dest->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_product_complex_with_real(dest, source)) return;
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
      dst[ijnz + k] *= (REAL complex) src[ijnz2 + k];
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
