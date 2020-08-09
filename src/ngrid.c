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
#include "cprivate.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

/*
 * @FUNC{grid_read_peek, "Get grid dimensions from file"}
 * @DESC{"Peek at a grid file to get the grid dimensions. This works for both 
          real and complex grids and hence it is just called grid_read_peek().
          This rewinds the file pointer so that Xgrid_read() can be called directly after this"}
 * @ARG1{FILE *fp, "File pointer for operation"}
 * @ARG2{INT *nx, "Number of points along x"}
 * @ARG3{INT *ny, "Number of points along y"}
 * @ARG4{INT *nz, "Number of points along z"}
 * @ARG5{REAL *step, "Spatial step length"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_real_to_complex_re, "Copy real grid to real part of complex grid"}
 * @DESC{"Copy a real grid to a complex grid (to real part). Note that this zeroes the imaginary part"}
 * @ARG1{cgrid *dest, "Destination grid"}
 * @ARG2{rgrid *source, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_real_to_complex_im, "Copy real grid to imaginary part of complex grid"}
 * @DESC{"Copy a real grid to a complex grid (to imaginary part). Note that this zeroes the real part"}
 * @ARG1{cgrid *dest, "Destination grid"}
 * @ARG2{rgrid *source, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_add_real_to_complex_re, "Add real grid to real part of complex grid"}
 * @DESC{"Add a real grid to a complex grid (to real part)"}
 * @ARG1{cgrid *dest, "Destination grid"}
 * @ARG2{rgrid *source, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_add_real_to_complex_im, "Add real grid to imaginary part of complex grid"}
 * @DESC{"Add a real grid to a complex grid (to imaginary part)"}
 * @ARG1{cgrid *dest, "Destination grid"}
 * @ARG2{rgrid *source, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_product_norm, "Multiply real grid with square norm of complex grid"}
 * @DESC{"Multiply real grid with square norm of complex grid: dest = src1 * $|src2|^2$"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src1, "Real source grid"}
 * @ARG3{cgrid *src2, "Complex source grid"}
 * @RVAL{void, "No return value"}
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
      dvalue[ijnz2 + k] = s1value[ijnz2 + k] * csqnorm(s2value[ijnz + k]);
  }
}

/*
 * @FUNC{grid_division_norm, "Divide real grid with square norm of complex grid"}
 * @DESC{"Divide real grid with square norm of complex grid: dest = src1 / ($|src2|^2$ + eps)"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src1, "Real grid"}
 * @ARG3{cgrid *src2, "Complex grid"}
 * @ARG4{REAL eps, "Epsilon to add when dividing"}
 * @RVAL{void, "No return value"}
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
      dvalue[ijnz2 + k] = s1value[ijnz2 + k] / (csqnorm(s2value[ijnz + k]) + eps);
  }
}

/*
 * @FUNCgrid_product_complex_with_real, "Product of real grid with complex grid"}
 * @DESC{"Product of a real grid with a complex grid: dest(complex) = dest(complex) * source(real)"}
 * @ARG1{cgrid *dst, "Destination grid (complex)"}
 * @ARG2{rgrid *src, "Source grid (real)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_complex_im_to_real, "Copy imaginary part of complex grid to real grid"}
 * @DESC{"Copy imaginary part of a complex grid to a real grid"}
 * @ARG1{rgrid *dest, "Destination grid (real)"}
 * @ARG2{cgrid *source, "Source grid (complex)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_complex_re_to_real, "Copy real part of complex grid to real grid"}
 * @DESC{"Copy real part of a complex grid to a real grid"}x
 * @ARG1{rgrid *dest, "Destination grid (real)"}
 * @ARG2{cgrid *source, "Source grid (complex)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_grid_expectation_value, "Expectation value of real grid over complex grid"}
 * @DESC{"Calculate the expectation value of a grid over a grid ($\int opgrid |dgrid|^2$)"}
 * @ARG1{cgrid *dgrid, "Grid giving the probability (complex)"}
 * @ARG2{rgrid *opgrid, "Grid to be averaged (real)"}
 * @RVAL{REAL, "Returns the average value"}
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
	sum += csqnorm(cgrid_value_at_index(dgrid, i, j, k)) * rgrid_value_at_index(opgrid, i, j, k);

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;

  return sum;
}

/*
 * @FUNC{grid_sizeof_real_complex, "Return current size of REAL complex data type"}
 * @DESC{"Return current size of REAL complex in bytes"}
 * @RVAL{char, "Size of REAL complex type"}
 *
 */

EXPORT char grid_sizeof_real_complex() {

  return (char) sizeof(REAL complex);
}

/*
 * @FUNC{grid_sizeof_real, "Return current size of REAL data type"}
 * @DESC{"Return current size of REAL in bytes"}
 * @RVAL{char, "Size of REAL type"}
 *
 */

EXPORT char grid_sizeof_real() {

  return (char) sizeof(REAL);
}

/*
 * @FUNC{grid_sizeof_int, "Return current size of INT data type"}
 * @DESC{"Return current size of INT in bytes"}
 * @RVAL{char, "Size of INT type"}
 *
 */

EXPORT char grid_sizeof_int() {

  return (char) sizeof(INT);
}
