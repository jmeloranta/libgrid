/*
 * Precomputed function routines (REAL).
 *
 * The purpose of these functions is to avoid using hard coded functions such as those in ext-cuda directory.
 *
 */

#include "grid.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

/*
 * Allocate and initialize precomputed function object.
 *
 * func   = Function to be mapped (REAL (*)(REAL, void *); input).
 * params = Additional parameters for func (void *; input).
 * begin  = Lowest x value (REAL; input).
 * step   = Step for x (REAL; input).
 * nsteps = Number of steps (INT; input).
 * id     = ID string identifying the function (char *; input).
 *
 * Returns pointer to the precomputed function object (rfunction *).
 *
 */

EXPORT rfunction *rgrid_functon_alloc(REAL (*func)(REAL, void *), void *params, REAL begin, REAL step, INT nsteps, char *id) {

  INT i;
  rfunction *ptr;

  if(!(ptr = (rfunction *) malloc(sizeof(rfunction)))) {
    fprintf(stderr, "libgrid: Out of memory in rgrid_function_alloc.\n");
    abort();
  }
  
  ptr->begin = begin;
  ptr->step = step;
  ptr->nsteps = nsteps;
  strcpy(ptr->id, id);
  ptr->length = sizeof(REAL) * (size_t) nsteps;

  if(!(ptr->value = (REAL *) malloc(ptr->length))) {
    fprintf(stderr, "libgrid: Out of memory in rgrid_function_alloc.\n");
    abort();
  }

  for (i = 0; i < nsteps; i++)
    ptr->value[i] = (*func)(begin + step * (REAL) i, params);

  return ptr;
}

/*
 * Free precomputed function object.
 *
 * pfunc = Precomputed function to be freed (rfunction *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_function_free(rfunction *pfunc) {

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(pfunc->value, 0);
#endif
  free(pfunc->value);
  free(pfunc);
}

/*
 * Return precomputed function value at given value (no interpolation). Not to be called directly.
 *
 * pfunc = Precomputed function (rfunction *; input).
 * x     = Value where the function is evaluated (REAL; input).
 * 
 * Returns the function value.
 *
 */

inline REAL rgrid_function_value(rfunction *pfunc, REAL x) {

  INT i;

  i = (INT) ((x - pfunc->begin) / pfunc->step);

  if(i < 0) i = 0;
  if(i >= pfunc->nsteps) i = pfunc->nsteps - 1;

  return pfunc->value[i];    
}

/*
 * Operate on a grid by precomputed function: dst = operator(src).
 *
 * dst      = Destination grid (rgrid *; output).
 * src      = Source grid (rgrid *; input).
 * operator = Operator grid (rfunction *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_function_operate_one(rgrid *dst, rgrid *src, rfunction *operator) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *asrc = src->value;
  REAL *adst = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_function_operate_one(dst, src, operator)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,adst,asrc,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      adst[ijnz + k] = rgrid_function_value(operator, asrc[ijnz + k]);
  }
}

/*
 * Operate on a grid by a given precomputed function and multiply: dst = src1 * operator(src2).
 *
 * dst      = Destination grid (rgrid *; output).
 * src1     = Multiply with this grid (rgrid *; input).
 * src2     = Source grid for precomputed function (rfunction *; input).
 * operator = Operator (function) (rfunction *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_function_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, rfunction *operator) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *adst = dst->value;
  REAL *asrc1 = src1->value;
  REAL *asrc2 = src2->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_function_operate_one_product(dst, src1, src2, operator)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,adst,asrc1,asrc2,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      adst[ijnz + k] = asrc1[ijnz + k] * rgrid_function_value(operator, asrc2[ijnz + k]);
  }
}
