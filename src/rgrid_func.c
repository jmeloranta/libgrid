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
 * @FUNC{rgrid_function_alloc, "Allocate space for precomputed function object"} 
 * @DESC{"Allocate and initialize precomputed 1-D function object.
          The function takes the x value (REAL) and user data (void *) as arguments"}
 * @ARG1{REAL (*func), "Function to be mapped"}
 * @ARG2{void *params, "Additional parameters for func"}
 * @ARG3{REAL begin, "Lowest x value"}
 * @ARG4{REAL end, "Highest x value"}
 * @ARG5{REAL step, "Step for x"}
 * @ARG6{char *id, "ID string identifying the function"}
 * @RVAL{rfunction *, "Returns pointer to the precomputed function object"}
 *
 */

EXPORT rfunction *rgrid_function_alloc(REAL (*func)(REAL, void *), void *params, REAL begin, REAL end, REAL step, char *id) {

  INT i, nsteps = 1 + (INT) ((end - begin) / step);
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
 * @FUNC{rgrid_function_free, "Free precomputed function object"}
 * @DESC{"Free precomputed function object"}
 * @ARG1{rfunctin *, "Precomputed function to be freed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_function_value, "Return value for precomputed function"}
 * @DESC{"Return precomputed function value at given value (no interpolation)"}
 * @ARG1{rfunction *, "Precomputed function"}
 * @ARG2{REAL x, "Value where the function is evaluated"}
 * @RVAL{REAL, "Returns the function value at x"}
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
 * @FUNC{rgrid_function_operate_one, "Operate on grid by precomputed function"}
 * @DESC{"Operate on a grid by precomputed function: dst = operator(src).
          Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid"}
 * @ARG3{rfunction *operator, "Function defining the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_function_operate_one_product, "Operate on grid by precomputed function and multiply"}
 * @DESC{"Operate on a grid by a given precomputed function and multiply: dst = src1 * operator(src2).
          Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src1, "Multiply with this grid"}
 * @ARG3{rgrid *src2, "Source grid entering the precomputed function"}
 * @ARG4{rfunction *operator, "Operator (function)"}
 * @RVAL{void, "No return value"}
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
