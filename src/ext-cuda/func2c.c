/*
 * Function #2: Backflow related function for libdft.
 *
 * Evaluate: rho * G(rho)
 *
 */

#include "../grid.h"
#include "func2.h"

#ifdef USE_CUDA
extern void grid_func2_cuda_operate_one_productW(gpu_mem_block *, gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, INT, INT, INT);
extern void grid_func2_cuda_operate_oneW(gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, INT, INT, INT);
#endif

/* rho * G(rho) */
static inline REAL grid_func2(REAL rhop, REAL xi, REAL rhobf) {

  return FUNCTION;
}

#ifdef USE_CUDA
EXPORT char grid_func2_cuda_operate_one(rgrid *dst, rgrid *src, REAL xi, REAL rhobf) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;
  grid_func2_cuda_operate_oneW(cuda_block_address(dst->value), cuda_block_address(src->value), xi, rhobf, dst->nx, dst->ny, dst->nz);

  return 0;
}

EXPORT char grid_func2_cuda_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, REAL xi, REAL rhobf) {

  if(cuda_three_block_policy(src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;
  grid_func2_cuda_operate_one_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), xi, rhobf, dst->nx, dst->ny, dst->nz);

  return 0;
}
#endif

/*
 * Evaluate rho * G(rho(r)): dst = src * G(src).
 *
 * dst = Destination grid (rgrid *; output).
 * src = Source grid (rgrid *; input)
 *
 * No return value.
 *
 */

EXPORT void grid_func2_operate_one(rgrid *dst, rgrid *src, REAL xi, REAL rhobf) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *svalue = src->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func2_cuda_operate_one(dst, src, xi, rhobf)) return;
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,svalue,xi,rhobf) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = grid_func2(svalue[ijnz + k], xi, rhobf);
  }
}

/*
 * Multiply by rho * G(rho(r)): dst = src1 * G(src2).
 *
 * dst  = Destination grid (rgrid *; output).
 * src1 = Source grid for multiplication (rgrid *; input).
 * src2 = Source grid for rho*G(rho) (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_func2_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, REAL xi, REAL rhobf) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *s2value = src2->value;
  REAL *s1value = src1->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func2_cuda_operate_one_product(dst, src1, src2, xi, rhobf)) return;
  cuda_remove_block(src1->value, 1);
  cuda_remove_block(src2->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,s1value,s2value,xi,rhobf) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = s1value[ijnz + k] * grid_func2(s2value[ijnz + k], xi, rhobf);
  }
}
