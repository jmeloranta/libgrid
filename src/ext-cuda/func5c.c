/*
 * Function #5: High density correction related function for libdft.
 *
 * Name: dft_ot3d_barranco_energy_op(rho, beta, rhom, C).
 *
 * In ot3d.c this used to be, e.g.,
 * rgrid_operate_one(workspace1, density, dft_ot3d_barranco_energy_op, otf);
 * and now it is:
 * grid_func5_operate_one(workspace1, density, otf->beta, otf->rhom, otf->C);
 *
 */

#include "../grid.h"
#include "func5.h"

#ifdef USE_CUDA
extern void grid_func5_cuda_operate_one_productW(gpu_mem_block *, gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
extern void grid_func5_cuda_operate_oneW(gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
#endif

static inline REAL grid_func5(REAL rhop, REAL beta, REAL rhom, REAL C) {

 return (C * (1.0 + TANH(beta * (rhop - rhom))) * rhop);
}

#ifdef USE_CUDA
EXPORT char grid_func5_cuda_operate_one(rgrid *dst, rgrid *src, REAL beta, REAL rhom, REAL C) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;
  grid_func5_cuda_operate_oneW(cuda_block_address(dst->value), cuda_block_address(src->value), beta, rhom, C, dst->nx, dst->ny, dst->nz);

  return 0;
}

EXPORT char grid_func5_cuda_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, REAL beta, REAL rhom, REAL C) {

  if(cuda_three_block_policy(src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) 
    return -1;
  grid_func5_cuda_operate_one_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), beta, rhom, C, dst->nx, dst->ny, dst->nz);

  return 0;
}
#endif

EXPORT void grid_func5_operate_one(rgrid *dst, rgrid *src, REAL beta, REAL rhom, REAL C) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *svalue = src->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func5_cuda_operate_one(dst, src, beta, rhom, C)) return;
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,svalue,beta,rhom,C) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = grid_func5(svalue[ijnz + k], beta, rhom, C);
  }
}

EXPORT void grid_func5_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, REAL beta, REAL rhom, REAL C) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *s2value = src2->value;
  REAL *s1value = src1->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func5_cuda_operate_one_product(dst, src1, src2, beta, rhom, C)) return;
  cuda_remove_block(src1->value, 1);
  cuda_remove_block(src2->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,s1value,s2value,beta,rhom,C) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = s1value[ijnz + k] * grid_func5(s2value[ijnz + k], beta, rhom, C);
  }
}
