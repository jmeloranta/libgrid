/*
 * Function #4: High density correction related function for libdft.
 *
 * Name: dft_ot3d_barranco_op(rho, beta, rhom, C).
 *
 * In ot3d.c this used to be, e.g.,
 * rgrid_operate_one(workspace1, rho, dft_ot3d_barranco_op, otf);
 * and now it is:
 * grid_func4_operate_one(workspace1, rho, otf->beta, otf->rhom, otf->C);
 *
 */

#include "../grid.h"
#include "func4.h"

#ifdef USE_CUDA
extern void grid_func4_cuda_operate_one_productW(CUREAL *, CUREAL *, CUREAL *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
extern void grid_func4_cuda_operate_oneW(CUREAL *, CUREAL *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
#endif

static inline REAL grid_func4(REAL rhop, REAL beta, REAL rhom, REAL C) {

  REAL stanh = TANH(beta * (rhop - rhom));

  return (C * ((1.0 + stanh) + beta * rhop * (1.0 - stanh * stanh)));
}

#ifdef USE_CUDA
EXPORT char grid_func4_cuda_operate_one(rgrid *gridc, rgrid *grida, REAL beta, REAL rhom, REAL C) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;
  grid_func4_cuda_operate_oneW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), beta, rhom, C, grida->nx, grida->ny, grida->nz);
  return 0;
}

EXPORT char grid_func4_cuda_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL beta, REAL rhom, REAL C) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;
  grid_func4_cuda_operate_one_productW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), beta, rhom, C, grida->nx, grida->ny, grida->nz);
  return 0;
}
#endif

EXPORT void grid_func4_operate_one(rgrid *gridc, rgrid *grida, REAL beta, REAL rhom, REAL C) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func4_cuda_operate_one(gridc, grida, beta, rhom, C)) return;
  cuda_remove_block(grida->value, 1);
  cuda_remove_block(gridc->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,cvalue,beta,rhom,C) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = grid_func4(avalue[ijnz + k], beta, rhom, C);
  }
}

EXPORT void grid_func4_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL beta, REAL rhom, REAL C) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func4_cuda_operate_one_product(gridc, gridb, grida, beta, rhom, C)) return;
  cuda_remove_block(gridb->value, 1);
  cuda_remove_block(grida->value, 1);
  cuda_remove_block(gridc->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,beta,rhom,C) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = bvalue[ijnz + k] * grid_func4(avalue[ijnz + k], beta, rhom, C);
  }
}
