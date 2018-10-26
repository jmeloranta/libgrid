/*
 * Function #1: Backflow related function for libdft.
 *
 * Name: dft_ot3d_bf_pi_energy_op2(rho, xi, rhobf). Corresponds to G(rho(r)).
 *
 * In ot3d.c this used to be, e.g., 
 * rgrid_operate_one_product(workspace2, workspace2, density, dft_ot3d_bf_pi_energy_op2, otf);
 * and now it is replacd by:
 * grid_func1_operate_one_product(workspace2, workspace2, density, otf->xi, otf->rhobf);
 *
 */

#include "../grid.h"
#include "func1.h"

#ifdef USE_CUDA
extern void grid_func1_cuda_operate_one_productW(CUREAL *, CUREAL *, CUREAL *, CUREAL, CUREAL, INT, INT, INT);
extern void grid_func1_cuda_operate_oneW(CUREAL *, CUREAL *, CUREAL, CUREAL, INT, INT, INT);
#endif

static inline REAL grid_func1(REAL rhop, REAL xi, REAL rhobf) {

  return FUNCTION;
}

#ifdef USE_CUDA
EXPORT char grid_func1_cuda_operate_one(rgrid *gridc, rgrid *grida, REAL xi, REAL rhobf) {

  if(cuda_two_block_policy(grida->value, grida->grid_len, grida->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;
  grid_func1_cuda_operate_oneW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(grida->value), xi, rhobf, grida->nx, grida->ny, grida->nz);
  return 0;
}

EXPORT char grid_func1_cuda_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL xi, REAL rhobf) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;
  grid_func1_cuda_operate_one_productW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), xi, rhobf, grida->nx, grida->ny, grida->nz);
  return 0;
}
#endif

EXPORT void grid_func1_operate_one(rgrid *gridc, rgrid *grida, REAL xi, REAL rhobf) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func1_cuda_operate_one(gridc, grida, xi, rhobf)) return;
  cuda_remove_block(grida->value, 1);
  cuda_remove_block(gridc->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,cvalue,xi,rhobf) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = grid_func1(avalue[ijnz + k], xi, rhobf);
  }
}

EXPORT void grid_func1_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL xi, REAL rhobf) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func1_cuda_operate_one_product(gridc, gridb, grida, xi, rhobf)) return;
  cuda_remove_block(gridb->value, 1);
  cuda_remove_block(grida->value, 1);
  cuda_remove_block(gridc->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,xi,rhobf) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = bvalue[ijnz + k] * grid_func1(avalue[ijnz + k], xi, rhobf);
  }
}
