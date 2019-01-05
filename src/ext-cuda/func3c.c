/*
 * Function #3: Backflow related function for libdft.
 *
 * Evaluate: rho * (dG/rho) + G(rho).
 *
 */

#include "../grid.h"
#include "func3.h"

#ifdef USE_CUDA
extern void grid_func3_cuda_operate_one_productW(CUREAL *, CUREAL *, CUREAL *, CUREAL, CUREAL, INT, INT, INT);
#endif

/* rho * (dG/rho) + G(rho) */
static inline REAL grid_func3(REAL rhop, REAL xi, REAL rhobf) {

  REAL tmp = 1.0 / (COSH((rhop - rhobf) * xi) + DFT_BF_EPS);
  REAL tmp2 = 0.5 * (1.0 - TANH(xi * (rhop - rhobf)));
  
  return rhop * (-0.5 * xi * tmp * tmp) + tmp2;
}

#ifdef USE_CUDA
EXPORT char grid_func3_cuda_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL xi, REAL rhobf) {

  if(cuda_three_block_policy(grida->value, grida->grid_len, grida->id, 1, gridb->value, gridb->grid_len, gridb->id, 1, gridc->value, gridc->grid_len, gridc->id, 0) < 0) return -1;
  grid_func3_cuda_operate_one_productW((CUREAL *) cuda_block_address(gridc->value), (CUREAL *) cuda_block_address(gridb->value), (CUREAL *) cuda_block_address(grida->value), xi, rhobf, grida->nx, grida->ny, grida->nz);
  return 0;
}
#endif

/*
 * Multiply grid by rho * (dG/drho) + G(rho): gridc = gridb * (grida * (dG/drho)(grida) + G(grida)).
 *
 * gridc = Destination grid (rgrid *; output).
 * gridb = Source grid for multiplication (rgrid *; input).
 * grida = Source grid for G() (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_func3_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL xi, REAL rhobf) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func3_cuda_operate_one_product(gridc, gridb, grida, xi, rhobf)) return;
  cuda_remove_block(gridb->value, 1);
  cuda_remove_block(grida->value, 1);
  cuda_remove_block(gridc->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,xi,rhobf) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = bvalue[ijnz + k] * grid_func3(avalue[ijnz + k], xi, rhobf);
  }
}
