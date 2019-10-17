/*
 * Precomputed function CUDA driver (REAL).
 *
 */

#include "grid.h"

EXPORT char rgrid_cuda_function_operate_one(rgrid *dst, rgrid *src, rfunction *operator) {

  if(dst->host_lock || src->host_lock) {
    cuda_remove_block(src->value, 1);
    cuda_remove_block(dst->value, 0);
    cuda_remove_block(operator->value, 1);
    return -1;
  }

  if(cuda_three_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, operator->value, operator->length, -1, operator->id, 1,
                             dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_function_operate_oneW(cuda_block_address(dst->value), cuda_block_address(src->value), cuda_block_address(operator->value), src->nx, src->ny, src->nz, operator->begin, operator->nsteps, operator->step);

  return 0;
}

EXPORT char rgrid_cuda_function_operate_one_product(rgrid *dst, rgrid *src1, rgrid *src2, rfunction *operator) {

  if(dst->host_lock || src1->host_lock || src2->host_lock) {
    cuda_remove_block(src1->value, 1);
    cuda_remove_block(src2->value, 1);
    cuda_remove_block(dst->value, 0);
    cuda_remove_block(operator->value, 1);
    return -1;
  }

  if(cuda_four_block_policy(src1->value, src1->grid_len, src1->cufft_handle_r2c, src1->id, 1, src2->value, src2->grid_len, src2->cufft_handle_r2c, src2->id, 1, 
                            operator->value, operator->length, -1, operator->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;

  rgrid_cuda_function_operate_one_productW(cuda_block_address(dst->value), cuda_block_address(src1->value), cuda_block_address(src2->value), cuda_block_address(operator->value), 
                                           src1->nx, src1->ny, src1->nz, operator->begin, operator->nsteps, operator->step);

  return 0;
}
