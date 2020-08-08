/*
 * Function #7: External potential function:
 *              V(R) = A0*exp(-A1*R) - A2 / R^3 - A3 / R^6 - A4 / R^8 - A5 / R^10
 *
 * Name: func7a = Add the above potential function to a given complex grid.
 *       func7b = First derivative with respect to x of the above potential function.
 *       func7c = First derivative with respect to x of the above potential function.
 *       func7d = First derivative with respect to x of the above potential function.
 *
 * rmin is the minimum distance where the potential will be evaluated and radd is a constant
 * that is subtracted from r (-> shifts the potential to larger distances).
 * 
 */

#include "../grid.h"
#include "func7.h"

#ifdef USE_CUDA
extern void grid_func7a_cuda_operate_oneW(gpu_mem_block *, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, INT, INT, INT, CUREAL, CUREAL, CUREAL, CUREAL);
extern void grid_func7b_cuda_operate_oneW(gpu_mem_block *, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, INT, INT, INT, CUREAL, CUREAL, CUREAL, CUREAL);
extern void grid_func7c_cuda_operate_oneW(gpu_mem_block *, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, INT, INT, INT, CUREAL, CUREAL, CUREAL, CUREAL);
extern void grid_func7d_cuda_operate_oneW(gpu_mem_block *, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, CUREAL, INT, INT, INT, CUREAL, CUREAL, CUREAL, CUREAL);
#endif

/************ Potential function ************************/

static inline REAL grid_func7a(REAL r, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL ri, r2, r4, r6, r8, r10;

  r -= radd;
  if(r < rmin) r = rmin;

  ri = 1.0 / r;
  r2 = ri * ri;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return a0 * EXP(-a1 * r) - a2 * r4 - a3 * r6 - a4 * r8 - a5 * r10;
}

#ifdef USE_CUDA
EXPORT char grid_func7a_cuda_operate_one(cgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL x0, y0, z0;

  if(cuda_one_block_policy(dst->value, dst->grid_len, dst->cufft_handle, dst->id, 1) < 0) return -1;
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
  grid_func7a_cuda_operate_oneW(cuda_find_block(dst->value), rmin, radd, a0, a1, a2, a3, a4, a5, dst->nx, dst->ny, dst->nz, x0, y0, z0, dst->step);

  return 0;
}
#endif

EXPORT void grid_func7a_operate_one(cgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  INT i, j, ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, ny = dst->ny;
  INT nx2 = dst->nx / 2, ny2 = dst->ny / 2, nz2 = dst->nz / 2;
  REAL complex *value = dst->value;
  REAL x, y, z, r, x0, y0, z0, step = dst->step, x2, y2;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func7a_cuda_operate_one(dst, rmin, radd, a0, a1, a2, a3, a4, a5)) return;
  cuda_remove_block(dst->value, 1);
#endif
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
#pragma omp parallel for firstprivate(nxy,nz,value,rmin,radd,a0,a1,a2,a3,a4,a5,x0,y0,z0,step,nx2,ny2,nz2,ny) private(i,j,ij,ijnz,k,x,y,z,r,x2,y2) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    x2 = x * x;
    y = ((REAL) (j - ny2)) * step - y0;    
    y2 = y * y;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;      
      r = SQRT(x2 + y2 + z*z);      
      value[ijnz + k] += (REAL complex) grid_func7a(rmin, radd, r, a0, a1, a2, a3, a4, a5);
    }
  }
}

/************* d/dx of potential function *******************/

static inline REAL grid_func7b(REAL rp, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL r, ri, r2, r3, r5, r7, r9, r11;

  r = rp - radd;
  if(r < rmin) return 0.0;

  ri = 1.0 / r;
  r2 = ri * ri;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (-a0 * a1 * EXP(-a1 * r) + 4.0 * a2 * r5 + 6.0 * a3 * r7 + 8.0 * a4 * r9 + 10.0 * a5 * r11) / rp;
}

#ifdef USE_CUDA
EXPORT char grid_func7b_cuda_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL x0, y0, z0;

  if(cuda_one_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 1) < 0) return -1;
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
  grid_func7b_cuda_operate_oneW(cuda_find_block(dst->value), rmin, radd, a0, a1, a2, a3, a4, a5, dst->nx, dst->ny, dst->nz, x0, y0, z0, dst->step);

  return 0;
}
#endif

EXPORT void grid_func7b_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  INT i, j, ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2, ny = dst->ny;
  INT nx2 = dst->nx / 2, ny2 = dst->ny / 2, nz2 = dst->nz / 2;
  REAL *value = dst->value, x, y, z, r, x0, y0, z0, step = dst->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func7b_cuda_operate_one(dst, rmin, radd, a0, a1, a2, a3, a4, a5)) return;
  cuda_remove_block(dst->value, 1);
#endif
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,rmin,radd,a0,a1,a2,a3,a4,a5,x0,y0,z0,step,nx2,ny2,nz2,ny) private(i,j,ij,ijnz,k,x,y,z,r) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;      
      r = SQRT(x*x + y*y + z*z);      
      value[ijnz + k] = x * grid_func7b(rmin, radd, r, a0, a1, a2, a3, a4, a5);
    }
  }
}

/************* d/dy of potential function *******************/

#ifdef USE_CUDA
EXPORT char grid_func7c_cuda_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL x0, y0, z0;

  if(cuda_one_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 1) < 0) return -1;
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
  grid_func7c_cuda_operate_oneW(cuda_find_block(dst->value), rmin, radd, a0, a1, a2, a3, a4, a5, dst->nx, dst->ny, dst->nz, x0, y0, z0, dst->step);

  return 0;
}
#endif

EXPORT void grid_func7c_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  INT i, j, ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2, ny = dst->ny;
  INT nx2 = dst->nx / 2, ny2 = dst->ny / 2, nz2 = dst->nz / 2;
  REAL *value = dst->value, x, y, z, r, x0, y0, z0, step = dst->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func7c_cuda_operate_one(dst, rmin, radd, a0, a1, a2, a3, a4, a5)) return;
  cuda_remove_block(dst->value, 1);
#endif
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,rmin,radd,a0,a1,a2,a3,a4,a5,x0,y0,z0,step,nx2,ny2,nz2,ny) private(i,j,ij,ijnz,k,x,y,z,r) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;      
      r = SQRT(x*x + y*y + z*z);      
      value[ijnz + k] = y * grid_func7b(rmin, radd, r, a0, a1, a2, a3, a4, a5);
    }
  }
}

/************* d/dz of potential function *******************/

#ifdef USE_CUDA
EXPORT char grid_func7d_cuda_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  REAL x0, y0, z0;

  if(cuda_one_block_policy(dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 1) < 0) return -1;
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
  grid_func7d_cuda_operate_oneW(cuda_find_block(dst->value), rmin, radd, a0, a1, a2, a3, a4, a5, dst->nx, dst->ny, dst->nz, x0, y0, z0, dst->step);

  return 0;
}
#endif

EXPORT void grid_func7d_operate_one(rgrid *dst, REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  INT i, j, ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2, ny = dst->ny;
  INT nx2 = dst->nx / 2, ny2 = dst->ny / 2, nz2 = dst->nz / 2;
  REAL *value = dst->value, x, y, z, r, x0, y0, z0, step = dst->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func7b_cuda_operate_one(dst, rmin, radd, a0, a1, a2, a3, a4, a5)) return;
  cuda_remove_block(dst->value, 1);
#endif
  x0 = dst->x0;
  y0 = dst->y0;
  z0 = dst->z0;
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,rmin,radd,a0,a1,a2,a3,a4,a5,x0,y0,z0,step,nx2,ny2,nz2,ny) private(i,j,ij,ijnz,k,x,y,z,r) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;      
      r = SQRT(x*x + y*y + z*z);      
      value[ijnz + k] = z * grid_func7b(rmin, radd, r, a0, a1, a2, a3, a4, a5);
    }
  }
}
