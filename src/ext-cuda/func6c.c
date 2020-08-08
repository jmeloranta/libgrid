/*
 * Function #6: Thermal corrections for libdft (two functions).
 *
 * Name: func6a = dft_common_bose_idealgas_dEdRho(REAL rhop, void *params)
 *       func6b = dft_common_bose_idealgas_energy(REAL rhop, void *params)
 *
 * In ot3d.c this used to be, e.g.,
 * rgrid_operate_one(workspace1, rho, dft_common_bose_idealgas_dEdRho, (void *) otf);
 * and now it is:
 * grid_func6a_operate_one(workspace1, rho, otf->mass, otf->temp, otf->c4);
 *
 * OR 
 * 
 * In ot3d.c this used to be, e.g.,
 * rgrid_operate_one(workspace1, density, dft_common_bose_idealgas_energy, otf);
 * and now it is:
 * grid_func6b_operate_one(workspace1, density, otf->mass, otf->temp, otf->c4);
 *
 */

#include "../grid.h"
#include "func6.h"

#ifdef USE_CUDA
extern void grid_func6a_cuda_operate_oneW(gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
extern void grid_func6b_cuda_operate_oneW(gpu_mem_block *, gpu_mem_block *, CUREAL, CUREAL, CUREAL, INT, INT, INT);
#endif

static inline REAL lwl3(REAL mass, REAL temp) {

  REAL lwl;

  /* hbar = 1 */
  lwl = SQRT(2.0 * M_PI * HBAR * HBAR / (mass * GRID_AUKB * temp));
  return lwl * lwl * lwl;
}

/* # of terms to evaluate in the series */
#define NTERMS 256

static REAL gf(REAL z, REAL s) {

  REAL val = 0.0, zk = 1.0;
  INT k;

  for (k = 1; k <= NTERMS; k++) {
    zk *= z;
    val += zk / POW((REAL) k, s);
  }
  return val;
}

static inline REAL find_z0(REAL val) {

  /* Golden sectioning */
  REAL a, b, c, d, fc, fd, tmp;

  if(val >= gf(1.0, 3.0 / 2.0)) return 1.0; /* g_{3/2}(1) */

  a = 0.0;
  b = 1.0;

  c = b - (b - a) / GOLDEN;
  d = a + (b - a) / GOLDEN;

  while (FABS(c - d) > STOP) {

    tmp = val - gf(c, 3.0 / 2.0);
    fc = tmp * tmp;
    tmp = val - gf(d, 3.0 / 2.0);
    fd = tmp * tmp;

    if(fc < fd) b = d; else a = c;

    c = b - (b - a) / GOLDEN;
    d = a + (b - a) / GOLDEN;
  }
  return (b + a) / 2.0;       
}

static inline REAL grid_func6a(REAL rhop, REAL mass, REAL temp, REAL c4) {

  REAL l3, z0, rl3, g12, g32;
  REAL tmp;

  l3 = lwl3(mass, temp);
  rl3 = rhop * l3;
  tmp = gf(1.0, 3.0 / 2.0);
  if(rl3 >= tmp) return -c4 * GRID_AUKB * (temp / l3) * tmp;
  z0 = find_z0(rl3);    /* check these fits */
  g12 = gf(z0, 1.0/2.0);
  g32 = gf(z0, 3.0/2.0);

  return c4 * GRID_AUKB * temp * (LOG(z0) + rl3 / g12 - g32 / g12);
}

#ifdef USE_CUDA
EXPORT char grid_func6a_cuda_operate_one(rgrid *dst, rgrid *src, REAL mass, REAL temp, REAL c4) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;
  grid_func6a_cuda_operate_oneW(cuda_find_block(dst->value), cuda_find_block(src->value), mass, temp, c4, dst->nx, dst->ny, dst->nz);

  return 0;
}
#endif

EXPORT void grid_func6a_operate_one(rgrid *dst, rgrid *src, REAL mass, REAL temp, REAL c4) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *svalue = src->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func6a_cuda_operate_one(dst, src, mass, temp, c4)) return;
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,svalue,mass,temp,c4) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = grid_func6a(svalue[ijnz + k], mass, temp, c4);
  }
}

static inline REAL grid_func6b(REAL rhop, REAL mass, REAL temp, REAL c4) {

  REAL z, l3;

  l3 = lwl3(mass, temp);
  z = find_z0(rhop * l3);
  return (c4 * GRID_AUKB * temp * (rhop * LOG(z) - gf(z, 5.0 / 2.0) / l3));
}

#ifdef USE_CUDA
EXPORT char grid_func6b_cuda_operate_one(rgrid *dst, rgrid *src, REAL mass, REAL temp, REAL c4) {

  if(cuda_two_block_policy(src->value, src->grid_len, src->cufft_handle_r2c, src->id, 1, dst->value, dst->grid_len, dst->cufft_handle_r2c, dst->id, 0) < 0) return -1;
  grid_func6b_cuda_operate_oneW(cuda_find_block(dst->value), cuda_find_block(src->value), mass, temp, c4, dst->nx, dst->ny, dst->nz);

  return 0;
}
#endif

EXPORT void grid_func6b_operate_one(rgrid *dst, rgrid *src, REAL mass, REAL temp, REAL c4) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *svalue = src->value;
  REAL *dvalue = dst->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_func6b_cuda_operate_one(dst, src, mass, temp, c4)) return;
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,dvalue,svalue,mass,temp,c4) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      dvalue[ijnz + k] = grid_func6b(svalue[ijnz + k], mass, temp, c4);
  }
}
