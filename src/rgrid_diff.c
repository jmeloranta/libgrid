/*
 * Routines for real grids involving differentiation.
 *
 * Nx is major index and Nz is minor index (varies most rapidly).
 *
 * For 2-D grids use: (1, NY, NZ)
 * For 1-D grids use: (1, 1, NZ)
 *
 * Note that due to FFT, the last index dimension is 2 * (nz / 2 + 1) rather than just nz.
 *
 */

#include "grid.h"
#include "private.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

extern char grid_analyze_method;

#ifdef USE_CUDA
static char rgrid_bc_conv(rgrid *grid) {

  if(grid->value_outside == RGRID_DIRICHLET_BOUNDARY) return 0;
  else if(grid->value_outside == RGRID_NEUMANN_BOUNDARY) return 1;
  else if(grid->value_outside == RGRID_PERIODIC_BOUNDARY) return 2;
  else {
    fprintf(stderr, "libgrid(cuda): Incompatible boundary condition.\n");
    abort();
  }
}
#endif

/* 
 * Differentiate a grid with respect to x, y, z (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradx    = differentiated grid x output (rgrid *; output).
 * grady    = differentiated grid y output (rgrid *; output).
 * gradz    = differentiated grid z output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_gradient(rgrid *grid, rgrid *gradx, rgrid *grady, rgrid *gradz) {

  if(grid_analyze_method) {
    rgrid_copy(gradx, grid);
    rgrid_fft(gradx);
    rgrid_copy(grady, gradx);
    rgrid_copy(gradz, gradx);
    rgrid_fft_gradient_x(gradx, gradx);
    rgrid_fft_gradient_y(grady, grady);
    rgrid_fft_gradient_z(gradz, gradz);
    rgrid_inverse_fft(gradx);
    rgrid_inverse_fft(grady);
    rgrid_inverse_fft(gradz);
  } else {
    rgrid_fd_gradient_x(grid, gradx);
    rgrid_fd_gradient_y(grid, grady);
    rgrid_fd_gradient_z(grid, gradz);
  }
}

/* 
 * Differentiate a grid with respect to x (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_gradient_x(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_x(gradient, gradient);
    rgrid_inverse_fft(gradient);
  } else rgrid_fd_gradient_x(grid, gradient);
}

/* 
 * Differentiate a grid with respect to y (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_gradient_y(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_y(gradient, gradient);
    rgrid_inverse_fft(gradient);
  } else rgrid_fd_gradient_y(grid, gradient);
}

/* 
 * Differentiate a grid with respect to z (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_gradient_z(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_z(gradient, gradient);
    rgrid_inverse_fft(gradient);
  } else rgrid_fd_gradient_z(grid, gradient);
}


/*
 * Calculate laplacian of the grid.
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (rgrid *; input).
 * laplace = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_laplace(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace(laplace, laplace);
    rgrid_inverse_fft(laplace);
  } else rgrid_fd_laplace(grid, laplace);
}

/*
 * Calculate laplacian of the grid (X).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (rgrid *; input).
 * laplace = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_laplace_x(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_x(laplace, laplace);
    rgrid_inverse_fft(laplace);
  } else rgrid_fd_laplace_x(grid, laplace);
}

/*
 * Calculate laplacian of the grid (Y).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (rgrid *; input).
 * laplace = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_laplace_y(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_y(laplace, laplace);
    rgrid_inverse_fft(laplace);
  } else rgrid_fd_laplace_y(grid, laplace);
}

/*
 * Calculate laplacian of the grid (Z).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (rgrid *; input).
 * laplace = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_laplace_z(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_z(laplace, laplace);
    rgrid_inverse_fft(laplace);
  } else rgrid_fd_laplace_z(grid, laplace);
}

/* 
 * Differentiate a grid with respect to x (central difference).
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_fd_gradient_x(rgrid *grid, rgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta = 1.0 / (2.0 * grid->step), *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_gradient_x().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_gradient_x(grid, gradient, inv_delta, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (rgrid_value_at_index(grid, i + 1, j, k) - rgrid_value_at_index(grid, i - 1, j, k));
  }
}

/* 
 * Differentiate a grid with respect to y.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_fd_gradient_y(rgrid *grid, rgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta = 1.0 / (2.0 * grid->step), *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_gradient_y().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_gradient_y(grid, gradient, inv_delta, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (rgrid_value_at_index(grid, i, j + 1, k) - rgrid_value_at_index(grid, i, j - 1, k));
  }
}

/* 
 * Differentiate a grid with respect to z.
 *
 * grid     = grid to be differentiated (rgrid *; input).
 * gradient = differentiated grid output (rgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_fd_gradient_z(rgrid *grid, rgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta = 1.0 / (2.0 * grid->step), *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_gradient_z().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_gradient_z(grid, gradient, inv_delta, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (rgrid_value_at_index(grid, i, j, k + 1) - rgrid_value_at_index(grid, i, j, k - 1));
  }
}

/*
 * Calculate laplacian of the grid.
 *
 * grid    = source grid (rgrid *; input).
 * laplace = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fd_laplace(rgrid *grid, rgrid *laplace) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step), *lvalue = laplace->value;
  
  if(grid == laplace) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_laplace().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_laplace(grid, laplace, inv_delta2, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-6.0 * rgrid_value_at_index(grid, i, j, k) + rgrid_value_at_index(grid, i, j, k + 1)
				       + rgrid_value_at_index(grid, i, j, k - 1) + rgrid_value_at_index(grid, i, j + 1, k) 
				       + rgrid_value_at_index(grid, i, j - 1, k) + rgrid_value_at_index(grid, i + 1, j, k)
				       + rgrid_value_at_index(grid, i - 1, j, k));
  }
}

/*
 * Calculate laplacian of the grid (x component). This is the second derivative with respect to x.
 *
 * grid     = source grid (rgrid *; input).
 * laplacex = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fd_laplace_x(rgrid *grid, rgrid *laplacex) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step), *lvalue = laplacex->value;
  
  if(grid == laplacex) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_laplace_x().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_laplace_x(grid, laplacex, inv_delta2, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * rgrid_value_at_index(grid, i, j, k) + rgrid_value_at_index(grid, i + 1, j, k) 
                                        + rgrid_value_at_index(grid, i - 1, j, k));
  }
}

/*
 * Calculate laplacian of the grid (y component). This is the second derivative with respect to y.
 *
 * grid     = source grid (rgrid *; input).
 * laplacey = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fd_laplace_y(rgrid *grid, rgrid *laplacey) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step), *lvalue = laplacey->value;
  
  if(grid == laplacey) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_laplace_y().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_laplace_y(grid, laplacey, inv_delta2, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * rgrid_value_at_index(grid, i, j, k) + rgrid_value_at_index(grid, i, j + 1, k) 
                                        + rgrid_value_at_index(grid, i, j - 1, k));
  }
}

/*
 * Calculate laplacian of the grid (z component). This is the second derivative with respect to z.
 *
 * grid     = source grid (rgrid *; input).
 * laplacez = output grid for the operation (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fd_laplace_z(rgrid *grid, rgrid *laplacez) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny, nzz = grid->nz2;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step), *lvalue = laplacez->value;
  
  if(grid == laplacez) {
    fprintf(stderr, "libgrid: source and destination must be different in rgrid_fd_laplace_z().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fd_laplace_z(grid, laplacez, inv_delta2, rgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * rgrid_value_at_index(grid, i, j, k) + rgrid_value_at_index(grid, i, j, k + 1) 
                                       + rgrid_value_at_index(grid, i, j, k - 1));
  }
}

/*
 * Differentiate real grid in the Fourier space along x.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * gradient_x = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_gradient_x(rgrid *grid, rgrid *gradient_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx0 = grid->kx0;
  REAL kx, step, norm, lx;
  REAL complex *gxvalue = (REAL complex *) gradient_x->value;
  
  if (gradient_x != grid) rgrid_copy(gradient_x, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_gradient_x(gradient_x)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  lx = 2.0 * M_PI / (((REAL) nx) * step);
  nx2 = nx / 2;

#pragma omp parallel for firstprivate(nx2,norm,nx,ny,nz,nxy,step,gxvalue,kx0,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
    if(i < nx2) 
      kx = ((REAL) i) * lx - kx0;
    else
      kx = -((REAL) (nx - i)) * lx - kx0;
    for(k = 0; k < nz; k++)	  
      gxvalue[ijnz + k] *= (kx * norm) * I;
  } 
}

/*
 * Differentiate real grid in the Fourier space along y.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * gradient_y = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_gradient_y(rgrid *grid, rgrid *gradient_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky0 = grid->ky0;
  REAL ky, step, norm, ly;
  REAL complex *gyvalue = (REAL complex *) gradient_y->value;
  
  if (gradient_y != grid) rgrid_copy(gradient_y, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_gradient_y(gradient_y)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  ny2 = ny / 2;

#pragma omp parallel for firstprivate(ny2,norm,nx,ny,nz,nxy,step,gyvalue,ky0,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
    if(j < ny2) 
      ky = ((REAL) j) * ly - ky0;
    else
      ky = -((REAL) (ny - j)) * ly - ky0;
    for(k = 0; k < nz; k++)	  
      gyvalue[ijnz + k] *= (ky * norm) * I;
  } 
}

/*
 * Differentiate real grid in the Fourier space along z.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * gradient_z = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_gradient_z(rgrid *grid, rgrid *gradient_z) {

  INT k, ij, ijnz, nx, ny, nxy, nz, nz2;
  REAL kz0 = grid->kz0;
  REAL kz, step, norm, lz;
  REAL complex *gzvalue = (REAL complex *) gradient_z->value;
  
  if (gradient_z != grid) rgrid_copy(gradient_z, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_gradient_z(gradient_z)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  lz = M_PI / (((REAL) nz - 1) * step);
  nz2 = nz / 2;

#pragma omp parallel for firstprivate(nz2,norm,nx,ny,nz,nxy,step,gzvalue,kz0,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz - kz0;
      else
        kz = -((REAL) (nz - k)) * lz - kz0;
      gzvalue[ijnz + k] *= (kz * norm) * I;
    }
  } 
}

/* 
 * Calculate second derivative of a grid (in Fourier space).
 *
 * grid    = grid to be differentiated (rgrid *; input).
 * laplace = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void rgrid_fft_laplace(rgrid *grid, rgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nxy, nz;
  INT nx2, ny2, nz2;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL kx, ky, kz, lx, ly, lz, step, norm;
  REAL complex *lvalue = (REAL complex *) laplace->value;
  
  if (grid != laplace) rgrid_copy(laplace, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace(laplace)) return;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;  
  lx = 2.0 * M_PI / (((REAL) nx) * step);
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  lz = M_PI / (((REAL) nz - 1) * step);
  
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    ijnz = ij * nz;
    
    if(i < nx2) 
      kx = ((REAL) i) * lx - kx0;
    else
      kx = -((REAL) (nx - i)) * lx - kx0;
    if(j < ny2) 
      ky = ((REAL) j) * ly - ky0;
    else
      ky = -((REAL) (ny - j)) * ly - ky0;
      
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz - kz0;
      else
        kz = -((REAL) (nz - k)) * lz - kz0;        
      lvalue[ijnz + k] *= -(kx * kx + ky * ky + kz * kz) * norm;
    }
  }
}

/*
 * Differentiate real grid in the Fourier space along x twice.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * laplace_x = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_laplace_x(rgrid *grid, rgrid *laplace_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx0 = grid->kx0;
  REAL kx, step, norm, lx;
  REAL complex *gxvalue = (REAL complex *) laplace_x->value;
  
  if (laplace_x != grid) rgrid_copy(laplace_x, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace_x(laplace_x)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  lx = 2.0 * M_PI / (((REAL) nx) * step);
  nx2 = nx / 2;

#pragma omp parallel for firstprivate(nx2,norm,nx,ny,nz,nxy,step,gxvalue,kx0,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
    if(i < nx2) 
      kx = ((REAL) i) * lx - kx0;
    else
      kx = -((REAL) (nx - i)) * lx - kx0;
    for(k = 0; k < nz; k++)	  
      gxvalue[ijnz + k] *= -kx * kx * norm;
  } 
}

/*
 * Differentiate real grid in the Fourier space along y twice.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * laplace_y  = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_laplace_y(rgrid *grid, rgrid *laplace_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky0 = grid->ky0;
  REAL ky, step, norm, ly;
  REAL complex *gyvalue = (REAL complex *) laplace_y->value;
  
  if (laplace_y != grid) rgrid_copy(laplace_y, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace_y(laplace_y)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  ny2 = ny / 2;

#pragma omp parallel for firstprivate(ny2,norm,nx,ny,nz,nxy,step,gyvalue,ky0,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
    if(j < ny2) 
      ky = ((REAL) j) * ly - ky0;
    else
      ky = -((REAL) (ny - j)) * ly - ky0;
    for(k = 0; k < nz; k++)	  
      gyvalue[ijnz + k] *= -ky * ky * norm;
  } 
}

/*
 * Differentiate real grid in the Fourier space along z twice.
 *
 * grid       = grid to be differentiated (in Fourier space) (rgrid *; input).
 * laplace_z  = output grid (rgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void rgrid_fft_laplace_z(rgrid *grid, rgrid *laplace_z) {

  INT k, ij, ijnz, nx, ny, nxy, nz, nz2;
  REAL kz0 = grid->kz0;
  REAL kz, step, norm, lz;
  REAL complex *gzvalue = (REAL complex *) laplace_z->value;
  
  if (laplace_z != grid) rgrid_copy(laplace_z, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace_z(laplace_z)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;
  lz = M_PI / (((REAL) nz - 1) * step);
  nz2 = nz / 2;

#pragma omp parallel for firstprivate(nz2,norm,nx,ny,nz,nxy,step,gzvalue,kz0,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz - kz0;
      else
        kz = -((REAL) (nz - k)) * lz - kz0;
      gzvalue[ijnz + k] *= -kz * kz * norm;
    }
  } 
}

/*
 * Calculate expectation value of laplace operator in the Fourier space (int grid^* grid'').
 *
 * grid    = source grid for the operation (in Fourier space) (rgrid *; input).
 * laplace = laplacian of the grid (input) (rgrid *; output).
 *
 * Returns the expectation value (REAL).
 *
 */

EXPORT REAL rgrid_fft_laplace_expectation_value(rgrid *grid, rgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nxy, nz;
  INT nx2, ny2, nz2;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL kx, ky, kz, lx, ly, lz, step, norm, sum = 0.0, ssum;
  REAL complex *lvalue = (REAL complex *) laplace->value;
  
  if (grid != laplace) rgrid_copy(laplace, grid);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace_expectation_value(laplace, &sum)) return sum;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
  nxy = nx * ny;
  step = grid->step;
  norm = grid->fft_norm;  
  if(nx != 1) norm *= step;
  if(ny != 1) norm *= step;
  if(nz != 1) norm *= step;

  lx = 2.0 * M_PI / (((REAL) nx) * step);
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  lz = M_PI / (((REAL) nz - 1) * step);
  
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(ssum,i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime) reduction(+:sum)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    ijnz = ij * nz;
    
    if(i < nx2) 
      kx = ((REAL) i) * lx - kx0;
    else
      kx = -((REAL) (nx - i)) * lx - kx0;
    if(j < ny2) 
      ky = ((REAL) j) * ly - ky0;
    else
      ky = -((REAL) (ny - j)) * ly - ky0;
      
    ssum = 0.0;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz - kz0;
      else
        kz = -((REAL) (nz - k)) * lz - kz0;        
      ssum -= (kx * kx + ky * ky + kz * kz) * sqnorm(lvalue[ijnz + k]);
    }
    sum += ssum;
  }
  return sum * norm;
}
