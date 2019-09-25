 /*
 * Routines for complex grids involving differentiation.
 *
 * Nx is major index and Nz is minor index (varies most rapidly).
 *
 * For 2-D grids use: (1, NY, NZ)
 * For 1-D grids use: (1, 1, NZ)
 *
 */

#include "grid.h"
#include "private.h"

extern char grid_analyze_method;

#define FFT_BOUNDARY_TEST(X) (X == CGRID_FFT_EEE_BOUNDARY || X == CGRID_FFT_OEE_BOUNDARY || X == CGRID_FFT_EOE_BOUNDARY || X == CGRID_FFT_EEO_BOUNDARY || X == CGRID_FFT_OOE_BOUNDARY || X == CGRID_FFT_EOO_BOUNDARY || X == CGRID_FFT_OEO_BOUNDARY || X == CGRID_FFT_OOO_BOUNDARY)

#ifdef USE_CUDA
static char cgrid_bc_conv(cgrid *grid) {

  if(grid->value_outside == CGRID_DIRICHLET_BOUNDARY) return 0;
  else if(grid->value_outside == CGRID_NEUMANN_BOUNDARY) return 1;
  else if(grid->value_outside == CGRID_PERIODIC_BOUNDARY) return 2;
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
 * grid     = grid to be differentiated (cgrid *; input).
 * gradx    = differentiated grid x output (cgrid *; output).
 * grady    = differentiated grid y output (cgrid *; output).
 * gradz    = differentiated grid z output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_gradient(cgrid *grid, cgrid *gradx, cgrid *grady, cgrid *gradz) {

  if(grid_analyze_method) {
    cgrid_copy(gradx, grid);
    cgrid_fft(gradx);
    cgrid_copy(grady, gradx);
    cgrid_copy(gradz, gradx);
    cgrid_fft_gradient_x(gradx, gradx);
    cgrid_fft_gradient_y(grady, grady);
    cgrid_fft_gradient_z(gradz, gradz);
    cgrid_inverse_fft(gradx);
    cgrid_inverse_fft(grady);
    cgrid_inverse_fft(gradz);
  } else {
    cgrid_fd_gradient_x(grid, gradx);
    cgrid_fd_gradient_y(grid, grady);
    cgrid_fd_gradient_z(grid, gradz);
  }
}

/* 
 * Differentiate a grid with respect to x (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_gradient_x(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_x(gradient, gradient);
    cgrid_inverse_fft(gradient);
  } else cgrid_fd_gradient_x(grid, gradient);
}

/* 
 * Differentiate a grid with respect to y (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_gradient_y(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_y(gradient, gradient);
    cgrid_inverse_fft(gradient);
  } else cgrid_fd_gradient_y(grid, gradient);
}

/* 
 * Differentiate a grid with respect to z (central difference).
 * Uses grid_analyze_method to determine FFT vs. FD.
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_gradient_z(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_z(gradient, gradient);
    cgrid_inverse_fft(gradient);
  } else cgrid_fd_gradient_z(grid, gradient);
}


/*
 * Calculate laplacian of the grid.
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (cgrid *; input).
 * laplace = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_laplace(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace(laplace, laplace);
    cgrid_inverse_fft(laplace);
  } else cgrid_fd_laplace(grid, laplace);
}

/*
 * Calculate laplacian of the grid (X).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (cgrid *; input).
 * laplace = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_laplace_x(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_x(laplace, laplace);
    cgrid_inverse_fft(laplace);
  } else cgrid_fd_laplace_x(grid, laplace);
}

/*
 * Calculate laplacian of the grid (Y).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (cgrid *; input).
 * laplace = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_laplace_y(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_y(laplace, laplace);
    cgrid_inverse_fft(laplace);
  } else cgrid_fd_laplace_y(grid, laplace);
}

/*
 * Calculate laplacian of the grid (Z).
 * Use FD or FFT according to grid_analuze_method.
 *
 * grid    = source grid (cgrid *; input).
 * laplace = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_laplace_z(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_z(laplace, laplace);
    cgrid_inverse_fft(laplace);
  } else cgrid_fd_laplace_z(grid, laplace);
}

/* 
 * Differentiate a grid with respect to x (central difference).
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_fd_gradient_x(cgrid *grid, cgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny;
  REAL inv_delta = 1.0 / (2.0 * grid->step);
  REAL complex *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_gradient_x().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_gradient_x(grid, gradient, inv_delta, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (cgrid_value_at_index(grid, i+1, j, k) - cgrid_value_at_index(grid, i-1, j, k));
  }
}

/* 
 * Differentiate a grid with respect to y.
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_fd_gradient_y(cgrid *grid, cgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny;
  REAL inv_delta = 1.0 / (2.0 * grid->step);
  REAL complex *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_gradient_y().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_gradient_y(grid, gradient, inv_delta, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (cgrid_value_at_index(grid, i, j+1, k) - cgrid_value_at_index(grid, i, j-1, k));
  }
}

/* 
 * Differentiate a grid with respect to z.
 *
 * grid     = grid to be differentiated (cgrid *; input).
 * gradient = differentiated grid output (cgrid *; output).
 * 
 * No return value.
 *
 */

EXPORT void cgrid_fd_gradient_z(cgrid *grid, cgrid *gradient) {

  INT i, j, k, ij, ijnz, ny = grid->ny, nz = grid->nz, nxy = grid->nx * grid->ny;
  REAL inv_delta = 1.0 / (2.0 * grid->step);
  REAL complex *lvalue = gradient->value;
  
  if(grid == gradient) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_gradient_z().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_gradient_z(grid, gradient, inv_delta, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (cgrid_value_at_index(grid, i, j, k + 1) - cgrid_value_at_index(grid, i, j, k - 1));
  }
}

/*
 * Calculate laplacian of the grid.
 *
 * grid    = source grid (cgrid *; input).
 * laplace = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_laplace(cgrid *grid, cgrid *laplace) {

  INT i, j, k, ij, ijnz;
  INT ny = grid->ny, nz = grid->nz;
  INT nxy = grid->nx * grid->ny;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step);
  REAL complex *lvalue = laplace->value;
  
  if(grid == laplace) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_laplace().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_laplace(grid, laplace, inv_delta2, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-6.0 * cgrid_value_at_index(grid, i, j, k) + cgrid_value_at_index(grid, i, j, k + 1)
				       + cgrid_value_at_index(grid, i, j, k - 1) + cgrid_value_at_index(grid, i, j + 1, k) 
				       + cgrid_value_at_index(grid, i, j - 1, k) + cgrid_value_at_index(grid,i + 1,j,k) 
				       + cgrid_value_at_index(grid,i - 1, j, k));
  }
}

/*
 * Calculate laplacian of the grid (x component). This is the second derivative with respect to x.
 *
 * grid     = source grid (cgrid *; input).
 * laplacex = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_laplace_x(cgrid *grid, cgrid *laplacex) {

  INT i, j, k, ij, ijnz;
  INT ny = grid->ny, nz = grid->nz;
  INT nxy = grid->nx * grid->ny;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step);
  REAL complex *lvalue = laplacex->value;
  
  if(grid == laplacex) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_laplace_x().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_laplace_x(grid, laplacex, inv_delta2, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * cgrid_value_at_index(grid, i, j, k) + cgrid_value_at_index(grid, i + 1, j, k) 
                                        + cgrid_value_at_index(grid, i - 1, j, k));
  }
}

/*
 * Calculate laplacian of the grid (y component). This is the second derivative with respect to y.
 *
 * grid     = source grid (cgrid *; input).
 * laplacey = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_laplace_y(cgrid *grid, cgrid *laplacey) {

  INT i, j, k, ij, ijnz;
  INT ny = grid->ny, nz = grid->nz;
  INT nxy = grid->nx * grid->ny;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step);
  REAL complex *lvalue = laplacey->value;
  
  if(grid == laplacey) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_laplace_y().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_laplace_y(grid, laplacey, inv_delta2, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * cgrid_value_at_index(grid, i, j, k) + cgrid_value_at_index(grid, i, j + 1, k) 
                                        + cgrid_value_at_index(grid, i, j - 1, k));
  }
}

/*
 * Calculate vector laplacian of the grid (z component). This is the second derivative with respect to z.
 *
 * grid     = source grid (cgrid *; input).
 * laplacez = output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_laplace_z(cgrid *grid, cgrid *laplacez) {

  INT i, j, k, ij, ijnz;
  INT ny = grid->ny, nz = grid->nz;
  INT nxy = grid->nx * grid->ny;
  REAL inv_delta2 = 1.0 / (grid->step * grid->step);
  REAL complex *lvalue = laplacez->value;
  
  if(grid == laplacez) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_laplace_z().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_laplace_z(grid, laplacez, inv_delta2, cgrid_bc_conv(grid))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta2 * (-2.0 * cgrid_value_at_index(grid, i, j, k) + cgrid_value_at_index(grid, i, j, k + 1)
                                        + cgrid_value_at_index(grid, i, j, k - 1));
  }
}

/*
 * Differentiate grid in the Fourier space along x.
 *
 * grid       = grid to be differentiated (in Fourier space) (cgrid *; input).
 * gradient_x = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void cgrid_fft_gradient_x(cgrid *grid, cgrid *gradient_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy;
  REAL kx0 = grid->kx0;
  REAL kx, step, norm;
  REAL complex *gxvalue = gradient_x->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_gradient_x(grid, gradient_x)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  norm = grid->fft_norm;
  
  if (gradient_x != grid) cgrid_copy(gradient_x, grid);

  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gxvalue,kx0) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      ijnz = ij * nz;
      kx = M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
      for(k = 0; k < nz; k++)	  
	gxvalue[ijnz + k] *= (kx * norm) * I;
    }
  } else {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gxvalue,kx0) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (i <= nx / 2)
	kx = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
      else 
	kx = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step) - kx0;
      
      for(k = 0; k < nz; k++)	  
	gxvalue[ijnz + k] *= (kx * norm) * I;
    }
  }
}

/*
 * Differentiate grid in the Fourier space along y.
 *
 * grid       = grid to be differentiated (in Fourier space) (cgrid *; input).
 * gradient_y = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void cgrid_fft_gradient_y(cgrid *grid, cgrid *gradient_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL ky, step, norm;
  REAL ky0 = grid->ky0;
  REAL complex *gyvalue = gradient_y->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_gradient_y(grid, gradient_y)) return;
#endif
  /* f'(y) = iF[ i ky F[f(y)] ] */  
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  /* David: fft_norm */
  norm = grid->fft_norm;
  
  if (gradient_y != grid)
    cgrid_copy(gradient_y, grid);
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gyvalue,ky0) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      j = ij % ny;
      ijnz = ij * nz;
        ky = M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
      for(k = 0; k < nz; k++)	  
        gyvalue[ijnz + k] *= ky * norm * I;
    }
  } else {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gyvalue,ky0) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (j <= ny / 2)
        ky = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
      else 
        ky = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - ky0;
      
      for(k = 0; k < nz; k++)	  
        gyvalue[ijnz + k] *= ky * norm * I;
    }
  }
}

/*
 * Differentiate grid in the Fourier space along z.
 *
 * grid       = grid to be differentiated (in Fourier space) (cgrid *; input).
 * gradient_z = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input and output grids may be the same.
 *
 */

EXPORT void cgrid_fft_gradient_z(cgrid *grid, cgrid *gradient_z) {

  INT k, ij, ijnz, nx, ny, nz, nxy;
  REAL kz, lz, step, norm;
  REAL kz0 = grid->kz0;
  REAL complex *gzvalue = gradient_z->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_gradient_z(grid, gradient_z)) return;
#endif
  /* f'(z) = iF[ i kz F[f(z)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  /* David: fft_norm */
  norm = grid->fft_norm;
  
  if(gradient_z != grid) cgrid_copy(gradient_z, grid);

  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gzvalue,kz0) private(ij,ijnz,k,kz,lz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      ijnz = ij * nz;
      
      lz = ((REAL) nz) * step;
      for(k = 0; k < nz; k++) {
        kz = M_PI * ((REAL) k) / lz - kz0;
        gzvalue[ijnz + k] *= kz * norm * I;
      }
    }
  } else {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,gzvalue,kz0) private(ij,ijnz,k,kz,lz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      
      lz = ((REAL) nz) * step;
      for(k = 0; k < nz; k++) {
        if (k <= nz / 2)
          kz = 2.0 * M_PI * ((REAL) k) / lz - kz0;
        else 
          kz = 2.0 * M_PI * ((REAL) (k - nz)) / lz - kz0;
        
        gzvalue[ijnz + k] *= kz * norm * I;
      }
    }    
  }
}

/* 
 * Calculate laplacian of a grid (in Fourier space).
 *
 * grid    = grid to be differentiated (cgrid *; input).
 * laplace = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_laplace(cgrid *grid, cgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL kx, ky, kz, lx, ly, lz, step, norm;
  REAL complex *lvalue = laplace->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace(grid, laplace)) return;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  norm = grid->fft_norm;
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
    lx = M_PI / (((REAL) nx) * step);
    ly = M_PI / (((REAL) ny) * step);
    lz = M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      kx = ((REAL) i) * lx - kx0;
      ky = ((REAL) j) * ly - ky0;

      for(k = 0; k < nz; k++) {
        kz = ((REAL) k) * lz - kz0;
        lvalue[ijnz + k] *= (-kx * kx -ky * ky -kz * kz) * norm;
      }
    }    
  } else {
    lx = 2.0 * M_PI / ((REAL) nx) * step;
    ly = 2.0 * M_PI / ((REAL) ny) * step;
    lz = 2.0 * M_PI / ((REAL) nz) * step;
    nx2 = nx / 2;
    ny2 = ny / 2;
    nz2 = nz / 2;
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (i <= nx2)
        kx = ((REAL) i) * lx - kx0;
      else 
        kx = ((REAL) (i - nx)) * lx - kx0;
      
      if (j <= ny2)
        ky = ((REAL) j) * ly - ky0;
      else 
        ky = ((REAL) (j - ny)) * ly - ky0;
      
      for(k = 0; k < nz; k++) {
        if (k <= nz2)
          kz = ((REAL) k) * lz - kz0;
        else 
          kz = ((REAL) (k - nz)) * lz - kz0;
        
        lvalue[ijnz + k] *= -(kx * kx + ky * ky + kz * kz) * norm;
      }
    }
  }
}

/* 
 * Calculate second derivative of a grid with respect to x in Fourier space.
 *
 * grid    = grid to be differentiated (cgrid *; input).
 * laplace = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_laplace_x(cgrid *grid, cgrid *laplace)  {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx0 = grid->kx0;
  REAL kx, lx, step, norm;
  REAL complex *lvalue = laplace->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_x(grid, laplace)) return;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  norm = grid->fft_norm;
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
    lx = M_PI / (((REAL) nx) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      ijnz = ij * nz;

      kx = ((REAL) i) * lx - kx0;
      
      for(k = 0; k < nz; k++)
        lvalue[ijnz + k] *= -kx * kx * norm;
    }    
  } else {
    lx = 2.0 * M_PI / ((REAL) nx) * step;
    nx2 = nx / 2;
#pragma omp parallel for firstprivate(nx2,norm,nx,ny,nz,nxy,step,lvalue,kx0,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (i <= nx2)
        kx = ((REAL) i) * lx - kx0;
      else 
        kx = ((REAL) (i - nx)) * lx - kx0;
      
      for(k = 0; k < nz; k++)
        lvalue[ijnz + k] *= -kx * kx * norm;
    }
  }
}

/* 
 * Calculate second derivative of a grid with respect to y (in Fourier space).
 *
 * grid    = grid to be differentiated (cgrid *; input).
 * laplace = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_laplace_y(cgrid *grid, cgrid *laplace)  {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky0 = grid->ky0;
  REAL ky, ly, step, norm;
  REAL complex *lvalue = laplace->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_y(grid, laplace)) return;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  norm = grid->fft_norm;
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
    ly = M_PI / (((REAL) ny) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,ky0,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      j = ij % ny;
      ijnz = ij * nz;
      
      ky = ((REAL) j) * ly - ky0;

      for(k = 0; k < nz; k++)
        lvalue[ijnz + k] *= -ky * ky * norm;
    }    
  } else {
    ly = 2.0 * M_PI / ((REAL) ny) * step;
    ny2 = ny / 2;
#pragma omp parallel for firstprivate(ny2,norm,nx,ny,nz,nxy,step,lvalue,ky0,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      
      if (j <= ny2)
        ky = ((REAL) j) * ly - ky0;
      else 
        ky = ((REAL) (j - ny)) * ly - ky0;
      
      for(k = 0; k < nz; k++)
        lvalue[ijnz + k] *= -ky * ky * norm;
    }
  }
}

/* 
 * Calculate second derivative of a grid with respect to z (in Fourier space).
 *
 * grid    = grid to be differentiated (cgrid *; input).
 * laplace = output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_laplace_z(cgrid *grid, cgrid *laplace)  {

  INT k, ij, ijnz, nx, ny, nz, nxy, nz2;
  REAL kz0 = grid->kz0;
  REAL kz, lz, step, norm;
  REAL complex *lvalue = laplace->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_z(grid, laplace)) return;
#endif

  /* f''(x) = iF[ -k^2 F[f(x)] ] */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  norm = grid->fft_norm;
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
    lz = M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kz0,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      ijnz = ij * nz;
      for(k = 0; k < nz; k++) {
        kz = ((REAL) k) * lz - kz0;
        lvalue[ijnz + k] *= -kz * kz * norm;
      }
    }    
  } else {
    lz = 2.0 * M_PI / ((REAL) nz) * step;
    nz2 = nz / 2;
#pragma omp parallel for firstprivate(nz2,norm,nx,ny,nz,nxy,step,lvalue,kz0,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      for(k = 0; k < nz; k++) {
        if (k <= nz2)
          kz = ((REAL) k) * lz - kz0;
        else 
          kz = ((REAL) (k - nz)) * lz - kz0;
        
        lvalue[ijnz + k] *= -kz * kz * norm;
      }
    }
  }
}


/*
 * Calculate expectation value of laplace operator in the Fourier space (int grid^* grid'').
 *
 * grid    = source grid for the operation (in Fourier space) (cgrid *; input).
 * laplace = laplacian of the grid (input) (cgrid *; output).
 *
 * Returns the expectation value (REAL).
 *
 */

EXPORT REAL cgrid_fft_laplace_expectation_value(cgrid *grid, cgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step, norm, sum = 0.0, ssum;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL complex *lvalue = laplace->value;
  REAL aux;

  if(grid != laplace) cgrid_copy(laplace, grid);

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_expectation_value(laplace, &sum)) return sum;
#endif

  /* int f*(x) f''(x) dx */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  /* int (delta FFT[f(x)] )^2 dk => delta^2 / N delta */

  norm = grid->fft_norm;
  if(nx != 1) norm *= step;
  if(ny != 1) norm *= step;
  if(nz != 1) norm *= step;
  
  if(FFT_BOUNDARY_TEST(grid->value_outside)) {
    lx = M_PI / (((REAL) nx) * step);
    ly = M_PI / (((REAL) ny) * step);
    lz = M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz,ssum,aux) reduction(+:sum) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      kx = ((REAL) i) * lx - kx0;
      ky = ((REAL) j) * ly - ky0;
      
      ssum = 0.0;
      
      for(k = 0; k < nz; k++) {
        kz = ((REAL) k) * lz - kz0;
        /* Manual fixing of boundaries: the symmetry points (i=0 or i=nx-1 etc) have 1/2 the weigth in the integral */
        aux = -(kx * kx + ky * ky + kz * kz) * sqnorm(lvalue[ijnz + k]);
        if(i==0 || i==nx-1) aux *= 0.5;
        if(j==0 || j==ny-1) aux *= 0.5;
        if(k==0 || k==nz-1) aux *= 0.5;
        ssum += aux;
      }
      sum += ssum;
    }
  } else {
    lx = 2.0 * M_PI / ((REAL) nx) * step;
    ly = 2.0 * M_PI / ((REAL) ny) * step;
    lz = 2.0 * M_PI / ((REAL) nz) * step;
    nx2 = nx / 2;
    ny2 = ny / 2;
    nz2 = nz / 2;
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz,ssum) reduction(+:sum) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (i <= nx2)
	kx = ((REAL) i) * lx - kx0;
      else 
	kx = ((REAL) (i - nx)) * lx - kx0;
      
      if (j <= ny2)
	ky = ((REAL) j) * ly - ky0;
      else 
	ky = ((REAL) (j - ny)) * ly - ky0;
      
      ssum = 0.0;
      
      for(k = 0; k < nz; k++) {
	if (k <= nz2)
	  kz = ((REAL) k) * lz - kz0;
	else 
	  kz = ((REAL) (k - nz)) * lz - kz0;
	
	ssum -= (kx * kx + ky * ky + kz * kz) * sqnorm(lvalue[ijnz + k]);
      }
      
      sum += ssum;
    }
  }
  
  return sum * norm;
}

/*
 * Solve Poisson equation: Laplace f = u subject to periodic boundary condition. Grid in Fourier space.
 * Uses finite difference for Laplacian (7 point) and FFT. Num. Recip. Sect. 19.4.
 *
 * grid = On entry function u specified over grid (input) 
 *        and function f (output) on exit (cgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_poisson(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, idx; // could move some of the computations below past cuda call
  REAL step = grid->step, step2 = step * step, ilx = 2.0 * M_PI / ((REAL) nx), ily = 2.0 * M_PI / ((REAL) ny), ilz = 2.0 * M_PI / ((REAL) nz), kx, ky, kz;
  REAL norm = grid->fft_norm;
  REAL complex *value = grid->value;

  if(grid->value_outside != CGRID_PERIODIC_BOUNDARY) {
    fprintf(stderr, "libgrid: Only periodic boundary Poisson solver implemented.\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_poisson(grid)) return;  
#endif
#pragma omp parallel for firstprivate(nx, ny, nz, value, ilx, ily, ilz, step2, norm) private(i, j, k, kx, ky, kz, idx) default(none) schedule(runtime)
  for(i = 0; i < nx; i++) {
    kx = COS(ilx * ((REAL) i));
    for(j = 0; j < ny; j++) {
      ky = COS(ily * ((REAL) j));
      for(k = 0; k < nz; k++) {
	kz = COS(ilz * ((REAL) k));
	idx = (i * ny + j) * nz + k;
	if(i || j || k)
	  value[idx] *= norm * step2 / (2.0 * (kx + ky + kz - 3.0));
	else
	  value[idx] = 0.0;
      }
    }
  }
}

/*
 * Calculate divergence of a vector field.
 *
 * div     = result (cgrid *; output).
 * fx      = x component of the field (cgrid *; input).
 * fy      = y component of the field (cgrid *; input).
 * fz      = z component of the field (cgrid *; input).
 *
 */

EXPORT void cgrid_fd_div(cgrid *div, cgrid *fx, cgrid *fy, cgrid *fz) {

  INT i, j, k, ij, ijnz, ny = div->ny, nz = div->nz, nxy = div->nx * div->ny;
  REAL inv_delta = 1.0 / (2.0 * div->step);
  REAL complex *lvalue = div->value;
  
  if(div == fx || div == fy || div == fz) {
    fprintf(stderr, "libgrid: Destination grid must be different from input grids in cgrid_div().\n");
    abort();
  }

#ifdef USE_CUDA
  cuda_remove_block(lvalue, 0);
  cuda_remove_block(fx->value, 1);
  cuda_remove_block(fy->value, 1);
  cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (cgrid_value_at_index(fx, i + 1, j, k) - cgrid_value_at_index(fx, i - 1, j, k)
	+ cgrid_value_at_index(fy, i, j + 1, k) - cgrid_value_at_index(fy, i, j - 1, k)
        + cgrid_value_at_index(fz, i, j, k + 1) - cgrid_value_at_index(fz, i, j, k - 1));
  }
}

/*
 * Calculate rot of a vector field.
 *
 * rotx = x component of rot (cgrid *; output). If NULL, not computed (fx not accessed and may also be NULL).
 * roty = y component of rot (cgrid *; output). If NULL, not computed (fy not accessed and may also be NULL).
 * rotz = z component of rot (cgrid *; output). If NULL, not computed (fz not accessed and may also be NULL).
 * fx   = x component of the field (cgrid *; input).
 * fy   = y component of the field (cgrid *; input).
 * fz   = z component of the field (cgrid *; input).
 *
 * TODO: CUDA implementation missing.
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_rot(cgrid *rotx, cgrid *roty, cgrid *rotz, cgrid *fx, cgrid *fy, cgrid *fz) {

  INT i, j, k, ij, ijnz, ny = rotx->ny, nz = rotx->nz, nxy = rotx->nx * rotx->ny;
  REAL inv_delta = 1.0 / (2.0 * rotx->step);
  REAL complex *lvaluex = rotx->value, *lvaluey = roty->value, *lvaluez = rotz->value;

  if(rotx == NULL && roty == NULL && rotz == NULL) return; /* Nothing to do */

  if(rotx || rotz) {
    ny = fy->ny;
    nz = fy->nz;
    nxy = fy->nx * fy->ny;
    inv_delta = 1.0 / (2.0 * fy->step);
  } else {
    ny = fx->ny;
    nz = fx->nz;
    nxy = fx->nx * fx->ny;
    inv_delta = 1.0 / (2.0 * fx->step);
  }

  if(rotx) lvaluex = rotx->value;
  else lvaluex = NULL;
  if(roty) lvaluey = roty->value;
  else lvaluey = NULL;
  if(rotz) lvaluez = rotz->value;
  else lvaluez = NULL;
  
#ifdef USE_CUDA
  if(lvaluex) cuda_remove_block(lvaluex, 0);
  if(lvaluey) cuda_remove_block(lvaluey, 0);
  if(lvaluez) cuda_remove_block(lvaluez, 0);
  if(roty || rotz) cuda_remove_block(fx->value, 1);
  if(rotx || rotz) cuda_remove_block(fy->value, 1);
  if(rotx || roty) cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nxy,lvaluex,lvaluey,lvaluez,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* x: (d/dy) fz - (d/dz) fy (no access to fx) */
      if(lvaluex)
        lvaluex[ijnz + k] = inv_delta * ((cgrid_value_at_index(fz, i, j + 1, k) - cgrid_value_at_index(fz, i, j - 1, k))
				      - (cgrid_value_at_index(fy, i, j, k + 1) - cgrid_value_at_index(fy, i, j, k - 1)));
      /* y: (d/dz) fx - (d/dx) fz (no access to fy) */
      if(lvaluey)
        lvaluey[ijnz + k] = inv_delta * ((cgrid_value_at_index(fx, i, j, k + 1) - cgrid_value_at_index(fx, i, j, k - 1))
				      - (cgrid_value_at_index(fz, i + 1, j, k) - cgrid_value_at_index(fz, i - 1, j, k)));
      /* z: (d/dx) fy - (d/dy) fx (no access to fz) */
      if(lvaluez)
        lvaluez[ijnz + k] = inv_delta * ((cgrid_value_at_index(fy, i + 1, j, k) - cgrid_value_at_index(fy, i - 1, j, k))
				      - (cgrid_value_at_index(fx, i, j + 1, k) - cgrid_value_at_index(fx, i, j - 1, k)));
    }
  }
}

/*
 * Calculate |rot| (|curl|; |\Nablda\Times|) of a vector field (i.e., magnitude).
 *
 * rot  = magnitudet of rot (rgrid *; output).
 * fx   = x component of the field (cgrid *; input).
 * fy   = y component of the field (cgrid *; input).
 * fz   = z component of the field (cgrid *; input).
 *
 * TODO: CUDA implementation missing.
 *
 */

EXPORT void cgrid_fd_abs_rot(rgrid *rot, cgrid *fx, cgrid *fy, cgrid *fz) {

  INT i, j, k, ij, ijnz, ny = rot->ny, nz = rot->nz, nxy = rot->nx * rot->ny;
  REAL inv_delta = 1.0 / (2.0 * rot->step);
  REAL complex tmp;
  REAL *lvalue = rot->value;
  
#ifdef USE_CUDA
  cuda_remove_block(lvalue, 0);
  cuda_remove_block(fx->value, 1);
  cuda_remove_block(fy->value, 1);
  cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nxy,lvalue,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* x: (d/dy) fz - (d/dz) fy */
      tmp = inv_delta * ((cgrid_value_at_index(fz, i, j + 1, k) - cgrid_value_at_index(fz, i, j - 1, k))
				      - (cgrid_value_at_index(fy, i, j, k + 1) - cgrid_value_at_index(fy, i, j, k - 1)));
      lvalue[ijnz + k] = (REAL) (CONJ(tmp) * tmp);
      /* y: (d/dz) fx - (d/dx) fz */
      tmp = inv_delta * ((cgrid_value_at_index(fx, i, j, k + 1) - cgrid_value_at_index(fx, i, j, k - 1))
				      - (cgrid_value_at_index(fz, i + 1, j, k) - cgrid_value_at_index(fz, i - 1, j, k)));
      lvalue[ijnz + k] += (REAL) (CONJ(tmp) * tmp);
      /* z: (d/dx) fy - (d/dy) fx */
      tmp = inv_delta * ((cgrid_value_at_index(fy, i+1, j, k) - cgrid_value_at_index(fy, i-1, j, k))
				      - (cgrid_value_at_index(fx, i, j + 1, k) - cgrid_value_at_index(fx, i, j - 1, k)));
      lvalue[ijnz + k] += (REAL) (CONJ(tmp) * tmp);
      lvalue[ijnz + k] = SQRT(lvalue[ijnz + k]);
    }
  }
}
