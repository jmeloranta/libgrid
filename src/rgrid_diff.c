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
#include "cprivate.h"

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
 * @FUNC{rgrid_gradient, "Gradient of real grid"}
 * @DESC{"Differentiate a grid with respect to x, y, z. Uses grid_analyze_method to determine FFT vs. FD"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradx, "Differentiated grid x output"}
 * @ARG3{rgrid *grady, "Differentiated grid y output"}
 * @ARG4{rgrid *gradz, "Differentiated grid z output"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_gradient(rgrid *grid, rgrid *gradx, rgrid *grady, rgrid *gradz) {

  if(grid_analyze_method) {
    rgrid_copy(gradz, grid);
    rgrid_fft(gradz);
    rgrid_fft_gradient_x(gradz, gradx);
    rgrid_fft_gradient_y(gradz, grady);
    rgrid_fft_gradient_z(gradz, gradz);
    rgrid_inverse_fft_norm(gradx);
    rgrid_inverse_fft_norm(grady);
    rgrid_inverse_fft_norm(gradz);
  } else {
    rgrid_fd_gradient_x(grid, gradx);
    rgrid_fd_gradient_y(grid, grady);
    rgrid_fd_gradient_z(grid, gradz);
  }
}

/* 
 * @FUNC{rgrid_gradient_x, "Differentiate real grid with respect to x"}
 * @DESC{"Differentiate a grid with respect to x.
          Uses grid_analyze_method to determine FFT vs. FD"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Differentiated grid output"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_gradient_x(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_x(gradient, gradient);
    rgrid_inverse_fft_norm(gradient);
  } else rgrid_fd_gradient_x(grid, gradient);
}

/* 
 * @FUNC{rgrid_gradient_y, "Differentiate real grid with respect to y"}
 * @DESC{"Differentiate a grid with respect to y.
          Uses grid_analyze_method to determine FFT vs. FD"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Differentiated grid output"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_gradient_y(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_y(gradient, gradient);
    rgrid_inverse_fft_norm(gradient);
  } else rgrid_fd_gradient_y(grid, gradient);
}

/* 
 * @FUNC{rgrid_gradient_z, "Differentiate real grid with respect to z"}
 * @DESC{"Differentiate a grid with respect to z.
          Uses grid_analyze_method to determine FFT vs. FD"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Differentiated grid output"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_gradient_z(rgrid *grid, rgrid *gradient) {

  if(grid_analyze_method) {
    rgrid_copy(gradient, grid);
    rgrid_fft(gradient);
    rgrid_fft_gradient_z(gradient, gradient);
    rgrid_inverse_fft_norm(gradient);
  } else rgrid_fd_gradient_z(grid, gradient);
}

/*
 * @FUNC{rgrid_laplace, "Laplacian of real grid"}
 * @DESC{"Calculate laplacian of the grid. Use FD or FFT according to grid_analuze_method"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_laplace(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace(laplace, laplace);
    rgrid_inverse_fft_norm(laplace);
  } else rgrid_fd_laplace(grid, laplace);
}

/*
 * @FUNC{rgrid_laplace_x, "Second derivative of real grid with respect to x"}
 * @DESC{"Calculate laplacian of the grid (X). Use FD or FFT according to grid_analuze_method"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_laplace_x(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_x(laplace, laplace);
    rgrid_inverse_fft_norm(laplace);
  } else rgrid_fd_laplace_x(grid, laplace);
}

/*
 * @FUNC{rgrid_laplace_y, "Second derivative of real grid with respect to y"}
 * @DESC{"Calculate laplacian of the grid (Y). Use FD or FFT according to grid_analuze_method"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_laplace_y(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_y(laplace, laplace);
    rgrid_inverse_fft_norm(laplace);
  } else rgrid_fd_laplace_y(grid, laplace);
}

/*
 * @FUNC{rgrid_laplace_z, "Second derivative of real grid with respect to z"}
 * @DESC{"Calculate laplacian of the grid (Z). Use FD or FFT according to grid_analuze_method"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_laplace_z(rgrid *grid, rgrid *laplace) {

  if(grid_analyze_method) {
    rgrid_copy(laplace, grid);
    rgrid_fft(laplace);
    rgrid_fft_laplace_z(laplace, laplace);
    rgrid_inverse_fft_norm(laplace);
  } else rgrid_fd_laplace_z(grid, laplace);
}

/* 
 * @FUNC{rgrid_fd_gradient_x, "Differentiate grid with respect to x (finite difference)"}
 * @DESC{"Differentiate a grid with respect to x (central difference)"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Output grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_gradient_y, "Differentiate grid with respect to y (finite difference)"}
 * @DESC{"Differentiate a grid with respect to y (central difference)"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Output grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_gradient_z, "Differentiate grid with respect to z (finite difference)"}
 * @DESC{"Differentiate a grid with respect to z (central difference)"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *gradient, "Output grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_laplace, "Laplacian of real grid (finite difference)"}
 * @DESC{"Calculate laplacian of the grid (central difference)"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_laplace_x, "Second derivative of real grid with respect to x"}
 * @DESC{"Calculate second derivative of the grid with respect to x (finite difference)"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplacex, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_laplace_y, "Second derivative of real grid with respect to y (finite difference)"}
 * @DESC{"Calculate second derivative of the grid with respect to y"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplacey, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fd_laplace_z, "Second derivative of real grid with respect to z (finite difference)"}
 * @DESC{"Calculate second derivative of the grid with respect to z"}
 * @ARG1{rgrid *grid, "Source grid"}
 * @ARG2{rgrid *laplacez, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fft_gradient_x, "Differentiate real grid with respect to x (FFT)"}
 * @DESC{"Differentiate real grid in the reciprocal space along x.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *gradient_x, "Output grid (in reciprocal space)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_gradient_x(rgrid *grid, rgrid *gradient_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx, step, lx;
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
  lx = 2.0 * M_PI / (step * (REAL) nx);
  nx2 = nx / 2;

#pragma omp parallel for firstprivate(nx2,nx,ny,nz,nxy,step,gxvalue,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
    if(i < nx2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;
    for(k = 0; k < nz; k++)	  
      gxvalue[ijnz + k] *= kx * I;
  } 
}

/*
 * @FUNC{rgrid_fft_gradient_y, "Differentiate real grid with respect to y (FFT)"}
 * @DESC{"Differentiate real grid in the reciprocal space along y.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *gradient_y, "Output grid (in reciprocal space)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_gradient_y(rgrid *grid, rgrid *gradient_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky, step, ly;
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
  ly = 2.0 * M_PI / (step * (REAL) ny);
  ny2 = ny / 2;

#pragma omp parallel for firstprivate(ny2,nx,ny,nz,nxy,step,gyvalue,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
    if(j < ny2) 
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;
    for(k = 0; k < nz; k++)	  
      gyvalue[ijnz + k] *= ky * I;
  } 
}

/*
 * @FUNC{rgrid_fft_gradient_z, "Differentiate real grid with respect to z (FFT)"}
 * @DESC{"Differentiate real grid in the reciprocal space along z.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *gradient_z, "Output grid (in reciprocal space)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_gradient_z(rgrid *grid, rgrid *gradient_z) {

  INT k, ij, ijnz, nx, ny, nxy, nz, nz2;
  REAL kz, step, lz;
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
  lz = M_PI / (step * (REAL) (nz - 1));
  nz2 = nz / 2;

#pragma omp parallel for firstprivate(nz2,nx,ny,nz,nxy,step,gzvalue,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz;
      else
        kz = -((REAL) (nz - k)) * lz;
      gzvalue[ijnz + k] *= kz * I;
    }
  } 
}

/* 
 * @FUNC{rgrid_fft_laplace, "Laplacian of real grid (FFT)"}
 * @DESC{"Calculate second derivative of a grid (in reciprocal space).
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated"}
 * @ARG2{rgrid *laplace, "Putput grid"}
 * @RVAL{void, "No return value"}x
 *
 */

EXPORT void rgrid_fft_laplace(rgrid *grid, rgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nxy, nz;
  INT nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step;
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
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = M_PI / (step * (REAL) (nz - 1));
  
#pragma omp parallel for firstprivate(nx2,ny2,nz2,nx,ny,nz,nxy,step,lvalue,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    ijnz = ij * nz;
    
    if(i < nx2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;

    if(j < ny2) 
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;
      
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz;
      else
        kz = -((REAL) (nz - k)) * lz;
      lvalue[ijnz + k] *= -(kx * kx + ky * ky + kz * kz);
    }
  }
}

/*
 * @FUNC{rgrid_fft_laplace_x, "Second derivative of real grid with respect to x (FFT)"}
 * @DESC{"Differentiate real grid in the Fourier space along x twice.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *laplace_x, "Output grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_laplace_x(rgrid *grid, rgrid *laplace_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx, step, lx;
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
  lx = 2.0 * M_PI / (step * (REAL) nx);
  nx2 = nx / 2;

#pragma omp parallel for firstprivate(nx2,nx,ny,nz,nxy,step,gxvalue,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
    if(i < nx2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;
    for(k = 0; k < nz; k++)	  
      gxvalue[ijnz + k] *= -kx * kx;
  } 
}

/*
 * @FUNC{rgrid_fft_laplace_y, "Second derivative of real grid with respect to y (FFT)"}
 * @DESC{"Differentiate real grid in the Fourier space along y twice.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *laplace_y, "Output grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_laplace_y(rgrid *grid, rgrid *laplace_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky, step, ly;
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
  ly = 2.0 * M_PI / (step * (REAL) ny);
  ny2 = ny / 2;

#pragma omp parallel for firstprivate(ny2,nx,ny,nz,nxy,step,gyvalue,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
    if(j < ny2) 
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;
    for(k = 0; k < nz; k++)	  
      gyvalue[ijnz + k] *= -ky * ky;
  } 
}

/*
 * @FUNC{rgrid_fft_laplace_z, "Second derivative of real grid with respect to z (FFT)"}
 * @DESC{"Differentiate real grid in the Fourier space along z twice.
          Note that the input and output grids may be the same"}
 * @ARG1{rgrid *grid, "Grid to be differentiated (in reciprocal space)"}
 * @ARG2{rgrid *laplace_z, "Output grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_laplace_z(rgrid *grid, rgrid *laplace_z) {

  INT k, ij, ijnz, nx, ny, nxy, nz, nz2;
  REAL kz, step, lz;
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
  lz = M_PI / (step * (REAL) (nz - 1));
  nz2 = nz / 2;

#pragma omp parallel for firstprivate(nz2,nx,ny,nz,nxy,step,gzvalue,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz;
      else
        kz = -((REAL) (nz - k)) * lz;
      gzvalue[ijnz + k] *= -kz * kz;
    }
  } 
}

/*
 * @FUNC{rgrid_fft_laplace_expectation_value, "Expectation value of laplacian for real grid (FFT)"}
 * @DESC{"Calculate expectation value of laplace operator in the reciprocal space ($\int grid^* \Delta grid$)"}
 * @ARG1{rgrid *grid, "Source grid for the operation (in reciprocal space)"}
 * @RVAL{REAL, "Returns the expectation value"}
 *
 */

EXPORT REAL rgrid_fft_laplace_expectation_value(rgrid *grid)  {

  INT i, j, k, ij, ijnz, nx, ny, nxy, nz;
  INT nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step, sum = 0.0, ssum, norm = grid->fft_norm2;
  REAL complex *lvalue = (REAL complex *) grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_laplace_expectation_value(grid, &sum)) return sum;
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

  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = M_PI / (step * (REAL) (nz - 1));
  
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,lx,ly,lz) private(ssum,i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime) reduction(+:sum)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    ijnz = ij * nz;
    
    if(i < nx2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;
    if(j < ny2) 
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;
      
    ssum = 0.0;
    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz;
      else
        kz = -((REAL) (nz - k)) * lz;
      ssum -= (kx * kx + ky * ky + kz * kz) * csqnorm(lvalue[ijnz + k]);
    }
    sum += ssum;
  }
  return sum * norm;
}

/*
 * @FUNC{rgrid_fft_poisson, "Solve Poisson equation for real grid (FFT)"}
 * @DESC{"Solve Poisson equation: $\Delta f = u$ subject to periodic boundaries (in reciprocal space).
          Uses finite difference for Laplacian (7 point) and FFT"}
 * @ARG1{rgrid *grid, "On entry function $u$ specified over grid (input) and function $f$ (solution) on exit"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_poisson(rgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz, idx;
  REAL step = grid->step, step2 = step * step, ilx, ily;
  REAL ilz, kx, ky, kz;
  REAL complex *val = (REAL complex *) grid->value;

  if(grid->value_outside != RGRID_PERIODIC_BOUNDARY) {
    fprintf(stderr, "libgrid: Only periodic boundary Poisson solver implemented.\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_poisson(grid)) return;
#endif
  /* the folllowing is in Fourier space -> k = 0, nz */
  nz = grid->nz2 / 2; // nz2 = 2 * (nz/2 + 1)
  ilx = 2.0 * M_PI / ((REAL) nx);
  ily = 2.0 * M_PI / ((REAL) ny);
  ilz = M_PI / ((REAL) nz);  // TODO: why not nz-1 like everywhere else?
#pragma omp parallel for firstprivate(val, nx, ny, nz, grid, ilx, ily, ilz, step2) private(i, j, k, kx, ky, kz, idx) default(none) schedule(runtime)
  for(i = 0; i < nx; i++) {
    kx = COS(ilx * (REAL) i);
    for(j = 0; j < ny; j++) {
      ky = COS(ily * (REAL) j);
      for(k = 0; k < nz; k++) {
	kz = COS(ilz * (REAL) k);
	idx = (i * ny + j) * nz + k;
	if(i || j || k)
	  val[idx] = val[idx] * step2 / (2.0 * (kx + ky + kz - 3.0));
	else
	  val[idx] = 0.0;
      }
    }
  }
}

/*
 * @FUNC{rgrid_div, "Divergence of real vector field"}
 * @DESC{"Calculate divergence of a vector field. Uses either FD or FFT based on grid_anaylze_method"}
 * @ARG1{rgrid *div, "Result"}
 * @ARG2{rgrid *fx, "x component of the field"}
 * @ARG3{rgrid *fy, "y component of the field"}
 * @ARG4{rgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_div(rgrid *div, rgrid *fx, rgrid *fy, rgrid *fz) {

  if(grid_analyze_method) {
    rgrid_fft(fx);
    rgrid_fft(fy);
    rgrid_fft(fz);
    rgrid_fft_div(div, fx, fy, fz);
    rgrid_inverse_fft_norm(fx);
    rgrid_inverse_fft_norm(fy);
    rgrid_inverse_fft_norm(fz);
    rgrid_inverse_fft_norm(div);
  } else rgrid_fd_div(div, fx, fy, fz);
}

/*
 * @FUNC{rgrid_fft_div, "Divergence of real vector field (FFT)"}
 * @DESC{"Calculate divergence of a vector field (in reciprocal space)"}
 * @ARG1{rgrid *div, "Result"}
 * @ARG2{rgrid *fx, "x component of the field (in reciprocal space)"}
 * @ARG3{rgrid *fy, "y component of the field (in reciprocal space)"}
 * @ARG4{rgrid *fz, "z component of the field (in reciprocal space)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_div(rgrid *div, rgrid *fx, rgrid *fy, rgrid *fz) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, step, lx, ly, lz;
  REAL complex *adiv = (REAL complex *) div->value, *afx = (REAL complex *) fx->value, *afy = (REAL complex *) fy->value, *afz = (REAL complex *) fz->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_div(div, fx, fy, fz)) return;
#endif

  /* f'(x) = iF[ i kx F[f(x)] ] */  
  nx = div->nx;
  ny = div->ny;
  nz = div->nz2 / 2;
  nxy = nx * ny;
  step = div->step;
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = M_PI / (step * (REAL) (nz - 1));
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;

#pragma omp parallel for firstprivate(adiv,afx,afy,afz,nx2,ny2,nz2,nx,ny,nz,nxy,step,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    ijnz = ij * nz;

    if(i < nx2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;

    if(j < ny2) 
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;

    for(k = 0; k < nz; k++) {
      if(k < nz2) 
        kz = ((REAL) k) * lz;
      else
        kz = -((REAL) (nz - k)) * lz;
      adiv[ijnz + k] = I * (kx * afx[ijnz + k] + ky * afy[ijnz + k] + kz * afz[ijnz + k]);
    }
  }
}

/*
 * @FUNC{rgrid_fd_div, "Divergence of real vector field (finite difference)"}
 * @DESC{"Calculate divergence of a vector field (finite difference)"}
 * @ARG1{rgrid *div, "Result"}
 * @ARG2{rgrid *fx, "x component of the field"}
 * @ARG3{rgrid *fy, "y component of the field"}
 * @ARG4{rgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fd_div(rgrid *div, rgrid *fx, rgrid *fy, rgrid *fz) {

  INT i, j, k, ij, ijnz, ny = div->ny, nz = div->nz, nxy = div->nx * div->ny, nzz = div->nz2;
  REAL inv_delta = 1.0 / (2.0 * div->step);
  REAL *lvalue = div->value;
  
  if(div == fx || div == fy || div == fz) {
    fprintf(stderr, "libgrid: Destination grid must be different from input grids in rgrid_div().\n");
    abort();
  }

#ifdef USE_CUDA
// TODO: single GPU implementation missing
  cuda_remove_block(lvalue, 0);
  cuda_remove_block(fx->value, 1);
  cuda_remove_block(fy->value, 1);
  cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] = inv_delta * (rgrid_value_at_index(fx, i+1, j, k) - rgrid_value_at_index(fx, i-1, j, k)
	+ rgrid_value_at_index(fy, i, j+1, k) - rgrid_value_at_index(fy, i, j-1, k)
        + rgrid_value_at_index(fz, i, j, k+1) - rgrid_value_at_index(fz, i, j, k-1));
  }
}

/*
 * @FUNC{rgrid_rot, "Curl of real vector field"}
 * @DESC{"Calculate rot (curl; $\nabla\times$) of a vector field f = (fx, fy, fz)"}
 * @ARG1{rgrid *rotx, "x component of rot. If NULL, not computed (fx not accessed and may also be NULL)"}
 * @ARG2{rgrid *roty, "y component of rot. If NULL, not computed (fy not accessed and may also be NULL)"}
 * @ARG3{rgrid *rotz, "z component of rot. If NULL, not computed (fz not accessed and may also be NULL)"}
 * @ARG4{rgrid *fx, "x component of the field"}
 * @ARG5{rgrid *fy, "y component of the field"}
 * @ARG6{rgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
 *
 * TODO: CUDA implementation missing.
 *
 */

EXPORT void rgrid_rot(rgrid *rotx, rgrid *roty, rgrid *rotz, rgrid *fx, rgrid *fy, rgrid *fz) {

  INT i, j, k, ij, ijnz, ny, nz, nxy, nzz;
  REAL inv_delta;
  REAL *lvaluex, *lvaluey, *lvaluez;

  if(rotx == NULL && roty == NULL && rotz == NULL) return; /* Nothing to do */

  if(rotx || rotz) {
    ny = fy->ny;
    nz = fy->nz;
    nxy = fy->nx * fy->ny;
    nzz = fy->nz2;
    inv_delta = 1.0 / (2.0 * fy->step);
  } else {
    ny = fx->ny;
    nz = fx->nz;
    nxy = fx->nx * fx->ny;
    nzz = fx->nz2;
    inv_delta = 1.0 / (2.0 * fx->step);
  }

  if(rotx) lvaluex = rotx->value;
  else lvaluex = NULL;
  if(roty) lvaluey = roty->value;
  else lvaluey = NULL;
  if(rotz) lvaluez = rotz->value;
  else lvaluez = NULL;
  
#ifdef USE_CUDA
  /* This operation is carried out on the CPU rather than GPU (usually large grids, so they won't fit in GPU memory) */
  if(lvaluex) cuda_remove_block(lvaluex, 0);
  if(lvaluey) cuda_remove_block(lvaluey, 0);
  if(lvaluez) cuda_remove_block(lvaluez, 0);
  if(roty || rotz) cuda_remove_block(fx->value, 1);
  if(rotx || rotz) cuda_remove_block(fy->value, 1);
  if(rotx || roty) cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvaluex,lvaluey,lvaluez,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* x: (d/dy) fz - (d/dz) fy (no access to fx) */
      if(lvaluex)
        lvaluex[ijnz + k] = inv_delta * ((rgrid_value_at_index(fz, i, j+1, k) - rgrid_value_at_index(fz, i, j-1, k))
				      - (rgrid_value_at_index(fy, i, j, k+1) - rgrid_value_at_index(fy, i, j, k-1)));
      /* y: (d/dz) fx - (d/dx) fz (no access to fy) */
      if(lvaluey)
        lvaluey[ijnz + k] = inv_delta * ((rgrid_value_at_index(fx, i, j, k+1) - rgrid_value_at_index(fx, i, j, k-1))
				      - (rgrid_value_at_index(fz, i+1, j, k) - rgrid_value_at_index(fz, i-1, j, k)));
      /* z: (d/dx) fy - (d/dy) fx (no acess to fz) */
      if(lvaluez)
        lvaluez[ijnz + k] = inv_delta * ((rgrid_value_at_index(fy, i+1, j, k) - rgrid_value_at_index(fy, i-1, j, k))
    				      - (rgrid_value_at_index(fx, i, j+1, k) - rgrid_value_at_index(fx, i, j-1, k)));
    }
  }
}

/*
 * @FUNC{rgrid_abs_rot, "Norm of curl of real vector field"}
 * @DESC{"Calculate $|rot|$ ($|curl|$; $|\nabla\times|$) of a vector field (i.e., magnitude)"}
 * @ARG1{rgrid *rot, "Magnitude of rot"}
 * @ARG2{rgrid *fx, "x component of the field"}
 * @ARG3{rgrid *fy, "y component of the field"}
 * @ARG4{rgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_abs_rot(rgrid *rot, rgrid *fx, rgrid *fy, rgrid *fz) {

  INT i, j, k, ij, ijnz, ny = rot->ny, nz = rot->nz, nxy = rot->nx * rot->ny, nzz = rot->nz2;
  REAL inv_delta = 1.0 / (2.0 * rot->step);
  REAL *lvalue = rot->value, tmp;
  
  if(rot == fx || rot == fy || rot == fz) {
    fprintf(stderr, "libgrid: Source and destination grids must be different in rgrid_abs_rot().\n");
    abort();
  }
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_abs_rot(rot, fx, fy, fz, inv_delta, rgrid_bc_conv(rot))) return;
#endif

#pragma omp parallel for firstprivate(ny,nz,nzz,nxy,lvalue,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* x: (d/dy) fz - (d/dz) fy */
      tmp = inv_delta * ((rgrid_value_at_index(fz, i, j+1, k) - rgrid_value_at_index(fz, i, j-1, k))
				      - (rgrid_value_at_index(fy, i, j, k+1) - rgrid_value_at_index(fy, i, j, k-1)));
      lvalue[ijnz + k] = tmp * tmp;
      /* y: (d/dz) fx - (d/dx) fz */
      tmp = inv_delta * ((rgrid_value_at_index(fx, i, j, k+1) - rgrid_value_at_index(fx, i, j, k-1))
				      - (rgrid_value_at_index(fz, i+1, j, k) - rgrid_value_at_index(fz, i-1, j, k)));
      lvalue[ijnz + k] += tmp * tmp;
      /* z: (d/dx) fy - (d/dy) fx */
      tmp = inv_delta * ((rgrid_value_at_index(fy, i+1, j, k) - rgrid_value_at_index(fy, i-1, j, k))
				      - (rgrid_value_at_index(fx, i, j+1, k) - rgrid_value_at_index(fx, i, j-1, k)));
      lvalue[ijnz + k] += tmp * tmp;
      lvalue[ijnz + k] = SQRT(lvalue[ijnz + k]);
    }
  }
}
