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
#include "cprivate.h"

extern char grid_analyze_method;

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
 * @FUNC{cgrid_gradient, "Gradient of grid"}
 * @DESC{"Differentiate a grid with respect to x, y, z (gradient). Based on grid_analyze_method setting,
 *       either FFT or finite difference (central difference) will be used"}
 * @ARG1{cgrid *grid, "Grid to be differentiated"}
 * @ARG2{cgrid *gradx, "Gradient x-component"}
 * @ARG3{cgrid *grady, "Gradient y-component"}
 * @ARG4{cgrid *gradz, "Gradient z-component"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_gradient(cgrid *grid, cgrid *gradx, cgrid *grady, cgrid *gradz) {

  if(grid_analyze_method) {
    cgrid_copy(gradz, grid);
    cgrid_fft(gradz);
    cgrid_fft_gradient_x(gradz, gradx);
    cgrid_fft_gradient_y(gradz, grady);
    cgrid_fft_gradient_z(gradz, gradz);
    cgrid_inverse_fft_norm(gradx);
    cgrid_inverse_fft_norm(grady);
    cgrid_inverse_fft_norm(gradz);
  } else {
    cgrid_fd_gradient_x(grid, gradx);
    cgrid_fd_gradient_y(grid, grady);
    cgrid_fd_gradient_z(grid, gradz);
  }
}

/* 
 * @FUNC{cgrid_gradient_x, "Differentiate grid with respect to x"}
 * @DESC{"Differentiate a grid with respect to x. Uses grid_analyze_method to determine FFT vs. finite difference method"}
 * @ARG1{cgrid *grid, "Grid to be differentiated"}
 * @ARG2{cgrid *gradient, "Differentiated grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_gradient_x(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_x(gradient, gradient);
    cgrid_inverse_fft_norm(gradient);
  } else cgrid_fd_gradient_x(grid, gradient);
}

/* 
 * @FUNC{cgrid_gradient_y, "Differentiate grid with respect to y"}
 * @DESC{"Differentiate a grid with respect to y. Uses grid_analyze_method to determine FFT vs. finite difference method"}
 * @ARG1{cgrid *grid, "Grid to be differentiated"}
 * @ARG2{cgrid *gradient, "Differentiated grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_gradient_y(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_y(gradient, gradient);
    cgrid_inverse_fft_norm(gradient);
  } else cgrid_fd_gradient_y(grid, gradient);
}

/* 
 * @FUNC{cgrid_gradient_z, "Differentiate grid with respect to z"}
 * @DESC{"Differentiate a grid with respect to z. Uses grid_analyze_method to determine FFT vs. finite difference method"}
 * @ARG1{cgrid *grid, "Grid to be differentiated"}
 * @ARG2{cgrid *gradient, "Differentiated grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_gradient_z(cgrid *grid, cgrid *gradient) {

  if(grid_analyze_method) {
    cgrid_copy(gradient, grid);
    cgrid_fft(gradient);
    cgrid_fft_gradient_z(gradient, gradient);
    cgrid_inverse_fft_norm(gradient);
  } else cgrid_fd_gradient_z(grid, gradient);
}

/*
 * @FUNC{cgrid_laplace, "Laplacian of grid"}
 * @DESC{"Calculate laplacian of the grid. Uses finite difference or FFT according based on grid_analyze_method setting"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_laplace(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace(laplace, laplace);
    cgrid_inverse_fft_norm(laplace);
  } else cgrid_fd_laplace(grid, laplace);
}

/*
 * @FUNC{cgrid_laplace_x, "Second derivative of grid with respect to x"}
 * @DESC{"Calculate second derivative of the grid with respect to x. Uses FD or FFT according to grid_analuze_method"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_laplace_x(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_x(laplace, laplace);
    cgrid_inverse_fft_norm(laplace);
  } else cgrid_fd_laplace_x(grid, laplace);
}

/*
 * @FUNC{cgrid_laplace_y, "Second derivative of grid with respect to y"}
 * @DESC{"Calculate second derivative of the grid with respect to y. Uses FD or FFT according to grid_analuze_method"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_laplace_y(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_y(laplace, laplace);
    cgrid_inverse_fft_norm(laplace);
  } else cgrid_fd_laplace_y(grid, laplace);
}

/*
 * @FUNC{cgrid_laplace_z, "Second derivative of grid with respect to z"}
 * @DESC{"Calculate second derivative of the grid with respect to z. Uses FD or FFT according to grid_analuze_method"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_laplace_z(cgrid *grid, cgrid *laplace) {

  if(grid_analyze_method) {
    cgrid_copy(laplace, grid);
    cgrid_fft(laplace);
    cgrid_fft_laplace_z(laplace, laplace);
    cgrid_inverse_fft_norm(laplace);
  } else cgrid_fd_laplace_z(grid, laplace);
}

/* 
 * @FUNC{cgrid_fd_gradient_x, "First derivative of grid with respect to x (finite difference)"}
 * @DESC{"Calculate first derivative of the grid with respect to x using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_gradient_y, "First derivative of grid with respect to y (finite difference)"}
 * @DESC{"Calculate first derivative of the grid with respect to y using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_gradient_z, "First derivative of grid with respect to z (finite difference)"}
 * @DESC{"Calculate first derivative of the grid with respect to z using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_laplace, "Laplacian of grid (finite difference)"}
 * @DESC{"Calculate laplacian of the grid using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_laplace_x, "Second derivative of grid with respect to x (finite difference)"}
 * @DESC{"Calculate second derivative of the grid with respect to x using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplacex, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_laplace_y, "Second derivative of grid with respect to y (finite difference)"}
 * @DESC{"Calculate second derivative of the grid with respect to y using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplacey, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_laplace_z, "Second derivative of grid with respect to z (finite difference)"}
 * @DESC{"Calculate second derivative of the grid with respect to z using finite difference (central difference)"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplacez, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fft_gradient_x, "First derivative of grid with respect to x (FFT)"}
 * @DESC{"Calculate first derivative of the grid with respect to x using FFT. Note that the input and output
          grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient_x, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_gradient_x(cgrid *grid, cgrid *gradient_x) {

  INT i, k, ij, ijnz, nx, ny, nz, nxy;
  REAL kx, step;
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
  
  if (gradient_x != grid) cgrid_copy(gradient_x, grid);

#pragma omp parallel for firstprivate(nx,ny,nz,nxy,step,gxvalue) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L 
     * if k < n/2, k = k
     * else k = -k
     */
    if (i <= nx / 2)
      kx = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step);
    else 
      kx = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step);
      
    for(k = 0; k < nz; k++)	  
      gxvalue[ijnz + k] *= kx * I;
  }
}

/*
 * @FUNC{cgrid_fft_gradient_y, "First derivative of grid with respect to y (FFT)"}
 * @DESC{"Calculate first derivative of the grid with respect to y using FFT. Note that the input and output
          grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient_y, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_gradient_y(cgrid *grid, cgrid *gradient_y) {

  INT j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL ky, step;
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
  
  if (gradient_y != grid)
    cgrid_copy(gradient_y, grid);

#pragma omp parallel for firstprivate(nx,ny,nz,nxy,step,gyvalue) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L 
     * if k < n/2, k = k
     * else k = -k
     */
    if (j <= ny / 2)
      ky = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step);
    else 
      ky = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step);
      
    for(k = 0; k < nz; k++)	  
      gyvalue[ijnz + k] *= ky * I;
  }
}

/*
 * @FUNC{cgrid_fft_gradient_z, "First derivative of grid with respect to z (FFT)"}
 * @DESC{"Calculate first derivative of the grid with respect to z using FFT. Note that the input and output
          grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *gradient_z, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_gradient_z(cgrid *grid, cgrid *gradient_z) {

  INT k, ij, ijnz, nx, ny, nz, nxy;
  REAL kz, lz, step;
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
  
  if(gradient_z != grid) cgrid_copy(gradient_z, grid);

#pragma omp parallel for firstprivate(nx,ny,nz,nxy,step,gzvalue) private(ij,ijnz,k,kz,lz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L (L = n * step)
     * if k < n/2, k = k
     * else k = -k
     */
      
    lz = 2.0 * M_PI / (step * (REAL) nz);
    for(k = 0; k < nz; k++) {
      if (k <= nz / 2)
        kz = lz * (REAL) k;
      else 
        kz = lz * (REAL) (k - nz);
       
      gzvalue[ijnz + k] *= kz * I;
    }
  }    
}

/* 
 * @FUNC{cgrid_fft_laplace, "Laplacian of grid (FFT)"}
 * @DESC{"Calculate laplacian of the grid using FFT. Note that the input and output grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_laplace(cgrid *grid, cgrid *laplace)  {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step;
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
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
#pragma omp parallel for firstprivate(nx2,ny2,nz2,nx,ny,nz,nxy,step,lvalue,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
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
      kx = ((REAL) i) * lx;
    else 
      kx = ((REAL) (i - nx)) * lx;
      
    if (j <= ny2)
      ky = ((REAL) j) * ly;
    else 
      ky = ((REAL) (j - ny)) * ly;
      
    for(k = 0; k < nz; k++) {
      if (k <= nz2)
        kz = ((REAL) k) * lz;
      else 
        kz = ((REAL) (k - nz)) * lz;
        
      lvalue[ijnz + k] *= -(kx * kx + ky * ky + kz * kz);
    }
  }
}

/* 
 * @FUNC{cgrid_fft_laplace_x, "Second derivative of grid with respect to x (FFT)"}
 * @DESC{"Calculate second derivative of the grid with respect to x using FFT. Note that the input and output grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_laplace_x(cgrid *grid, cgrid *laplace)  {

  INT i, k, ij, ijnz, nx, ny, nz, nxy, nx2;
  REAL kx, lx, step;
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
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  nx2 = nx / 2;
#pragma omp parallel for firstprivate(nx2,nx,ny,nz,nxy,step,lvalue,lx) private(i,ij,ijnz,k,kx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L 
     * if k < n/2, k = k
     * else k = -k
     */
    if (i <= nx2)
      kx = ((REAL) i) * lx;
    else 
      kx = ((REAL) (i - nx)) * lx;
      
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] *= -kx * kx;
  }
}

/* 
 * @FUNC{cgrid_fft_laplace_y, "Second derivative of grid with respect to y (FFT)"}
 * @DESC{"Calculate second derivative of the grid with respect to y using FFT. Note that the input and output grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_laplace_y(cgrid *grid, cgrid *laplace)  {

  INT j, k, ij, ijnz, nx, ny, nz, nxy, ny2;
  REAL ky, ly, step;
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
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  ly = 2.0 * M_PI / (step * (REAL) ny);
  ny2 = ny / 2;
#pragma omp parallel for firstprivate(ny2,nx,ny,nz,nxy,step,lvalue,ly) private(j,ij,ijnz,k,ky) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    j = ij % ny;
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L 
     * if k < n/2, k = k
     * else k = -k
     */
      
    if (j <= ny2)
      ky = ((REAL) j) * ly;
    else 
      ky = ((REAL) (j - ny)) * ly;
      
    for(k = 0; k < nz; k++)
      lvalue[ijnz + k] *= -ky * ky;
  }
}

/* 
 * @FUNC{cgrid_fft_laplace_z, "Second derivative of grid with respect to z (FFT)"}
 * @DESC{"Calculate second derivative of the grid with respect to z using FFT. Note that the input and output grids may be the same"}
 * @ARG1{cgrid *grid, "Source grid"}
 * @ARG2{cgrid *laplace, "Output grid for the operation"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_laplace_z(cgrid *grid, cgrid *laplace)  {

  INT k, ij, ijnz, nx, ny, nz, nxy, nz2;
  REAL kz, lz, step;
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
  
  if (grid != laplace) cgrid_copy(laplace, grid);
  
  lz = 2.0 * M_PI / (step * (REAL) nz);
  nz2 = nz / 2;
#pragma omp parallel for firstprivate(nz2,nx,ny,nz,nxy,step,lvalue,lz) private(ij,ijnz,k,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
      
    /* 
     * k = 2 pi n / L 
     * if k < n/2, k = k
     * else k = -k
     */
    for(k = 0; k < nz; k++) {
      if (k <= nz2)
        kz = ((REAL) k) * lz;
      else 
        kz = ((REAL) (k - nz)) * lz;
        
      lvalue[ijnz + k] *= -kz * kz;
    }
  }
}

/*
 * @FUNC{cgrid_fft_laplace_expectation_value, "Expectation value of laplacian (FFT)"}
 * @DESC{"Calculate expectation value of laplacian in the reciprocal space ($\int grid^* \Delta grid$)"}
 * @ARG1{cgrid *grid, "Source grid for the operation (must be in reciprocal space)"}
 * @RVAL{REAL, "Returns the expectation value"}
 *
 */

EXPORT REAL cgrid_fft_laplace_expectation_value(cgrid *grid)  {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step, norm, sum = 0.0, ssum;
  REAL complex *lvalue = grid->value;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_expectation_value(grid, &sum)) return sum;
#endif

  /* int f*(x) f''(x) dx */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  /* norm = step^3 / (nx ny nz) */
  norm = grid->fft_norm2;
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
#pragma omp parallel for firstprivate(nx2,ny2,nz2,norm,nx,ny,nz,nxy,step,lvalue,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz,ssum) reduction(+:sum) default(none) schedule(runtime)
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
      kx = ((REAL) i) * lx;
    else 
      kx = ((REAL) (i - nx)) * lx;
      
    if (j <= ny2)
      ky = ((REAL) j) * ly;
    else 
      ky = ((REAL) (j - ny)) * ly;
      
    ssum = 0.0;
      
    for(k = 0; k < nz; k++) {
      if (k <= nz2)
        kz = ((REAL) k) * lz;
      else 
        kz = ((REAL) (k - nz)) * lz;
	
      ssum -= (kx * kx + ky * ky + kz * kz) * csqnorm(lvalue[ijnz + k]);
    }

    sum += ssum;
  }

  return sum * norm;
}

/*
 * @FUNC{cgrid_fft_poisson, "Solve Poisson equation (FFT)"}
 * @DESC{"Solve Poisson equation: $\Delta f = u$ subject to periodic boundary condition. Grid must be given in reciprocal space.
          This uses finite difference for Laplacian (7 point) -- see FFT. Num. Recip. Sect. 19.4. Note that this does not include
          the moving background in the calculation.
          To use this routine:\\
          // grid contains function u\\
          cgrid_fft(grid);\\
          cgrid_fft_poisson(grid);\\
          cgrid_inverse_fft_norm(grid);\\
          // grid contains solution f"}
 * @ARG1{cgrid *grid, "On entry Fourier transform of function u (input) and Fourier transform of function f on exit (output)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_poisson(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, idx; // could move some of the computations below past cuda call
  REAL step = grid->step, step2 = step * step, ilx = 2.0 * M_PI / ((REAL) nx), ily = 2.0 * M_PI / ((REAL) ny), ilz = 2.0 * M_PI / ((REAL) nz), kx, ky, kz;
  REAL complex *value = grid->value;

  if(grid->value_outside != CGRID_PERIODIC_BOUNDARY) {
    fprintf(stderr, "libgrid: Only periodic boundary Poisson solver implemented.\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_poisson(grid)) return;  
#endif
#pragma omp parallel for firstprivate(nx, ny, nz, value, ilx, ily, ilz, step2) private(i, j, k, kx, ky, kz, idx) default(none) schedule(runtime)
  for(i = 0; i < nx; i++) {
    kx = COS(ilx * ((REAL) i));
    for(j = 0; j < ny; j++) {
      ky = COS(ily * ((REAL) j));
      for(k = 0; k < nz; k++) {
	kz = COS(ilz * ((REAL) k));
	idx = (i * ny + j) * nz + k;
	if(i || j || k)
	  value[idx] *= step2 / (2.0 * (kx + ky + kz - 3.0));
	else
	  value[idx] = 0.0;
      }
    }
  }
}

/*
 * @FUNC{cgrid_fd_div, "Divergence of vector field (finite difference)"}
 * @DESC{"Calculate divergence of a vector field"}
 * @ARG1{cgird *div, "Result"}
 * @ARG2{cgrid *fx, "x component of the field"}
 * @ARG3{cgrid *fy, "y component of the field"}
 * @ARG4{cgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_fd_rot, "Rot of vector field (finite difference)"}
 * @DESC{"Calculate rot operator of a vector field"}
 * @ARG1{cgrid *rotx, "x component of rot. If NULL, not computed (fx not accessed and may also be NULL)"}
 * @ARG2{cgrid *roty, "y component of rot. If NULL, not computed (fy not accessed and may also be NULL)"}
 * @ARG3{cgrid *rotz, "z component of rot. If NULL, not computed (fz not accessed and may also be NULL)"}
 * @ARG4{cgrid *fx, "x component of the field"}
 * @ARG5{cgrid *fy, "y component of the field"}
 * @ARG6{cgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
 *
 * TODO: CUDA implementation missing.
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
 * @FUNC{cgrid_fd_abs_rot, "Norm of rot of vector field (finite difference)"}
 * @DESC{"Calculate $|rot|$ ($|curl|$; $|\nabla\times|$) of a vector field (i.e., the magnitude)"}
 * @ARG1{cgrid *rot, "Magnitudet of rot"}
 * @ARG2{cgrid *fx, "x component of the field"}
 * @ARG3{cgrid *fy, "y component of the field"}
 * @ARG4{cgrid *fz, "z component of the field"}
 * @RVAL{void, "No return value"}
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
