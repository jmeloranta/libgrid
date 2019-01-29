
 /*
 * Routines for complex grids.
 *
 * Nx is major index and Nz is minor index (varies most rapidly).
 *
 * For 2-D grids use: (1, NY, NZ)
 * For 1-D grids use: (1, 1, NZ)
 *
 */

#include "grid.h"
#include "private.h"

#ifdef USE_CUDA
static char cgrid_bc_conv(cgrid *grid) {

  if(grid->value_outside == CGRID_DIRICHLET_BOUNDARY) return 0;
  else if(grid->value_outside == CGRID_NEUMANN_BOUNDARY) return 1;
  else if(grid->value_outside == CGRID_PERIODIC_BOUNDARY) return 2;
  else {
    fprintf(stderr, "libgrid(cuda): Incompatible boundary condition.\n");
    exit(1);
  }
}
#endif

/*
 * Allocate complex grid.
 *
 * nx                 = number of points on the grid along x (INT).
 * ny                 = number of points on the grid along y (INT).
 * nz                 = number of points on the grid along z (INT).
 * step               = spatial step length on the grid (REAL).
 * value_outside      = condition for accessing boundary points:
 *                      CGRID_DIRICHLET_BOUNDARY: Dirichlet boundary
 *                      or CGRID_NEUMANN_BOUNDARY: Neumann boundary
 *                      or CGRID_PERIODIC_BOUNDARY: Periodic boundary
 *                      or CGRID_VORTEX_X_BOUNDARY: Vortex along X
 *                      or CGRID_VORTEX_Y_BOUNDARY: Vortex along Y
 *                      or CGRID_VORTEX_Z_BOUNDARY: Vortex along Z
 *                      or user supplied function with pointer to grid and
 *                         grid index as parameters to provide boundary access.
 * outside_params_ptr = pointer for passing parameters for the given boundary
 *                      access function. Use 0 to with the predefined boundary
 *                      functions (void *).
 * id                 = String ID describing the grid (char *; input).
 *
 * Return value: pointer to the allocated grid (cgrid *). Returns NULL on
 * error.
 *
 */

EXPORT cgrid *cgrid_alloc(INT nx, INT ny, INT nz, REAL step, REAL complex (*value_outside)(cgrid *grid, INT i, INT j, INT k), void *outside_params_ptr, char *id) {

  cgrid *grid;
  INT i;
  size_t len;
  
  if(!(grid = (cgrid *) malloc(sizeof(cgrid)))) {
    fprintf(stderr, "libgrid: Error in cgrid_alloc(). Could not allocate memory for grid structure.\n");
    abort();
  }

  grid->nx = nx;
  grid->ny = ny;
  grid->nz = nz;
  grid->grid_len = len = ((size_t) (nx * ny * nz)) * sizeof(REAL complex);
  grid->step = step;
  
#ifdef USE_CUDA
  if(cudaMallocHost((void **) &(grid->value), len) != cudaSuccess) { /* Use page-locked grids */
#else
#if defined(SINGLE_PREC)
  if (!(grid->value = (REAL complex *) fftwf_malloc(len))) {
#elif defined(DOUBLE_PREC)
  if (!(grid->value = (REAL complex *) fftw_malloc(len))) {
#elif defined(QUAD_PREC)
  if (!(grid->value = (REAL complex *) fftwl_malloc(len))) {
#endif
#endif
    fprintf(stderr, "libgrid: Error in cgrid_alloc(). Could not allocate memory for cgrid->value.\n");
    abort();
  }
  
  /* Set the origin of coordinates to its default value */
  grid->x0 = 0.0;
  grid->y0 = 0.0;
  grid->z0 = 0.0;
  /* Set the origin of momentum (i.e. frame of reference velocity) to its default value */
  grid->kx0 = 0.0;
  grid->ky0 = 0.0;
  grid->kz0 = 0.0;
  /* X-Y plane rotation frequency */
  grid->omega = 0.0;

  grid->fft_norm = 1.0 / (REAL) (grid->nx * grid->ny * grid->nz);

  // Account for the correct dimensionality of the grid
  grid->fft_norm2 = grid->fft_norm;
  if(grid->nx > 1) grid->fft_norm2 *= grid->step;
  if(grid->ny > 1) grid->fft_norm2 *= grid->step;
  if(grid->nz > 1) grid->fft_norm2 *= grid->step;

  strncpy(grid->id, id, 32);
  grid->id[31] = 0;
  
  if (value_outside)
    grid->value_outside = value_outside;
  else
    grid->value_outside = cgrid_value_outside_constantdirichlet;
    
  if (outside_params_ptr)
    grid->outside_params_ptr = outside_params_ptr;
  else {
    grid->default_outside_params = 0.0;
    grid->outside_params_ptr = &grid->default_outside_params;
  }
  
  grid->plan = grid->iplan = grid->implan = grid->iimplan = NULL;
#ifdef USE_CUDA
  grid->cufft_handle = -1;
#endif
  
#ifdef USE_CUDA
  cuda_set_gpu(CUDA_DEVICE);
  cgrid_cuda_init(sizeof(REAL complex) 
     * ((((size_t) nx) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
     * ((((size_t) ny) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
     * ((((size_t) nz) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK));  // reduction along blocks
#endif
  
  for(i = 0; i < nx * ny * nz; i++)
    grid->value[i] = 0.0;

  grid->flag = 0;

  return grid;
}

/*
 * "Clone" a complex grid with the parameters idential to the given grid (except new grid->value is allocated).
 *
 * grid = Grid to be cloned (cgrid *; input).
 * id   = ID string describing the grid (char *; input);
 *
 * Returns pointer to the new grid (rgrid *).
 *
 */

EXPORT cgrid *cgrid_clone(cgrid *grid, char *id) {

  cgrid *ngrid;
  size_t len = sizeof(REAL complex) * (size_t) (grid->nx * grid->ny * grid->nz);

  if(!(ngrid = (cgrid *) malloc(sizeof(cgrid)))) {
    fprintf(stderr, "libgrid: Out of memory in cgrid_clone().\n");
    exit(1);
  }
  bcopy((void *) grid, (void *) ngrid, sizeof(cgrid));
#ifdef USE_CUDA
  if(cudaMallocHost((void **) &(ngrid->value), len) != cudaSuccess) { /* Use page-locked grids */
#else
#if defined(SINGLE_PREC)
  if (!(ngrid->value = (REAL complex *) fftwf_malloc(len))) {
#elif defined(DOUBLE_PREC)
  if (!(ngrid->value = (REAL complex *) fftw_malloc(len))) {
#elif defined(QUAD_PREC)
  if (!(ngrid->value = (REAL complex *) fftwl_malloc(len))) {
#endif
#endif
    fprintf(stderr, "libgrid: Error in cgrid_clone(). Could not allocate memory for ngrid->value.\n");
    free(ngrid);
    return NULL;
  }
#ifdef USE_CUDA
  ngrid->cufft_handle = -1;
#endif
  ngrid->plan = ngrid->iplan = ngrid->implan = ngrid->iimplan = NULL;
  return ngrid;
}

/*
 * Claim grid (simple locking system for the workspace model).
 *
 * grid = Grid to be claimed (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_claim(cgrid *grid) {

  if(grid->flag) {
    fprintf(stderr, "libgrid: Attempting to claim grid twice.\n");
    abort();
  }
  grid->flag = 1;
}

/*
 * Release grid (simple locking system for the workspace model).
 *
 * grid = Grid to be claimed (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void cgrid_release(cgrid *grid) {

  if(!grid->flag) {
    fprintf(stderr, "libgrid: Attempting to release grid twice.\n");
    abort();
  }
  grid->flag = 0;
}

/*
 * Set the origin of coordinates. The coordinates of the grid will be:
 * 	x(i)  = (i - nx/2)* step - x0
 * 	y(j)  = (j - ny/2)* step - y0
 * 	z(k)  = (k - nz/2)* step - z0
 *
 * grid = grid whose origin will be set (cgrid *; input).
 * x0   = X origin (REAL; input).
 * y0   = Y origin (REAL; input).
 * z0   = Z origin (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_set_origin(cgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 = x0;
  grid->y0 = y0;
  grid->z0 = z0;
}

/*
 * Shift the origin to (x0, y0, z0).
 *
 * grid = grid whose origin is to be shifted (cgrid *; input).
 * x0 = X shift (REAL; input).
 * y0 = Y shift (REAL; input).
 * z0 = Z shift (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_shift_origin(cgrid *grid , REAL x0, REAL y0, REAL z0) {

  grid->x0 += x0;
  grid->y0 += y0;
  grid->z0 += z0;
}

/*
 * Set the origin in the momentum space (or the moving frame of reference).
 * kx0, ky0 and kz0 should be multiples of
 *  kx0min = 2.0 * M_PI / (NX * STEP) 
 *  ky0min = 2.0 * M_PI / (NY * STEP) 
 *  kz0min = 2.0 * M_PI / (NZ * STEP)
 *
 * kx0 = Momentum origin along x (REAL; input). 
 * ky0 = Momentum origin along y (REAL; input). 
 * kz0 = Momentum origin along z (REAL; input). 
 *
 * No return value.
 *
 */

EXPORT void cgrid_set_momentum(cgrid *grid, REAL kx0, REAL ky0, REAL kz0) {

  grid->kx0 = kx0;
  grid->ky0 = ky0;
  grid->kz0 = kz0;
}

/*
 * Set the rotation about the Z-axis.
 *
 * omega = rotation frequency about the Z axis (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_set_rotation(cgrid *grid, REAL omega) {

  grid->omega = omega;
}

/*
 * Free grid.
 *
 * grid = pointer to grid to be freed (cgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_free(cgrid *grid) {

  if (grid) {
#ifdef USE_CUDA
    cuda_remove_block(grid->value, 0);
    if(grid->value) cudaFreeHost(grid->value);
    if(grid->cufft_handle != -1) cufftDestroy(grid->cufft_handle);
#else
#if defined(SINGLE_PREC)
    if (grid->value) fftwf_free(grid->value);
#elif defined(DOUBLE_PREC)
    if (grid->value) fftw_free(grid->value);
#elif defined(QUAD_PREC)
    if (grid->value) fftwl_free(grid->value);
#endif
#endif
    cgrid_fftw_free(grid);
    free(grid);
  }
}

/* 
 * Write grid to disk in binary format.
 *
 * grid = grid to be written (cgrid *; input).
 * out  = file handle for the file (FILE * as defined in stdio.h; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_write(cgrid *grid, FILE *out) {

#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif
  fwrite(&grid->nx, sizeof(INT), 1, out);
  fwrite(&grid->ny, sizeof(INT), 1, out);
  fwrite(&grid->nz, sizeof(INT), 1, out);
  fwrite(&grid->step, sizeof(REAL), 1, out);
  fwrite(grid->value, sizeof(REAL complex), (size_t) (grid->nx * grid->ny * grid->nz), out);
}

/* 
 * Read complex grid from disk in binary format.
 *
 * grid = grid to be read (cgrid *; input).
 * in   = file handle for reading the file (FILE * as defined in stdio.h; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_read(cgrid *grid, FILE *in) {

  INT nx, ny, nz;
  REAL step;
  
#ifdef USE_CUDA
  cuda_remove_block(grid->value, 0);  // grid will be overwritten below
#endif
  fread(&nx, sizeof(INT), 1, in);
  fread(&ny, sizeof(INT), 1, in);
  fread(&nz, sizeof(INT), 1, in);
  fread(&step, sizeof(REAL), 1, in);
  
  if (nx != grid->nx || ny != grid->ny || nz != grid->nz || step != grid->step) {
    cgrid *tmp;

    fprintf(stderr, "libgrid: Grid in file has different size than grid in memory.\n");
    fprintf(stderr, "libgrid: Interpolating between grids.\n");
    if(!(tmp = cgrid_alloc(nx, ny, nz, step, grid->value_outside, NULL, "cgrid_read_temp"))) {
      fprintf(stderr, "libgrid: Error allocating grid in cgrid_read().\n");
      abort();
    }
    fread(tmp->value, sizeof(REAL complex), (size_t) (nx * ny * nz), in);
    cgrid_extrapolate(grid, tmp);
    cgrid_free(tmp);
    return;
  }
  
  fread(grid->value, sizeof(REAL complex), (size_t) (grid->nx * grid->ny * grid->nz), in);
}

/*
 * Write complex grid to disk including cuts along x, y, and z axes.
 *
 * basename = Base filename where suffixes (.x, .y, .z, and .grd) are added (char *; input).
 * grid     = Grid to be written to disk (cgrid *; input).
 * 
 * No return value.
 *
 * See also cgrid_write().
 *
 */

EXPORT void cgrid_write_grid(char *base, cgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;
  REAL complex tmp;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

  /* Write binary grid */
  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  cgrid_write(grid, fp);
  fclose(fp);

  /* Write cut along x-axis */
  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  j = ny / 2;
  k = nz / 2;
  for(i = 0; i < nx; i++) { 
    x = ((REAL) (i - nx / 2)) * step;
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", x, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);

  /* Write cut along y-axis */
  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2;
  k = nz / 2;
  for(j = 0; j < ny; j++) {
    y = ((REAL) (j - ny / 2)) * step;
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", y, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);

  /* Write cut along z-axis */
  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2;
  j = ny / 2;
  for(k = 0; k < nz; k++) {
    z = ((REAL) (k - nz / 2)) * step;
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", z, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);
}

/*
 * Read in a grid from a binary file (.grd).
 *
 * grid = grid where the data is placed (cgrid *, output).
 * file = filename for the file (char *, input). Note: the .grd extension must be given.
 *
 * No return value.
 *
 */

EXPORT void cgrid_read_grid(cgrid *grid, char *file) {

  FILE *fp;

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libgrid: Can't open complex grid file %s.\n", file);
    exit(1);
  }
  cgrid_read(grid, fp);
  fclose(fp);
}

/*
 * Copy grid from one grid to another.
 *
 * copy = destination grid (cgrid *; input).
 * grid = source grid (cgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_copy(cgrid *copy, cgrid *grid) {

  INT i, nx = grid->nx, nyz = grid->ny * grid->nz;
  size_t bytes = ((size_t) nyz) * sizeof(REAL complex);
  REAL complex *gvalue = grid->value;
  REAL complex *cvalue = copy->value;

  copy->nx = grid->nx;
  copy->ny = grid->ny;
  copy->nz = grid->nz;
  copy->step = grid->step;
  
  copy->x0 = grid->x0;
  copy->y0 = grid->y0;
  copy->z0 = grid->z0;
  copy->kx0 = grid->kx0;
  copy->ky0 = grid->ky0;
  copy->kz0 = grid->kz0;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_copy(copy, grid)) return;
#endif
  
#pragma omp parallel for firstprivate(nx,nyz,bytes,gvalue,cvalue) private(i) default(none) schedule(runtime)
  for(i = 0; i < nx; i++)
    memmove(&cvalue[i*nyz], &gvalue[i*nyz], bytes);
}

/*
 * Take complex conjugate of grid.
 * 
 * conjugate = destination for complex conjugated grid (cgrid *; input).
 * grid      = source grid for the operation (cgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination may be the same grid.
 * 
 */

EXPORT void cgrid_conjugate(cgrid *conjugate, cgrid *grid) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *cvalue = conjugate->value;
  REAL complex *gvalue = grid->value;
  
  conjugate->nx = grid->nx;
  conjugate->ny = grid->ny;
  conjugate->nz = grid->nz;
  conjugate->step = grid->step;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_conjugate(conjugate, grid)) return;
#endif
  
#pragma omp parallel for firstprivate(nxy,nz,cvalue,gvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = CONJ(gvalue[ijnz + k]);
  }
}

/*
 * Shift grid by given amount spatially.
 *
 * shifted = destination grid for the operation (cgrid *; output).
 * grid    = source grid for the operation (cgrid *; input).
 * x       = shift spatially by this amount in x (REAL; input).
 * y       = shift spatially by this amount in y (REAL; input).
 * z       = shift spatially by this amount in z (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_shift(cgrid *shifted, cgrid *grid, REAL x, REAL y, REAL z) {

  sShiftParametersc params;

  if(grid == shifted) {
    fprintf(stderr, "libgrid: Source and destination must be different in cgrid_shift().\n");
    abort();
  }
  /* shift by (x,y,z) i.e. current grid center to (x,y,z) */
  params.x = x;
  params.y = y;
  params.z = z;
  params.grid = grid;
  cgrid_map(shifted, shift_cgrid, &params);
}

/* 
 * Zero grid.
 *
 * grid = grid to be zeroed (cgrid *; ouput).
 *
 * No return value.
 * 
 */

EXPORT void cgrid_zero(cgrid *grid) { 

  cgrid_constant(grid, 0.0); 
}

/* 
 * Set grid to a constant value.
 *
 * grid = grid to be set (cgrid *; output).
 * c    = value (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_constant(cgrid *grid, REAL complex c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_constant(grid, c)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      value[ijnz + k] = c;
    }
  }
}

/*
 * Multiply grid by a function.
 *
 * grid = destination grid for the operation (cgrid *; input/output).
 * func = function providing the mapping (REAL complex (*)(void *, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data, next is the grid value,
 *        and x,y,z are the coordinates (REAL) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_product_func(cgrid *grid, REAL complex (*func)(void *arg, REAL complex val, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 1);
#endif

#pragma omp parallel for firstprivate(farg,nx,ny,nz,nxy,step,func,value,x0,y0,z0) private(i,j,ij,ijnz,k,x,y,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx/2)) * step - x0;
    y = ((REAL) (j - ny/2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz/2)) * step - z0;
      value[ijnz + k] *= func(farg, value[ijnz + k], x, y, z);
    }
  }
}

/*
 * Map function onto grid.
 *
 * grid = destination grid for the operation (cgrid *; output).
 * func = function providing the mapping (REAL complex (*)(void *, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data
 *        and x,y,z are the coordinates (REAL) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_map(cgrid *grid, REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 0);
#endif
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nxy,step,func,value,x0,y0,z0) private(i,j,ij,ijnz,k,x,y,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx/2)) * step - x0;
    y = ((REAL) (j - ny/2)) * step - y0;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz/2)) * step - z0;
      value[ijnz + k] = func(farg, x, y, z);
    }
  }
}

/*
 * Map a given function onto grid with linear "smoothing".
 * This can be used to weight the values at grid points to produce more
 * accurate integration over the grid.
 * *
 * grid = destination grid for the operation (cgrid *; output).
 * func = function providing the mapping (REAL complex (*)(void *, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data
 *        and (x, y, z) is the point (REALs) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 * ns   = number of intermediate points to be used in smoothing (INT; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_smooth_map(cgrid *grid, REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, INT ns) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL xc, yc, zc, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 0);
#endif

#pragma omp parallel for firstprivate(farg,nx,ny,nz,nxy,ns,step,func,value,x0,y0,z0) private(i,j,k,ijnz,xc,yc,zc) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    xc = ((REAL) (i - nx/2)) * step - x0;
    yc = ((REAL) (j - ny/2)) * step - y0;
    for(k = 0; k < nz; k++) {
      zc = ((REAL) (k - nz/2)) * step - z0;
      value[ijnz + k] = linearly_weighted_integralc(func, farg, xc, yc, zc, step, ns);
    }
  }
}

/*
 * Map a given function onto grid with linear "smoothing".
 * This can be used to weight the values at grid points to produce more
 * accurate integration over the grid. Limits for intermediate steps and
 * tolerance can be given.
 *
 * grid   = destination grid for the operation (cgrid *; output).
 * func   = function providing the mapping (REAL complex (*)(void *, REAL, REAL, REAL); input).
 *          The first argument (void *) is for external user specified data
 *          and x,y,z are the coordinates (REAL) where the function is evaluated.
 * farg   = pointer to user specified data (void *; input).
 * min_ns = minimum number of intermediate points to be used in smoothing (INT; input).
 * max_ns = maximum number of intermediate points to be used in smoothing (INT; input).
 * tol    = tolerance for weighing (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_adaptive_map(cgrid *grid, REAL complex (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, INT min_ns, INT max_ns, REAL tol) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, ns;
  REAL xc, yc, zc, step = grid->step;
  REAL tol2 = tol * tol;
  REAL complex sum, sump;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 0);
#endif

  if (min_ns < 1) min_ns = 1;
  if (max_ns < min_ns) max_ns = min_ns;
  
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nxy,min_ns,max_ns,step,func,value,tol2,x0,y0,z0) private(i,j,k,ijnz,ns,xc,yc,zc,sum,sump) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    xc = ((REAL) (i - nx/2)) * step - x0;
    yc = ((REAL) (j - ny/2)) * step - y0;
    for(k = 0; k < nz; k++) {
      zc = ((REAL) (k - nz/2)) * step - z0;
      sum  = func(farg, xc, yc, zc); sump = 0.0;
      for(ns = min_ns; ns <= max_ns; ns *= 2) {
        sum  = linearly_weighted_integralc(func, farg, xc, yc, zc, step, ns);
        sump = linearly_weighted_integralc(func, farg, xc, yc, zc, step, ns+1);
        if (sqnorm(sum - sump) < tol2) break;
      }
      
      /*
        if (ns >= max_ns)
        fprintf(stderr, "#");
        else if (ns > min_ns + 1)
        fprintf(stderr, "+");
        else
        fprintf(stderr, "-");
      */
      
      value[ijnz + k] = 0.5 * (sum + sump);
    }
    
    /*fprintf(stderr, "\n");*/
  }
}

/*
 * Add two grids: gridc = grida + gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st of the grids to be added (cgrid *; input).
 * gridb = 2nd of the grids to be added (cgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_sum(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_sum(gridc, grida, gridb)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] + bvalue[ijnz + k];
  }
}

/* 
 * Subtract two grids: gridc = grida - gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 *
 * No return value.
 *
 * Note: both source and destination may be the same.
 *
 */

EXPORT void cgrid_difference(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_difference(gridc, grida, gridb)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] - bvalue[ijnz + k];
  }
}

/* 
 * Calculate product of two grids: gridc = grida * gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_product(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_product(gridc, grida, gridb)) return;
#endif  
#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] * bvalue[ijnz + k];
  }
}

/* 
 * Rise absolute value of a grid to given power.
 *
 * gridb    = destination grid (cgrid *; output).
 * grida    = 1st source grid (cgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 * No return value.
 *
 * Notes: - Source and destination grids may be the same.
 *        - This routine uses pow() so that the exponent can be
 *          fractional but this is slow! Do not use this for integer
 *          exponents.
 *
 */

EXPORT void cgrid_abs_power(cgrid *gridb, cgrid *grida, REAL exponent) {

  INT ij, k, ijnz, nxy = gridb->nx * gridb->ny, nz = gridb->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_abs_power(gridb, grida, exponent)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = POW(CABS(avalue[ijnz + k]), exponent);
  }
}

/* 
 * Rise grid to given power.
 *
 * gridb    = destination grid (cgrid *; output).
 * grida    = 1st source grid (cgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 * No return value.
 *
 * Notes: - Source and destination grids may be the same.
 *
 */

EXPORT void cgrid_power(cgrid *gridb, cgrid *grida, REAL exponent) {

  INT ij, k, ijnz, nxy = gridb->nx * gridb->ny, nz = gridb->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_power(gridb, grida, exponent)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = CPOW(avalue[ijnz + k], exponent);
  }
}

/*
 * Divide two grids: gridc = grida / gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 *
 * No return value.
 *
 * Note: Source and destination grids may be the same.
 *
 */

EXPORT void cgrid_division(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_division(gridc, grida, gridb)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] / bvalue[ijnz + k];
  }
}

/*
 * "Safely" divide two grids: gridc = grida / gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 * eps   = epsilon to add to the divisor.
 *
 * No return value.
 *
 * Note: Source and destination grids may be the same.
 *
 */

EXPORT void cgrid_division_eps(cgrid *gridc, cgrid *grida, cgrid *gridb, REAL eps) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_division_eps(gridc, grida, gridb, eps)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue,eps) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] / (bvalue[ijnz + k] + eps);
  }
}

/* 
 * Conjugate product of two grids: gridc = CONJ(grida) * gridb
 *
 * gridc = destination grid (cgrid *; output).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 *
 * No return value.
 * 
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_conjugate_product(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_conjugate_product(gridc, grida, gridb)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = CONJ(avalue[ijnz + k]) * bvalue[ijnz + k];
  }
}

/*
 * Add a constant to a grid.
 *
 * grid = grid where the constant is added (cgrid *; output).
 * c    = constant to be added (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_add(cgrid *grid, REAL complex c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_add(grid, c)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] += c;
  }
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (cgrid *; output).
 * c    = multiplier (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_multiply(cgrid *grid, REAL complex c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_multiply(grid, c)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= c;
  }
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (cgrid *; input/output).
 * ca   = constant to be added (REAL complex; input).
 * cm   = multiplier (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_add_and_multiply(cgrid *grid, REAL complex ca, REAL complex cm) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_add_and_multiply(grid, ca, cm)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,value,ca,cm) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] = (value[ijnz + k] + ca) * cm;
  }
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (cgrid *; input/output).
 * cm   = multiplier (REAL complex; input).
 * ca   = constant to be added (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_multiply_and_add(cgrid *grid, REAL complex cm, REAL complex ca) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_multiply_and_add(grid, cm, ca)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,value,ca,cm) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] = value[ijnz + k] * cm + ca;
  }
}

/* 
 * Add scaled grid (multiply/add): gridc = gridc + d * grida
 *
 * gridc = destination grid for the operation (cgrid *; input/output).
 * d     = multiplier for the operation (REAL complex; input).
 * grida = source grid for the operation (cgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_add_scaled(cgrid *gridc, REAL complex d, cgrid *grida) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_add_scaled(gridc, d, grida)) return;
#endif

#pragma omp parallel for firstprivate(d,nxy,nz,avalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] += d * avalue[ijnz + k];
  }
}

/*
 * Perform the following operation: gridc = gridc + d * grida * gridb.
 *
 * gridc = destination grid (cgrid *; input/output).
 * d     = constant multiplier (REAL complex; input).
 * grida = 1st source grid (cgrid *; input).
 * gridb = 2nd source grid (cgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_add_scaled_product(cgrid *gridc, REAL complex d, cgrid *grida, cgrid *gridb) {
  
  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_add_scaled_product(gridc, d, grida, gridb)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue,d) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] += d * avalue[ijnz + k] * bvalue[ijnz + k];
  }
}

/*
 * Operate on a grid by a given operator: gridc = O(grida).
 *
 * gridc    = destination grid (cgrid *; output).
 * grida    = source grid (cgrid *; input).
 * operator = operator (REAL complex (*)(REAL complex, void *); input). Args are value and params.
 *            (i.e. a function mapping a given C-number to another)
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_operate_one(cgrid *gridc, cgrid *grida, REAL complex (*operator)(REAL complex a, void *params), void *params) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    cuda_remove_block(avalue, 1);
    if(avalue != cvalue) cuda_remove_block(cvalue, 0);
  }
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,cvalue,operator,params) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = operator(avalue[ijnz + k], params);
  }
}

/*
 * Operate on a grid by a given operator and multiply: gridc = gridb * O(grida).
 *
 * gridc    = destination grid (cgrid *; output).
 * gridb    = multiply with this grid (cgrid *; input).
 * grida    = source grid (cgrid *; input).
 * operator = operator (REAL complex (*)(REAL complex, void *); input). Args are value and params.
 *            (i.e., a function mapping a given C-number to another)
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void cgrid_operate_one_product(cgrid *gridc, cgrid *gridb, cgrid *grida, REAL complex (*operator)(REAL complex a, void *params), void *params) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    if(gridc != gridb && gridc != grida) cuda_remove_block(cvalue, 0);
    cuda_remove_block(bvalue, 1);
    if(grida != gridb) cuda_remove_block(avalue, 1);
  }
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue,operator,params) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = bvalue[ijnz + k] * operator(avalue[ijnz + k], params);
  }
}

/* 
 * Operate on two grids and place the result in third: gridc = O(grida, gridb).
 * where O is the operator.
 *
 * gridc    = destination grid (cgrid *; output).
 * grida    = 1s source grid (cgrid *; input).
 * gridb    = 2nd source grid (cgrid *; input).
 * operator = operator mapping grida and gridb (REAL complex (*)(REAL
 *            complex, REAL complex); input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void cgrid_operate_two(cgrid *gridc, cgrid *grida, cgrid *gridb, REAL complex (*operator)(REAL complex a, REAL complex b)) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  REAL complex *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    if(gridc != gridb && gridc != grida) cuda_remove_block(cvalue, 0);
    cuda_remove_block(bvalue, 1);
    if(grida != gridb) cuda_remove_block(avalue, 1);
  }
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,cvalue,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = operator(avalue[ijnz + k], bvalue[ijnz + k]);
  }
}

/*
 * Operate on a grid by a given operator.
 *
 * grid     = grid to be operated (cgrid *; output).
 * operator = operator (void (*)(REAL complex *); input).
 * 
 * No return value.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void cgrid_transform_one(cgrid *grid, void (*operator)(REAL complex *a)) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 1);
#endif

#pragma omp parallel for firstprivate(nxy,nz,value,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      operator(&value[ijnz + k]);
  }
}

/*
 * Operate on two separate grids by a given operator.
 *
 * grida    = grid to be operated (cgrid *; input/output).
 * gridb    = grid to be operated (cgrid *; input/output).
 * operator = operator (void (*)(REAL complex *); input).
 * 
 * No return value.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void cgrid_transform_two(cgrid *grida, cgrid *gridb, void (*operator)(REAL complex *a, REAL complex *b)) {

  INT ij, k, ijnz, nxy = grida->nx * grida->ny, nz = grida->nz;
  REAL complex *avalue = grida->value;
  REAL complex *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    cuda_remove_block(bvalue, 1);
    cuda_remove_block(avalue, 1);
  }
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      operator(&avalue[ijnz + k], &bvalue[ijnz + k]);
  }
}

/*
 * Integrate over a grid.
 *
 * grid = grid to be integrated (cgrid *; input).
 *
 * Returns the integral value (REAL complex).
 *
 */

EXPORT REAL complex cgrid_integral(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny =  grid->ny, nz = grid->nz;
  REAL complex sum = 0.0, step = grid->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_integral(grid, &sum)) return sum;
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,grid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += cgrid_value_at_index(grid, i, j, k);
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}

/*
 * Integrate over a grid with limits.
 *
 * grid = grid to be integrated (cgrid *; input).
 * xl   = lower limit for x (REAL; input).
 * xu   = upper limit for x (REAL; input).
 * yl   = lower limit for y (REAL; input).
 * yu   = upper limit for y (REAL; input).
 * zl   = lower limit for z (REAL; input).
 * zu   = upper limit for z (REAL; input).
 *
 * Returns the integral value (REAL complex).
 *
 */

EXPORT REAL complex cgrid_integral_region(cgrid *grid, REAL xl, REAL xu, REAL yl, REAL yu, REAL zl, REAL zu) {

  INT iu, il, i, ju, jl, j, ku, kl, k, nx = grid->nx, ny = grid->ny;
  REAL complex sum;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL step = grid->step;
   
  il = grid->nx / 2 + (INT) ((xl + x0) / step);
  iu = grid->nx / 2 + (INT) ((xu + x0) / step);
  jl = grid->ny / 2 + (INT) ((yl + y0) / step);
  ju = grid->ny / 2 + (INT) ((yu + y0) / step);
  kl = grid->nz / 2 + (INT) ((zl + z0) / step);
  ku = grid->nz / 2 + (INT) ((zu + z0) / step);
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_integral_region(grid, il, iu, jl, ju, kl, ku, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(il,iu,jl,ju,kl,ku,grid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = il; i <= iu; i++)
    for (j = jl; j <= ju; j++)
      for (k = kl; k <= ku; k++)
	sum += cgrid_value_at_index(grid, i, j, k);
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}
 
/* 
 * Integrate over the grid squared (int |grid|^2).
 *
 * grid = grid to be integrated (cgrid *; input).
 *
 * Returns the integral (REAL complex).
 *
 */

EXPORT REAL cgrid_integral_of_square(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL sum = 0, step = grid->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_integral_of_square(grid, &sum)) return sum;
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,grid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += sqnorm(cgrid_value_at_index(grid, i, j, k));
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}

/*
 * Calculate overlap between two grids (int grida^*gridb).
 *
 * grida = 1st grid (complex conjugated) (cgrid *; input).
 * gridb = 2nd grid (no complex conjugation) (cgrid *; input).
 *
 * Returns the value of the overlap integral (REAL complex).
 *
 */

EXPORT REAL complex cgrid_integral_of_conjugate_product(cgrid *grida, cgrid *gridb) {

  INT i, j, k, nx = grida->nx, ny = grida->ny, nz = grida->nz;
  REAL complex sum = 0.0, step = grida->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_integral_of_conjugate_product(grida, gridb, &sum)) return sum;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,grida,gridb) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += CONJ(cgrid_value_at_index(grida, i, j, k)) * cgrid_value_at_index(gridb, i, j, k);

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int gridb^* grida gridb = int gridb |grida|^2).
 *
 * grida = grid giving the probability (|grida|^2) (cgrid *; input).
 * gridb = grid to be averaged (cgrid *; input).
 *
 * Returns the average value (REAL complex).
 *
 */

EXPORT REAL complex cgrid_grid_expectation_value(cgrid *grida, cgrid *gridb) {

  INT i, j, k, nx = grida->nx, ny = grida->ny , nz = grida->nz;
  REAL complex sum = 0.0, step = grida->step;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_grid_expectation_value(grida, gridb, &sum)) return sum;
#endif
  
#pragma omp parallel for firstprivate(nx,ny,nz,grida,gridb) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += sqnorm(cgrid_value_at_index(grida, i, j, k)) * cgrid_value_at_index(gridb, i, j, k);
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}
 
/*
 * Calculate the expectation value of a function over a grid.
 * (int grida^* func grida = int func |grida|^2).
 *
 * func  = function to be averaged (REAL complex (*)(void *, REAL complex, REAL, REAL, REAL); input).
 *         The arguments are: optional arg, grida(x,y,z), x, y, z.
 * grida = grid giving the probability (|grida|^2) (cgrid *; input).
 *
 * Returns the average value (REAL complex).
 *
 */
 
EXPORT REAL complex cgrid_grid_expectation_value_func(void *arg, REAL complex (*func)(void *arg, REAL complex val, REAL x, REAL y, REAL z), cgrid *grida) {
   
  INT i, j, k, nx = grida->nx, ny = grida->ny, nz = grida->nz;
  REAL complex sum = 0.0, tmp;
  REAL x0 = grida->x0, y0 = grida->y0, z0 = grida->z0;
  REAL x, y, z, step = grida->step;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grida->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,grida,x0,y0,z0,step,func,arg) private(x,y,z,i,j,k,tmp) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx/2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny/2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz/2)) * step - z0;
	tmp = cgrid_value_at_index(grida, i, j, k);
	sum += sqnorm(tmp) * func(arg, tmp, x, y, z);
      }
    }
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}

/* 
 * Integrate over the grid multiplied by weighting function (int grid w(x)).
 *
 * grid   = grid to be integrated over (cgrid *; input).
 * weight = function defining the weight (REAL complex (*)(REAL, REAL, REAL); input). The arguments are (x,y,z) coordinates.
 * farg   = argument to the weight function (void *; input).
 *
 * Returns the value of the integral (REAL complex).
 *
 */

EXPORT REAL complex cgrid_weighted_integral(cgrid *grid, REAL complex (*weight)(void *farg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;
  REAL complex sum = 0.0;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,grid,x0,y0,z0,step,weight,farg) private(x,y,z,i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx / 2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny / 2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz / 2)) * step - z0;
	sum += weight(farg, x, y, z) * cgrid_value_at_index(grid, i, j, k);
      }
    }
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
}

/* 
 * Integrate over square of the grid multiplied by weighting function (int grid^2 w(x)).
 *
 * grid   = grid to be integrated over (cgrid *; input).
 * weight = function defining the weight (REAL complex (*)(REAL, REAL, REAL); input).
 *          The arguments are (x,y,z) coordinates.
 * farg   = argument to the weight function (void *; input).
 *
 * Returns the value of the integral (REAL complex).
 *
 */

EXPORT REAL cgrid_weighted_integral_of_square(cgrid *grid, REAL (*weight)(void *farg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;
  REAL sum = 0;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,grid,x0,y0,z0,step,weight,farg) private(x,y,z,i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx / 2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny / 2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz / 2)) * step - z0;
	sum += weight(farg, x, y, z) * sqnorm(cgrid_value_at_index(grid, i, j, k));
      }
    }
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  return sum * step;
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
 * Calculate gradient of a grid.
 *
 * grid       = grid to be differentiated twice (cgrid *; input).
 * gradient_x = x output grid for the operation (cgrid *; output).
 * gradient_y = y output grid for the operation (cgrid *; output).
 * gradient_z = z output grid for the operation (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_gradient(cgrid *grid, cgrid *gradient_x, cgrid *gradient_y, cgrid *gradient_z) {

  cgrid_fd_gradient_x(grid, gradient_x);
  cgrid_fd_gradient_y(grid, gradient_y);
  cgrid_fd_gradient_z(grid, gradient_z);
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
 * Calculate vector laplacian of the grid (x component). This is the second derivative with respect to x.
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
 * Calculate vector laplacian of the grid (y component). This is the second derivative with respect to y.
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
 * Calculate dot product of the gradient of the grid.
 *
 * grid          = source grid for the operation (cgrid *; input).
 * grad_dot_grad = destination grid (cgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_fd_gradient_dot_gradient(cgrid *grid, cgrid *grad_dot_grad) {

  INT i, j, k, ij, ijnz;
  INT ny = grid->ny, nz = grid->nz;
  INT nxy = grid->nx * grid->ny;
  REAL inv_2delta2 = 1.0 / (2.0 * grid->step * 2.0 * grid->step);
  REAL complex *gvalue = grad_dot_grad->value;
  
  if(grid == grad_dot_grad) {
    fprintf(stderr, "libgrid: source and destination must be different in cgrid_fd_gradient_dot_gradient().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fd_gradient_dot_gradient(grid, grad_dot_grad, inv_2delta2, cgrid_bc_conv(grid))) return;
#endif

/*  grad f(x,y,z) dot grad f(x,y,z) = [ |f(+,0,0) - f(-,0,0)|^2 + |f(0,+,0) - f(0,-,0)|^2 + |f(0,0,+) - f(0,0,-)|^2 ] / (2h)^2 */
#pragma omp parallel for firstprivate(ny,nz,nxy,gvalue,inv_2delta2,grid) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      gvalue[ijnz + k] = inv_2delta2 * 
	(sqnorm(cgrid_value_at_index(grid, i, j, k + 1) - cgrid_value_at_index(grid, i, j, k - 1)) 
         + sqnorm(cgrid_value_at_index(grid, i, j + 1, k) - cgrid_value_at_index(grid, i, j - 1, k))
	 + sqnorm(cgrid_value_at_index(grid, i + 1, j, k) - cgrid_value_at_index(grid, i - 1, j, k)));
  }
}

/*
 * Print the grid with both real and imaginary parts into file (ASCII format).
 *
 * grid = grid to be printed out (cgrid *; input).
 * out  = output file pointer (FILE *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_print(cgrid *grid, FILE *out) {

  INT i, j, k;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx; i++) {
    for(j = 0; j < grid->ny; j++) {
      for(k = 0; k < grid->nz; k++) {
        fprintf(out, FMT_R " " FMT_R,
		CREAL(cgrid_value_at_index(grid, i, j, k)),
		CIMAG(cgrid_value_at_index(grid, i, j, k)));
	  }
      fprintf(out, "\n");
    }
    fprintf(out, "\n");
  }
}

/*
 * Perform Fast Fourier Transformation of a grid.
 *
 * grid = grid to be Fourier transformed (input/output) (cgrid *; input/output).
 *
 * No return value.
 *
 * Notes: - The input grid is overwritten with the output.
 *        - No normalization is performed.
 *
 */

EXPORT void cgrid_fft(cgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cufft_fft(grid)) return;
#endif

  cgrid_fftw(grid);
}

/*
 * Perform inverse Fast Fourier Transformation of a grid.
 *
 * grid = grid to be inverse Fourier transformed (input/output) (cgrid *; input/output).
 *
 * No return value.
 *
 * Notes: - The input grid is overwritten with the output.
 *        - No normalization.
 *
 */

EXPORT void cgrid_inverse_fft(cgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cufft_fft_inv(grid)) return;
#endif

  cgrid_fftw_inv(grid);
}

/*
 * Perform scaled inverse Fast Fourier Transformation of a grid.
 *
 * grid = grid to be inverse Fourier transformed (cgrid *; input/output).
 * c    = scaling factor (i.e. the output is multiplied by this constant) (REAL complex; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_scaled_inverse_fft(cgrid *grid, REAL complex c) {

  cgrid_inverse_fft(grid);
  cgrid_multiply(grid, c);  
}

/*
 * Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm.
 *
 * grid = grid to be inverse Fourier transformed (input/output) (cgrid *; input/output).
 *
 * No return value.
 *
 * Note: The input grid is overwritten with the output.
 *
 */

EXPORT void cgrid_inverse_fft_norm(cgrid *grid) {

  cgrid_scaled_inverse_fft(grid, grid->fft_norm);
}

/*
 * Convolute FFT transformed grids. 
 *
 * To convolute grids grida and gridb and place the result in gridc:
 * cgrid_fft(grida);
 * cgrid_fft(gridb);
 * cgrid_convolue(gridc, grida, gridb);
 * cgrid_inverse_fft(gridc);
 * gridc now contains the convolution of grida and gridb.
 *
 * gridc = output grid (cgrid *; output).
 * grida = 1st grid to be convoluted (cgrid *; input).
 * gridb = 2nd grid to be convoluted (cgrid *; input).
 *
 * No return value.
 *
 * Note: the input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_convolute(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL step, norm;
  REAL complex *cvalue, *bvalue, *avalue;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_convolute(gridc, grida, gridb)) return;
#endif

  /* int f(r) g(r-r') d^3r' = iF[ F[f] F[g] ] = (step / N)^3 iFFT[ FFT[f] FFT[g] ] */
  nx = gridc->nx;
  ny = gridc->ny;
  nz = gridc->nz;
  nxy = nx * ny;
  step = gridc->step;
  
  cvalue = gridc->value;
  bvalue = gridb->value;
  avalue = grida->value; 
 
  norm = step * step * step * grida->fft_norm;         /* David: fft_norm */
  
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,cvalue,bvalue,avalue,norm) private(i,j,ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* if odd */
      if ((i + j + k) & 1)
        cvalue[ijnz + k] = -norm * avalue[ijnz + k] * bvalue[ijnz + k];
      else
        cvalue[ijnz + k] = norm * avalue[ijnz + k] * bvalue[ijnz + k];
    }
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
  
  /* David: fft_norm */
  norm = grid->fft_norm;
  
  if (gradient_x != grid) cgrid_copy(gradient_x, grid);

  if(grid -> value_outside == CGRID_NEUMANN_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_X_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Y_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Z_BOUNDARY) {
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
  if(grid -> value_outside == CGRID_NEUMANN_BOUNDARY  ||
     grid -> value_outside == CGRID_VORTEX_X_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Y_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Z_BOUNDARY ) {
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

  if(grid -> value_outside == CGRID_NEUMANN_BOUNDARY  ||
     grid -> value_outside == CGRID_VORTEX_X_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Y_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Z_BOUNDARY ) {
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
 * Calculate gradient of a grid (in Fourier space).
 *
 * grid       = grid to be differentiated (cgrid *; input).
 * gradient_x = x output grid (cgrid *; output).
 * gradient_y = y output grid (cgrid *; output).
 * gradient_z = z output grid (cgrid *; output).
 *
 * No return value.
 *
 * Note: input/output grids may be the same.
 *
 */

EXPORT void cgrid_fft_gradient(cgrid *grid, cgrid *gradient_x, cgrid *gradient_y, cgrid *gradient_z) {

  cgrid_fft_gradient_x(grid, gradient_x);
  cgrid_fft_gradient_y(grid, gradient_y);
  cgrid_fft_gradient_z(grid, gradient_z);
}

/* 
 * Calculate second derivative of a grid (in Fourier space).
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

  INT i, j, k, ij, ijnz, nx,ny,nz, nxy;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL kx, ky, kz, lz, step, norm;
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
  
  if(grid -> value_outside == CGRID_NEUMANN_BOUNDARY  ||
     grid -> value_outside == CGRID_VORTEX_X_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Y_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Z_BOUNDARY ) {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,lz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      kx = M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
      ky = M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
      lz = ((REAL) nz) * step;
      for(k = 0; k < nz; k++) {
        kz = M_PI * ((REAL) k) / lz - kz0;
        lvalue[ijnz + k] *= (-kx * kx -ky * ky -kz * kz) * norm;
      }
    }    
  } else {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,lz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * k = 2 pi n / L 
       * if k < n/2, k = k
       * else k = -k
       */
      if (i <= nx / 2)
        kx = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
      else 
        kx = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step) -kx0;
      
      if (j <= ny / 2)
        ky = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
      else 
        ky = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - ky0;
      
      lz = ((REAL) nz) * step;
      for(k = 0; k < nz; k++) {
        if (k <= nz / 2)
          kz = 2.0 * M_PI * ((REAL) k) / lz - kz0;
        else 
          kz = 2.0 * M_PI * ((REAL) (k - nz)) / lz - kz0;
        
        lvalue[ijnz + k] *= -(kx * kx + ky * ky + kz * kz) * norm;
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

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL kx, ky, kz, lz, step, norm, sum = 0.0, ssum;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL complex *lvalue = laplace->value;
  REAL aux;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_laplace_expectation_value(grid, laplace, &sum)) return sum;
#endif

  /* int f*(x) f''(x) dx */
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  /* int (delta FFT[f(x)] )^2 dk => delta^2 / N delta */
  /* David: Normalization */
  norm = grid->fft_norm;
  norm = step * step * step * grid->fft_norm;
  
  if(grid != laplace) cgrid_copy(laplace, grid);

  if(grid -> value_outside == CGRID_NEUMANN_BOUNDARY  ||
     grid -> value_outside == CGRID_VORTEX_X_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Y_BOUNDARY ||
     grid -> value_outside == CGRID_VORTEX_Z_BOUNDARY ) {
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,lz, ssum, aux) reduction(+:sum) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
   
      kx = M_PI * ((REAL) i) / (((REAL) nx) * step) - kx0;
      ky = M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
    
      ssum = 0.0;
      lz = ((REAL) nz) * step;
    
      for(k = 0; k < nz; k++) {
        kz = M_PI * ((REAL) k) / lz - kz0;
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
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,step,lvalue,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,lz, ssum) reduction(+:sum) default(none) schedule(runtime)
   for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
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
    
    if (j <= ny / 2)
      ky = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - ky0;
    else 
      ky = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - ky0;
    
    ssum = 0.0;
    lz = ((REAL) nz) * step;
    
    for(k = 0; k < nz; k++) {
      if (k <= nz / 2)
        kz = 2.0 * M_PI * ((REAL) k) / lz - kz0;
      else 
        kz = 2.0 * M_PI * ((REAL) (k - nz)) / lz - kz0;
      
      ssum += (-kx * kx - ky * ky - kz * kz) * sqnorm(lvalue[ijnz + k]);
    }
    
    sum += ssum;
   }
  }

  return sum * norm;
}

/*
 * Access grid point at given index.
 *
 * grid = grid to be accessed (cgrid *; input).
 * i    = index along x (INT; input).
 * j    = index along y (INT; input).
 * k    = index along z (INT; input).
 *
 * Returns grid value at index (i, j, k).
 *
 * NOTE: This is *very* slow on cuda as it transfers each element individually.
 *
 */

EXPORT inline REAL complex cgrid_value_at_index(cgrid *grid, INT i, INT j, INT k) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz)
    return grid->value_outside(grid, i, j, k);

#ifdef USE_CUDA
  if(cuda_find_block(grid->value)) {
    REAL complex tmp;
    cuda_get_element(grid->value, (size_t) ((i * grid->ny + j) * grid->nz + k), sizeof(REAL complex), &tmp);
    return tmp;
  } else
#endif
  return grid->value[(i * grid->ny + j) * grid->nz + k];
}

/*
 * Set value to a grid point at given index.
 *
 * grid  = grid to be accessed (cgrid *; output).
 * i     = index along x (INT; input).
 * j     = index along y (INT; input).
 * k     = index along z (INT; input).
 * value = value to be set at (i, j, k) (REAL complex; input).
 *
 * No return value.
 *
 * NOTE: This is *very* slow on cuda as it transfers each element individually.
 *
 */

EXPORT inline void cgrid_value_to_index(cgrid *grid, INT i, INT j, INT k, REAL complex value) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz) return;

#ifdef USE_CUDA
  if(cuda_find_block(grid->value))
    cuda_set_element(grid->value, (size_t) ((i * grid->ny + j) * grid->nz + k), sizeof(REAL complex), &value);
  else
#endif
   grid->value[(i * grid->ny + j) * grid->nz + k] = value;
}

/*
 * Access grid point at given (x,y,z) point using linear interpolation.
 *
 * grid = grid to be accessed (cgrid *; input).
 * x    = x value (REAL; input).
 * y    = y value (REAL; input).
 * z    = z value (REAL; input).
 *
 * Returns grid value at (x,y,z).
 *
 */

EXPORT inline REAL complex cgrid_value(cgrid *grid, REAL x, REAL y, REAL z) {

  REAL complex f000, f100, f010, f001, f110, f101, f011, f111;
  REAL omx, omy, omz, step = grid->step;
  INT i, j, k;
  
  /* i to index and 0 <= x < 1 */
  x = (x + grid->x0) / step;
  i = (INT) x;
  if (x < 0) i--;
  x -= (REAL) i;
  i += grid->nx / 2;
  
  /* j to index and 0 <= y < 1 */
  y = (y + grid->y0) / step;
  j = (INT) y;
  if (y < 0) j--;
  y -= (REAL) j;
  j += grid->ny / 2;
  
  /* k to index and 0 <= z < 1 */
  z = (z + grid->z0) / step;
  k = (INT) z;
  if (z < 0) k--;
  z -= (REAL) k;
  k += grid->nz / 2;
  
  /* linear extrapolation 
   *
   * f(x,y) = (1-x) (1-y) (1-z) f(0,0,0) + x (1-y) (1-z) f(1,0,0) + (1-x) y (1-z) f(0,1,0) + (1-x) (1-y) z f(0,0,1) 
   *          + x     y   (1-z) f(1,1,0) + x (1-y)   z   f(1,0,1) + (1-x) y   z   f(0,1,1) +   x     y   z f(1,1,1)
   */ 
  f000 = cgrid_value_at_index(grid, i, j, k);
  f100 = cgrid_value_at_index(grid, i + 1, j, k);
  f010 = cgrid_value_at_index(grid, i, j + 1, k);
  f001 = cgrid_value_at_index(grid, i, j, k + 1);
  f110 = cgrid_value_at_index(grid, i + 1, j + 1, k);
  f101 = cgrid_value_at_index(grid, i + 1, j, k + 1);
  f011 = cgrid_value_at_index(grid, i, j + 1, k + 1);
  f111 = cgrid_value_at_index(grid, i + 1, j + 1, k + 1);
  
  omx = 1.0 - x;
  omy = 1.0 - y;
  omz = 1.0 - z;

  return omx * (omy * (omz * f000 + z * f001) + y * (omz * f010 + z * f011))
    + x * (omy * (omz * f100 + z * f101) + y * (omz * f110 + z * f111));
}

/*
 * Extrapolate between two grids of different sizes.
 *
 * dest = Destination grid (cgrid *; output).
 * src  = Source grid (cgrid *; input).
 *
 */

EXPORT void cgrid_extrapolate(cgrid *dest, cgrid *src) {

  INT i, j, k, nx = dest->nx, ny = dest->ny, nz = dest->nz;
  REAL x0 = dest->x0, y0 = dest->y0, z0 = dest->z0;
  REAL step = dest->step, x, y, z;

  if(dest == src) {
    fprintf(stderr, "libgrid: Source and destination grids cannot be the same in cgrid_extrapolate().\n");
    abort();
  }
#ifdef USE_CUDA
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dest->value, 0);
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,step,dest,src,x0,y0,z0) private(i,j,k,x,y,z) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx / 2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny / 2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz / 2)) * step - z0;
	dest->value[i * ny * nz + j * nz + k] = cgrid_value(src, x, y, z);
      }
    }
  }
}

/*
 * Subroutine for rotating grid around z axis. See below.
 *
 */

static REAL complex cgrid_value_rotate_z(void *arg, REAL x, REAL y, REAL z) {

  /* Unpack the values in arg */ 
  cgrid *grid = ((rotation *) arg)->cgrid;
  REAL sth = ((rotation *) arg)->sinth, cth = ((rotation *) arg)->costh, xp, yp;

  xp = -y * sth + x * cth;
  yp =  y * cth + x * sth;

  return cgrid_value(grid, xp, yp, z);
}

/*
 * Rotate a grid by a given angle around the z-axis.
 *
 * in  = Input grid (cgrid *; input).
 * out = Rotated grid (cgrid *; output).
 * th  = Rotation angle theta in radians (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_rotate_z(cgrid *out, cgrid *in, REAL th) {

  rotation *r;

  if(in == out) {
    fprintf(stderr,"libgrid: in and out grids in cgrid_rotate_z must be different\n");
    abort();
  }
  if(!(r = malloc(sizeof(rotation)))) {
    fprintf(stderr, "libgrid: cannot allocate rotation structure.\n");
    abort();
  }
  r->cgrid = in;
  r->sinth = SIN(-th);  // same direction of rotation as -wLz
  r->costh = COS(th);
  
  cgrid_map(out, cgrid_value_rotate_z, (void *) r);
  free(r);
}

/*
 * Clear real part of complex grid.
 *
 * grid = grid for the operation (cgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void cgrid_zero_re(cgrid *grid) {

  INT i;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_zero_re(grid)) return;
#endif

#pragma omp parallel for firstprivate(grid) private(i) default(none) schedule(runtime)
  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++)
    grid->value[i] = I * CIMAG(grid->value[i]);
}

/*
 * Clear imaginary part of complex grid.
 *
 * grid = grid for the operation (cgrid *; input/output).
 *
 */

EXPORT void cgrid_zero_im(cgrid *grid) {

  INT i;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_zero_im(grid)) return;
#endif

#pragma omp parallel for firstprivate(grid) private(i) default(none) schedule(runtime)
  for(i = 0; i < grid->nx * grid->ny * grid->nz; i++)
    grid->value[i] = CREAL(grid->value[i]);
}

/*
 * Extract complex phase factors from a given grid.
 *
 * dst = Dest grid (rgrid *; output).
 * src = Source grid (cgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_phase(rgrid *dst, cgrid *src) {

  INT i;

#ifdef USE_CUDA
  // Not implemented yet (TODO)
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dst->value, 0);
#endif
  if(dst->nx != src->nx || dst->ny != src->ny || dst->nz != src->nz) {
    fprintf(stderr, "libgrid: incompatible dimensions in cgrid_phase().\n");
    exit(1);
  }

#pragma omp parallel for firstprivate(src,dst) private(i) default(none) schedule(runtime)
  for (i = 0; i < dst->nx * dst->ny * dst->nz; i++) {
    if(CABS(src->value[i]) < GRID_EPS) dst->value[i] = 0.0;
    else dst->value[i] = CREAL(-I * CLOG(src->value[i] / CABS(src->value[i])));
    if(dst->value[i] < 0.0) dst->value[i] = 2.0 * M_PI + dst->value[i];
  }
}

/*
 * Add random noise to grid.
 *
 * grid  = Grid where the noise will be added (cgrid *; input/output).
 * scale = Scaling for random numbers [-scale,+scale[ (REAL; input).
 *
 */

EXPORT void cgrid_random(cgrid *grid, REAL scale) {

  static char been_here = 0;
  INT i;

  if(!been_here) {
    srand48(time(0));
    been_here = 1;
  }

#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  // drand48 is not thread safe.
  for (i = 0; i < grid->nx * grid->ny * grid->nz; i++)
    grid->value[i] += scale * (2.0 * (((REAL) drand48()) - 0.5) + 2.0 * (((REAL) drand48()) - 0.5) * I);
}

/*
 * Solve Poisson equation: Laplace f = u subject to periodic boundary condition.
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
    exit(1);
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_poisson(grid)) return;  
#endif
  cgrid_fftw(grid);
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
  cgrid_fftw_inv(grid);
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

EXPORT void cgrid_div(cgrid *div, cgrid *fx, cgrid *fy, cgrid *fz) {

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
 * rotx = x component of rot (cgrid *; output).
 * roty = y component of rot (cgrid *; output).
 * rotz = z component of rot (cgrid *; output).
 * fx   = x component of the field (cgrid *; input).
 * fy   = y component of the field (cgrid *; input).
 * fz   = z component of the field (cgrid *; input).
 *
 * TODO: CUDA implementation missing.
 *
 */

EXPORT void cgrid_rot(cgrid *rotx, cgrid *roty, cgrid *rotz, cgrid *fx, cgrid *fy, cgrid *fz) {

  INT i, j, k, ij, ijnz, ny = rotx->ny, nz = rotx->nz, nxy = rotx->nx * rotx->ny;
  REAL inv_delta = 1.0 / (2.0 * rotx->step);
  REAL complex *lvaluex = rotx->value, *lvaluey = roty->value, *lvaluez = rotz->value;
  
  if(rotx == fx || rotx == fy || rotx == fz || roty == fx || roty == fy || roty == fz || rotz == fx || rotz == fy || rotz == fz) {
    fprintf(stderr, "libgrid: Source and destination grids must be different in cgrid_rot().\n");
    abort();
  }

#ifdef USE_CUDA
  cuda_remove_block(lvaluex, 0);
  cuda_remove_block(lvaluey, 0);
  cuda_remove_block(lvaluez, 0);
  cuda_remove_block(fx->value, 1);
  cuda_remove_block(fy->value, 1);
  cuda_remove_block(fz->value, 1);
#endif
#pragma omp parallel for firstprivate(ny,nz,nxy,lvaluex,lvaluey,lvaluez,inv_delta,fx,fy,fz) private(ij,ijnz,i,j,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* x: (d/dy) fz - (d/dz) fy */
      lvaluex[ijnz + k] = inv_delta * ((cgrid_value_at_index(fz, i, j + 1, k) - cgrid_value_at_index(fz, i, j - 1, k))
				      - (cgrid_value_at_index(fy, i, j, k + 1) - cgrid_value_at_index(fy, i, j, k - 1)));
      /* y: (d/dz) fx - (d/dx) fz */
      lvaluey[ijnz + k] = inv_delta * ((cgrid_value_at_index(fx, i, j, k + 1) - cgrid_value_at_index(fx, i, j, k - 1))
				      - (cgrid_value_at_index(fz, i + 1, j, k) - cgrid_value_at_index(fz, i - 1, j, k)));
      /* z: (d/dx) fy - (d/dy) fx */
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

EXPORT void cgrid_abs_rot(rgrid *rot, cgrid *fx, cgrid *fy, cgrid *fz) {

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

/*
 * Zero a given index range of a complex grid over [lx, hx[ X [ly, hy[ X [lz, hz[ .
 *
 * grid     = Grid to be operated on (cgrid *; input/output).
 * lx       = Lower limit for x index (INT; input).
 * hx       = Upper limit for x index (INT; input).
 * ly       = Lower limit for y index (INT; input).
 * hy       = Upper limit for y index (INT; input).
 * lz       = Lower limit for z index (INT; input).
 * hz       = Upper limit for z index (INT; input).
 *
 * No return value.
 *
 */

EXPORT void cgrid_zero_index(cgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nynz = ny * nz;
  REAL complex *value = grid->value;

  if(hx > nx || lx < 0 || hy > ny || ly < 0 || hz > nz || lz < 0) return;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_zero_index(grid, lx, hx, ly, hy, lz, hz)) return;
#endif

#pragma omp parallel for firstprivate(lx,hx,nx,ly,hy,ny,lz,hz,nz,value,nynz) private(i,j,k) default(none) schedule(runtime)
  for(i = lx; i < hx; i++)
    for(j = ly; j < hy; j++)
      for(k = lz; k < hz; k++)
        value[i * nynz + j * nz + k] = 0.0;
}

/*
 * de-alias grid in Fourier space by the 2/3 rule: zero high wavenumber components between:
 * [n/2 - n/3, n/2 + n/3] in each direction.
 *
 * grid     = Grid in Fourier space (cgrid *; input/output).
 *
 * Note: The combination component in the reciprocal space is at n/2 (f = \pm 1/(2 delta)).
 *
 * No return value.
 *
 */

EXPORT void cgrid_dealias23(cgrid *grid) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  cgrid_zero_index(grid, nx / 2 - nx / 3, nx / 2 + nx / 3 + 1, ny / 2 - ny / 3, ny / 2 + ny / 3 + 1, 
                         nz / 2 - nz / 3, nz / 2 + nz / 3 + 1);
}

/*
 * Raise grid to integer power (fast). |grid|^n
 *
 * dst      = Destination grid (cgrid *; output).
 * src      = Source grid (cgrid *; input).
 * exponent = Exponent to be used (INT; input). This value can be negative.
 *
 * No return value.
 *
 * TODO: CUDA could use ipow equivalent. Uses pow() now.
 *
 */

EXPORT void cgrid_ipower(cgrid *dst, cgrid *src, INT exponent) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz;
  REAL complex *avalue = src->value;
  REAL complex *bvalue = dst->value;
  
  if(exponent == 1) {
    cgrid_copy(dst, src);
    return;
  }

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_abs_power(dst, src, (REAL) exponent)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = ipow(CABS(avalue[ijnz + k]), exponent);  /* ipow() already compiled with the right precision */
  }
}
