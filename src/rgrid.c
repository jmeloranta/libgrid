/*
 * Routines for real grids.
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
#include "rprivate.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

extern char grid_analyze_method;

/*
 * Allocate new complex grid.
 *
 *
 * nx                 = number of points on the grid along x (INT).
 * ny                 = number of points on the grid along y (INT).
 * nz                 = number of points on the grid along z (INT).
 * step               = spatial step length on the grid (REAL).
 * value_outside      = condition for accessing boundary points:
 *                      RGRID_DIRICHLET_BOUNDARY: Dirichlet boundary.
 *                      or RGRID_NEUMANN_BOUNDARY: Neumann boundary.
 *                      or RGRID_PERIODIC_BOUNDARY: Periodic boundary.
 *                      or user supplied function with pointer to grid and
 *                         grid index as parameters to provide boundary access.
 * outside_params_ptr = pointer for passing parameters for the given boundary
 *                      access function. Use 0 to with the predefined boundary
 *                      functions (void *).
 * id                 = String ID describing the grid (char *; input).
 *
 * Return value: pointer to the allocated grid (rgrid *). Returns NULL on
 * error.
 *
 * Note: We keep the grid in padded form, which can be directly used for in-place FFT.
 *
 */

EXPORT rgrid *rgrid_alloc(INT nx, INT ny, INT nz, REAL step, REAL (*value_outside)(rgrid *grid, INT i, INT j, INT k), void *outside_params_ptr, char *id) {

  rgrid *grid;
  INT nz2, i;
  size_t len; 
 
  if(!(grid = (rgrid *) malloc(sizeof(rgrid)))) {
    fprintf(stderr, "libgrid: Error in rgrid_alloc(). Could not allocate memory for grid structure.\n");
   return 0;
  }
  
  grid->nx = nx;
  grid->ny = ny;
  grid->nz = nz;
  grid->nz2 = nz2 = 2 * (nz / 2 + 1);
  grid->grid_len = len = ((size_t) (nx * ny * nz2)) * sizeof(REAL);

#if defined(SINGLE_PREC)
  if (!(grid->value = (REAL *) fftwf_malloc(len))) {  /* Extra space needed to hold the FFT data */
#elif defined(DOUBLE_PREC)
  if (!(grid->value = (REAL *) fftw_malloc(len))) {  /* Extra space needed to hold the FFT data */
#elif defined(QUAD_PREC)
  if (!(grid->value = (REAL *) fftwl_malloc(len))) {  /* Extra space needed to hold the FFT data */
#endif
    fprintf(stderr, "libgrid: Error in rgrid_alloc(). Could not allocate memory for rgrid->value.\n");
    abort();
  }

  grid->step = step;
  /* Set the origin of coordinates to its default value */
  grid->x0 = 0.0;
  grid->y0 = 0.0;
  grid->z0 = 0.0;
  /* Set the origin of momentum (i.e. frame of reference velocity) to its default value */
  grid->kx0 = 0.0;
  grid->ky0 = 0.0;
  grid->kz0 = 0.0;

  /* value outside not set */
  
  if (value_outside)
    grid->value_outside = value_outside;
  else
    grid->value_outside = rgrid_value_outside_dirichlet;

  if (outside_params_ptr) grid->outside_params_ptr = outside_params_ptr;
  else {
    grid->default_outside_params = 0.0;
    grid->outside_params_ptr = &grid->default_outside_params;
  }
  
  grid->plan = grid->iplan = NULL;  // No need to allocate these yet
#ifdef USE_CUDA  
  if(cuda_status()) {
    rgrid_cufft_alloc_r2c(grid);
    rgrid_cufft_alloc_c2r(grid);
  }
#endif

  if (grid->value_outside == RGRID_NEUMANN_BOUNDARY)
    grid->fft_norm = 1.0 / (2.0 * ((REAL) grid->nx) * 2.0 * ((REAL) grid->ny) * 2.0 * ((REAL) grid->nz));
  else
    grid->fft_norm = 1.0 / ((REAL) (grid->nx * grid->ny * grid->nz));

  // Account for the correct dimensionality of the grid
  grid->fft_norm2 = grid->fft_norm;
  if(grid->nx > 1) grid->fft_norm2 *= grid->step;
  if(grid->ny > 1) grid->fft_norm2 *= grid->step;
  if(grid->nz > 1) grid->fft_norm2 *= grid->step;

  strncpy(grid->id, id, 32);
  grid->id[31] = 0;

#ifdef USE_CUDA
  /* rgrid_cuda_init(sizeof(REAL) * (size_t) nx); */ // reduction along x (was len)
  rgrid_cuda_init(sizeof(REAL) 
     * ((((size_t) nx) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
     * ((((size_t) ny) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
     * ((((size_t) nz) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK));  // reduction along blocks
#endif

  for(i = 0; i < nx * ny * nz2; i++)
    grid->value[i] = 0.0;

  grid->flag = 0;
#ifdef USE_CUDA
  grid->host_lock = 0;
#endif

  if(grid_analyze_method == -1) {
#ifdef USE_CUDA
    if(cuda_ngpus() > 1) {
      fprintf(stderr, "libgrid: More than one GPU requested - using FFT for grid analysis.\n");
      grid_analyze_method = 1; // FFT-based differentiation is required for multi-GPU
    } else
#endif
      grid_analyze_method = 0; // Default to using finite difference for analysis
  }

#ifdef USE_CUDA
  if(cuda_ngpus() > 1 && grid_analyze_method == 0) {
    fprintf(stderr, "libgrid: Finite difference cannot be used with more than one GPU.\n");
    abort();
  }
#endif

  return grid;
}

/*
 * "Clone" a real grid with the parameters identical to the given grid (except new grid->value is allocated).
 *
 * grid = Grid to be cloned (rgrid *; input).
 * id   = ID string describing the grid (char *; input);
 *
 * Returns pointer to the new grid (rgrid *).
 *
 */

EXPORT rgrid *rgrid_clone(rgrid *grid, char *id) {

  rgrid *ngrid;

  if(!(ngrid = (rgrid *) malloc(sizeof(rgrid)))) {
    fprintf(stderr, "libgrid: Out of memory in rgrid_clone().\n");
    abort();
  }
  bcopy((void *) grid, (void *) ngrid, sizeof(rgrid));
  strcpy(ngrid->id, id);

#if defined(SINGLE_PREC)
  if (!(ngrid->value = (REAL *) fftwf_malloc(ngrid->grid_len))) {  /* Extra space needed to hold the FFT data */
#elif defined(DOUBLE_PREC)
  if (!(ngrid->value = (REAL *) fftw_malloc(ngrid->grid_len))) {  /* Extra space needed to hold the FFT data */
#elif defined(QUAD_PREC)
  if (!(ngrid->value = (REAL *) fftwl_malloc(ngrid->grid_len))) {  /* Extra space needed to hold the FFT data */
#endif
    fprintf(stderr, "libgrid: Error in rgrid_clone(). Could not allocate memory for ngrid->value.\n");
    free(ngrid);
    return NULL;
  }
#ifdef USE_CUDA
  if(cuda_status()) {
    rgrid_cufft_alloc_r2c(ngrid);
    rgrid_cufft_alloc_c2r(ngrid);
  }
#endif
  ngrid->plan = ngrid->iplan = NULL;
  ngrid->flag = 0;

  return ngrid;
}

/*
 * Claim grid (simple locking system for the workspace model).
 *
 * grid = Grid to be claimed (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_claim(rgrid *grid) {

  if(!grid) {
    fprintf(stderr, "libgrid: Attempting to claim a non-existent grid.\n");
    abort();
  }
  if(grid->flag) {
    fprintf(stderr, "libgrid: Attempting to claim grid twice.\n");
    abort();
  }
  grid->flag = 1;
}

/*
 * Release grid (simple locking system for the workspace model).
 *
 * grid = Grid to be claimed (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_release(rgrid *grid) {

  if(!grid->flag) {
    fprintf(stderr, "libgrid: Attempting to release grid twice.\n");
    abort();
  }
  grid->flag = 0;
}

/*
 * Set the grid origin.
 *
 * grid   = Grid for which the origin is to be defined (rgrid *; input/output).
 * x0     = X coordinate for the origin (REAL; input).
 * y0     = Y coordinate for the origin (REAL; input).
 * z0     = Z coordinate for the origin (REAL; input).
 *
 * The grid coordinates will be evaluated as:
 * x(i)  = (i - nx / 2) * step - x0
 * y(j)  = (j - ny / 2) * step - y0
 * z(k)  = (k - nz / 2) * step - z0
 *
 * No return value.
 *
 */

EXPORT void rgrid_set_origin(rgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 = x0;
  grid->y0 = y0;
  grid->z0 = z0;
}

/* 
 * Shift the grid origin.
 * 
 * grid  = Grid for which the origin is to be shifted (rgrid *; input/output).
 * x0    = Shift in X coordinate (REAL; input).
 * y0    = Shift in Y coordinate (REAL; input).
 * z0    = Shift in Z coordinate (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_shift_origin(rgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 += x0;
  grid->y0 += y0;
  grid->z0 += z0;
}

/*
 * Set the grid origin in momentum space (or the velocity of the frame of reference).
 * 
 * grid    = Grid for which the momentum origin is to be defined (rgrid *; input/output).
 * kx0     = Momentum origin along the X axis (REAL; input).
 * ky0     = Momentum origin along the Y axis (REAL; input).
 * kz0     = Momentum origin along the Z axis (REAL; input).
 *
 * kx0, ky0 and kz0 can be any real numbers but keep in mind that the grid
 * will only contain the component k = 0 if they are multiples of:
 *
 *  kx0min = 2 * M_PI / (NX * STEP) 
 *  ky0min = 2 * M_PI / (NY * STEP) 
 *  kz0min = 2 * M_PI / (NZ * STEP)
 *
 * No return value.
 *
 */

EXPORT void rgrid_set_momentum(rgrid *grid, REAL kx0, REAL ky0, REAL kz0) {

  grid->kx0 = kx0;
  grid->ky0 = ky0;
  grid->kz0 = kz0;
}

/*
 * Free grid.
 *
 * grid = pointer to  grid to be freed (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_free(rgrid *grid) {

  if (grid) {
#ifdef USE_CUDA
    cuda_remove_block(grid->value, 0);
    if(grid->cufft_handle_r2c != -1) cufftDestroy(grid->cufft_handle_r2c);
    if(grid->cufft_handle_c2r != -1) cufftDestroy(grid->cufft_handle_c2r);
#endif
#if defined(SINGLE_PREC)
    if (grid->value) fftwf_free(grid->value);
#elif defined(DOUBLE_PREC)
    if (grid->value) fftw_free(grid->value);
#elif defined(QUAD_PREC)
    if (grid->value) fftwl_free(grid->value);
#endif
    rgrid_fftw_free(grid);
    free(grid);
  }
}

/* 
 * Write grid on disk in binary format.
 *
 * grid = grid to be written (rgrid *; input).
 * out  = file handle for the file (FILE *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_write(rgrid *grid, FILE *out) {

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif
  fwrite(&grid->nx, sizeof(INT), 1, out);
  fwrite(&grid->ny, sizeof(INT), 1, out);
  fwrite(&grid->nz, sizeof(INT), 1, out);
  fwrite(&grid->step, sizeof(REAL), 1, out);
  fwrite(grid->value, sizeof(REAL), (size_t) (grid->nx * grid->ny * grid->nz2), out);
}

/* 
 * Read grid from disk in binary format.
 *
 * grid = grid to be read (rgrid *; output). If NULL, a grid with the correct dimensions will be allocated.
 *        Note that the boundary condition will assigned to PERIODIC by default.
 * in   = file handle for reading the file (FILE *; input).
 *
 * Returns pointer to the grid (NULL on error).
 *
 */

EXPORT rgrid *rgrid_read(rgrid *grid, FILE *in) {

  INT nx, ny, nz, nzz;
  REAL step;
  
#ifdef USE_CUDA
  if(grid) cuda_remove_block(grid->value, 0);  // grid will be overwritten below
#endif
  fread(&nx, sizeof(INT), 1, in);
  fread(&ny, sizeof(INT), 1, in);
  fread(&nz, sizeof(INT), 1, in);
  nzz = 2 * (nz / 2 + 1);
  fread(&step, sizeof(REAL), 1, in);
  
  if (!grid) {
    if(!(grid = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "read_grid"))) {
      fprintf(stderr, "libgrid: Failed to allocate grid in rgrid_read().\n");
      return NULL;
    }
  }

  if (nx != grid->nx || ny != grid->ny || nz != grid->nz || step != grid->step) {
    rgrid *tmp;

    fprintf(stderr, "libgrid: Grid in file has different size than grid in memory.\n");
    fprintf(stderr, "libgrid: Interpolating between grids.\n");
    if(!(tmp = rgrid_alloc(nx, ny, nz, step, grid->value_outside, NULL, "rgrid_read_temp"))) {
      fprintf(stderr, "libgrid: Error allocating grid in rgrid_read().\n");
      abort();
    }
    fread(tmp->value, sizeof(REAL), (size_t) (nx * ny * nzz), in);
    rgrid_extrapolate(grid, tmp);
    rgrid_free(tmp);
    return grid;
  }
  
  fread(grid->value, sizeof(REAL), (size_t) (nx * ny * nzz), in);
  return grid;
}

/* 
 * Read grid from disk in binary format. This is compatible with old libgrid binary grid format.
 * Due to in place FFT being used, the new grids have holes in them...
 *
 * grid = grid to be read (rgrid *; output). If NULL, a grid with the correct dimensions will be allocated.
 *        Note that the boundary condition will assigned to PERIODIC by default.
 * in   = file handle for reading the file (FILE *; input).
 *
 * Returns pointer to the grid (NULL on error).
 *
 */

EXPORT rgrid *rgrid_read_compat(rgrid *grid, FILE *in) {

  INT nx, ny, nz, i, j, k;
  REAL step, val;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);  // grid will be overwritten below
#endif
  fread(&nx, sizeof(INT), 1, in);
  fread(&ny, sizeof(INT), 1, in);
  fread(&nz, sizeof(INT), 1, in);
  fread(&step, sizeof(REAL), 1, in);
  
  if (!grid) {
    if(!(grid = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "read_grid"))) {
      fprintf(stderr, "libgrid: Failed to allocate grid in rgrid_read_compat().\n");
      return NULL;
    }
  }

  if (nx != grid->nx || ny != grid->ny || nz != grid->nz || step != grid->step) {
    fprintf(stderr, "libgrid: Interpolation not supported for compatibility mode.\n");
    abort();
  }
  
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
        fread(&val, sizeof(REAL), 1, in);
        rgrid_value_to_index(grid, i, j, k, val);
      }
  return grid;
}

/*
 * Read in real grid from a binary file (.grd).
 *
 * grid = place to store the read density (output, rgrid *).
 * file = filename for the file (char *). Note: the .grd extension must NOT be given (input, char *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_read_grid(rgrid *grid, char *file) {

  FILE *fp;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libgrid: Can't open real grid file %s.\n", file);
    abort();
  }
  rgrid_read(grid, fp);
  fclose(fp);
  fprintf(stderr, "libgrid: Real grid read from %s.\n", file);
}

/*
 * Read in real grid from a binary file (.grd). Compatibility with old libgrid binary grid files.
 *
 * grid = place to store the read density (output, rgrid *).
 * file = filename for the file (char *). Note: the .grd extension must NOT be given (input, char *).
 *
 * No return value.
 *
 */

EXPORT void rgrid_read_grid_compat(rgrid *grid, char *file) {

  FILE *fp;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libgrid: Can't open real grid file %s.\n", file);
    abort();
  }
  rgrid_read_compat(grid, fp);
  fclose(fp);
  fprintf(stderr, "libgrid: Real grid read from %s.\n", file);
}

/*
 * Write real grid to disk including cuts along x, y, and z axes.
 *
 * basename = Base filename where suffixes (.x, .y, .z, and .grd) are added (char *; input).
 * grid     = Grid to be written to disk (rgrid *; input).
 * 
 * No return value.
 *
 * See also rgrid_write().
 *
 */

EXPORT void rgrid_write_grid(char *base, rgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x, y, z, step = grid->step;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

  /* Write binary grid */
  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  rgrid_write(grid, fp);
  fclose(fp);

  /* Write cut along x-axis */
  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  j = ny / 2;
  k = nz / 2;
  for(i = 0; i < nx; i++) { 
    x = ((REAL) (i - nx / 2)) * step;
    fprintf(fp, FMT_R " " FMT_R "\n", x, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  /* Write cut along y-axis */
  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  i = nx / 2;
  k = nz / 2;
  for(j = 0; j < ny; j++) {
    y = ((REAL) (j - ny / 2)) * step;
    fprintf(fp, FMT_R " " FMT_R "\n", y, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  /* Write cut along z-axis */
  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  i = nx / 2;
  j = ny / 2;
  for(k = 0; k < nz; k++) {
    z = ((REAL) (k - nz / 2)) * step;
    fprintf(fp, FMT_R " " FMT_R "\n", z, rgrid_value_at_index(grid, i, j, k));
  }
  fclose(fp);
}

/*
 * Copy grid from one grid to another.
 *
 * dst = destination grid (rgrid *; output).
 * src = source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_copy(rgrid *dst, rgrid *src) {

  INT i, nx = src->nx, nyz = src->ny * src->nz2;
  size_t bytes = ((size_t) nyz) * sizeof(REAL);
  REAL *svalue = src->value;
  REAL *dvalue = dst->value;
  
  if(src->nx != dst->nx || src->ny != dst->ny || src->nz != dst->nz) {
    fprintf(stderr, "libgrid: Different grid dimensions in rgrid_copy.\n");
    abort();
  }

  dst->step = src->step;  
  dst->x0 = src->x0;
  dst->y0 = src->y0;
  dst->z0 = src->z0;
  dst->kx0 = src->kx0;
  dst->ky0 = src->ky0;
  dst->kz0 = src->kz0;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_copy(dst, src)) return;
#endif

#pragma omp parallel for firstprivate(nx,nyz,bytes,svalue,dvalue) private(i) default(none) schedule(runtime)
  for(i = 0; i < nx; i++)
    bcopy(&svalue[i * nyz], &dvalue[i * nyz], bytes);
}

/*
 * Shift grid by given amount spatially.
 *
 * shifted = destination grid for the operation (rgrid *; output).
 * grid    = source grid for the operation (rgrid *; input).
 * x       = shift spatially by this amount in x (REAL; input).
 * y       = shift spatially by this amount in y (REAL; input).
 * z       = shift spatially by this amount in z (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_shift(rgrid *shifted, rgrid *grid, REAL x, REAL y, REAL z) {

  sShiftParametersr params;

  if(grid == shifted) {
    fprintf(stderr, "libgrid: Source and destination must be different in rgrid_shift().\n");
    abort();
  }
  /* shift by (x,y,z) i.e. current grid center to (x,y,z) */
  params.x = x;
  params.y = y;
  params.z = z; 
  params.grid = grid;
  rgrid_map(shifted, shift_rgrid, &params);
}

/* 
 * Zero grid.
 *
 * grid = grid to be zeroed (rgrid *; input/output).
 *
 * No return value.
 * 
 */

EXPORT void rgrid_zero(rgrid *grid) { 

  rgrid_constant(grid, 0.0); 
}

/* 
 * Set grid to a constant value.
 *
 * grid = grid to be set (rgrid *; input/output).
 * c    = constant value (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_constant(rgrid *grid, REAL c) {

   INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
   REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_constant(grid, c)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
   for(ij = 0; ij < nxy; ij++) {
     ijnz = ij * nzz;
     for(k = 0; k < nz; k++)
       value[ijnz + k] = c;
     if(nz != nzz) value[ijnz + nz] = value[ijnz + nz + 1] = 0.0;   // TODO: Needed?
   }
}

/*
 * Multiply a given grid by a function.
 *
 * grid = destination grid for the operation (rgrid *; output).
 * func = function providing the mapping (REAL (*)(void *, REAL, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data, the next is grid value at the point,
 *        and x, y, z are the coordinates (REAL) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_product_func(rgrid *grid, REAL (*func)(void *arg, REAL val, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, nx2 = nx / 2, 
      ny2 = ny / 2, nz2 = nz / 2;
  REAL x, y, z, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nzz,nx2,ny2,nz2,nxy,step,func,value,x0,y0,z0) private(i,j,ij,ijnz,k,x,y,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      value[ijnz + k] *= func(farg, value[ijnz + k], x, y, z);
    }
  }
}

/*
 * Map a given function onto grid.
 *
 * grid = destination grid for the operation (rgrid *; output).
 * func = function providing the mapping (REAL (*)(void *, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data
 *        and x,y,z are the coordinates (REAL) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_map(rgrid *grid, REAL (*func)(void *arg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, nx2 = nx / 2, ny2 = ny / 2, 
      nz2 = nz / 2;
  REAL x, y, z, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nzz,nx2,ny2,nz2,nxy,step,func,value,x0,y0,z0) private(i,j,ij,ijnz,k,x,y,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      value[ijnz + k] = func(farg, x, y, z);
    }
  }
}

/*
 * Map a given function onto  grid with linear "smoothing".
 * This can be used to weight the values at grid points to produce more
 * accurate integration over the grid.
 *
 * grid = destination grid for the operation (rgrid *; output).
 * func = function providing the mapping (REAL (*)(void *, REAL, REAL, REAL); input).
 *        The first argument (void *) is for external user specified data
 *        and (x, y, z) is the point (REALs) where the function is evaluated.
 * farg = pointer to user specified data (void *; input).
 * ns   = number of intermediate points to be used in smoothing (INT; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_smooth_map(rgrid *grid, REAL (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, INT ns) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, nx2 = nx / 2, ny2 = ny / 2, 
      nz2 = nz / 2;
  REAL xc, yc, zc, step = grid->step;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nzz,nx2,ny2,nz2,nxy,ns,step,func,value,x0,y0,z0) private(i,j,k,ijnz,xc,yc,zc) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    xc = ((REAL) (i - nx2)) * step - x0;
    yc = ((REAL) (j - ny2)) * step - y0;
    for(k = 0; k < nz; k++) {
      zc = ((REAL) (k - nz2)) * step - z0;
      value[ijnz + k] = linearly_weighted_integralr(func, farg, xc, yc, zc, step, ns);
    }
  }
}

/*
 * Map a given function onto grid with linear "smoothing".
 * This can be used to weight the values at grid points to produce more
 * accurate integration over the grid. Limits for intermediate steps and
 * tolerance can be given.
 *
 * grid   = destination grid for the operation (rgrid *; output).
 * func   = function providing the mapping (REAL (*)(void *, REAL, REAL, REAL); input).
 *          The first argument (void *) is for external user specified data
 *          and x, y, z are the coordinates (REAL) where the function is evaluated.
 * farg   = pointer to user specified data (void *; input).
 * min_ns = minimum number of intermediate points to be used in smoothing (INT; input).
 * max_ns = maximum number of intermediate points to be used in smoothing (INT; input).
 * tol    = tolerance for the converge of integral over the function (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_adaptive_map(rgrid *grid, REAL (*func)(void *arg, REAL x, REAL y, REAL z), void *farg, INT min_ns, INT max_ns, REAL tol) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, ns, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL xc, yc, zc, step = grid->step;
  REAL tol2 = tol * tol;
  REAL sum, sump;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0;
  REAL *value = grid->value, tmp;
  
  if (min_ns < 1) min_ns = 1;
  if (max_ns < min_ns) max_ns = min_ns;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif

#pragma omp parallel for firstprivate(stderr,farg,nx,ny,nz,nzz,nx2,ny2,nz2,nxy,min_ns,max_ns,step,func,value,tol2,x0,y0,z0) private(i,j,k,ijnz,ns,xc,yc,zc,sum,sump,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    xc = ((REAL) (i - nx2)) * step - x0;
    yc = ((REAL) (j - ny2)) * step - y0;
    for(k = 0; k < nz; k++) {
      zc = ((REAL) (k - nz2)) * step - z0;
      sum  = func(farg, xc, yc, zc); sump = 0.0;
      for(ns = min_ns; ns <= max_ns; ns *= 2) {
        sum  = linearly_weighted_integralr(func, farg, xc, yc, zc, step, ns);
        sump = linearly_weighted_integralr(func, farg, xc, yc, zc, step, ns + 1);
        tmp = sum - sump;
        if (tmp * tmp < tol2) break;
      }
#if 0
      if (ns >= max_ns)
        fprintf(stderr, "#");
      else if (ns > min_ns + 1)
        fprintf(stderr, "+");
      else
        fprintf(stderr, "-");
#endif      
      value[ijnz + k] = 0.5 * (sum + sump);
    }    
// fprintf(stderr, "\n");
  }
}

/*
 * Add two grids: gridc = grida + gridb
 *
 * gridc = destination grid (rgrid *; output).
 * grida = 1st of the grids to be added (rgrid *; input).
 * gridb = 2nd of the grids to be added (rgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_sum(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_sum(gridc, grida, gridb)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] + bvalue[ijnz + k];
  }
}

/* 
 * Subtract two grids: gridc = grida - gridb
 *
 * gridc = destination grid (rgrid *; output).
 * grida = 1st source grid (rgrid *; input).
 * gridb = 2nd source grid (rgrid *; input).
 *
 * No return value.
 *
 * Note: both source and destination may be the same.
 *
 */

EXPORT void rgrid_difference(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = grida->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_difference(gridc, grida, gridb)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] - bvalue[ijnz + k];
  }
}

/* 
 * Calculate product of two grids: gridc = grida * gridb
 *
 * gridc = destination grid (rgrid *; output).
 * grida = 1st source grid (rgrid *; input).
 * gridb = 2nd source grid (rgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_product(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_product(gridc, grida, gridb)) return;
#endif  
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] * bvalue[ijnz + k];
  }
}

/* 
 * Rise a grid to given power.
 *
 * gridb    = destination grid (rgrid *; output).
 * grida    = 1st source grid (rgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 * No return value.
 *
 * Notes: - Source and destination grids may be the same.
 *         - This routine uses pow() so that the exponent can be
 *           fractional but this is slow! Do not use this for integer
 *           exponents.
 *
 */

EXPORT void rgrid_power(rgrid *gridb, rgrid *grida, REAL exponent) {

  INT ij, k, ijnz, nxy = gridb->nx * gridb->ny, nz = gridb->nz, nzz = gridb->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_power(gridb, grida, exponent)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = POW(avalue[ijnz + k], exponent);
  }
}

/* 
 * Rise absolute value of a grid to given power.
 *
 * gridb    = destination grid (rgrid *; output).
 * grida    = 1st source grid (rgrid *; input).
 * exponent = exponent to be used (REAL; input).
 *
 * No return value.
 *
 * Notes: - Source and destination grids may be the same.
 *         - This routine uses pow() so that the exponent can be
 *           fractional but this is slow! Do not use this for integer
 *           exponents.
 *
 */

EXPORT void rgrid_abs_power(rgrid *gridb, rgrid *grida, REAL exponent) {

  INT ij, k, ijnz, nxy = gridb->nx * gridb->ny, nz = gridb->nz, nzz = gridb->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_abs_power(gridb, grida, exponent)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = POW(FABS(avalue[ijnz + k]), exponent);
  }
}

/*
 * Divide two grids: gridc = grida / gridb
 *
 * gridc = destination grid (rgrid *; output).
 * grida = 1st source grid (rgrid *; input).
 * gridb = 2nd source grid (rgrid *; input).
 *
 * No return value.
 *
 * Note: Source and destination grids may be the same.
 *
 */

EXPORT void rgrid_division(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_division(gridc, grida, gridb)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] / bvalue[ijnz + k];
  }
}

/*
 * "Safely" divide two grids: gridc = grida / (gridb + eps)
 *
 * gridc = destination grid (rgrid *; output).
 * grida = 1st source grid (rgrid *; input).
 * gridb = 2nd source grid (rgrid *; input).
 * eps   = Epsilon to add to the divisor (REAL; input).
 *
 * No return value.
 *
 * Note: Source and destination grids may be the same.
 *
 */

EXPORT void rgrid_division_eps(rgrid *gridc, rgrid *grida, rgrid *gridb, REAL eps) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_division_eps(gridc, grida, gridb, eps)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,eps) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] / (bvalue[ijnz + k] + eps);
  }
}

/*
 * Add a constant to a grid.
 *
 * grid = grid where the constant is added (rgrid *; input/output).
 * c    = constant to be added (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_add(rgrid *grid, REAL c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_add(grid, c)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] += c;
  }
}

/*
 * Multiply grid by a constant.
 *
 * grid = grid to be multiplied (rgrid *; input/output).
 * c    = multiplier (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_multiply(rgrid *grid, REAL c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_multiply(grid, c)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= c;
  }
}

/* 
 * Add and multiply: grid = (grid + ca) * cm.
 *
 * grid = grid to be operated (rgrid *; input/output).
 * ca   = constant to be added (REAL; input).
 * cm   = multiplier (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_add_and_multiply(rgrid *grid, REAL ca, REAL cm) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_add_and_multiply(grid, ca, cm)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,ca,cm) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] = (value[ijnz + k] + ca) * cm;
  }
}

/*
 * Multiply and add: grid = cm * grid + ca.
 *
 * grid = grid to be operated (rgrid *; input/output).
 * cm   = multiplier (REAL; input).
 * ca   = constant to be added (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_multiply_and_add(rgrid *grid, REAL cm, REAL ca) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_multiply_and_add(grid, cm, ca)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,ca,cm) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] = value[ijnz + k] * cm + ca;
  }
}

/* 
 * Add scaled grids (multiply/add): gridc = gridc + d * grida
 *
 * gridc = destination grid for the operation (rgrid *; input/output).
 * d     = multiplier for the operation (REAL; input).
 * grida = source grid for the operation (rgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_add_scaled(rgrid *gridc, REAL d, rgrid *grida) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_add_scaled(gridc, d, grida)) return;
#endif
#pragma omp parallel for firstprivate(d,nxy,nz,nzz,avalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] += d * avalue[ijnz + k];
  }
}

/*
 * Perform the following operation: gridc = gridc + d * grida * gridb.
 *
 * gridc = destination grid (rgrid *; output).
 * d     = constant multiplier (REAL; input).
 * grida = 1st source grid (rgrid *; input).
 * gridb = 2nd source grid (rgrid *; input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_add_scaled_product(rgrid *gridc, REAL d, rgrid *grida, rgrid *gridb) {
  
  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_add_scaled_product(gridc, d, grida, gridb)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,d) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] += d * avalue[ijnz + k] * bvalue[ijnz + k];
  }
}

/*
 * Operate on a grid by a given operator: gridc = O(grida).
 *
 * gridc    = destination grid (rgrid *; output).
 * grida    = source grid (rgrid *; input).
 * operator = operator (REAL (*)(REAL, void *); input). Args are value and param pointer.
 *            (i.e., a function mapping a given R-number to another)
 * params   = parameters for operator (void *).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_operate_one(rgrid *gridc, rgrid *grida, REAL (*operator)(REAL a, void *params), void *params) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    cuda_remove_block(avalue, 1);
    if(avalue != cvalue) cuda_remove_block(cvalue, 0);
  }
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,cvalue,operator,params) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = operator(avalue[ijnz + k], params);
  }
}

/*
 * Operate on a grid by a given operator and multiply: gridc = gridb * O(grida).
 *
 * gridc    = destination grid (rgrid *; output).
 * gridb    = multiply with this grid (rgrid *; input).
 * grida    = source grid (rgrid *; input).
 * operator = operator (REAL (*)(REAL), void *; input). Args are value and params.
 *            (i.e., a function mapping a given R-number to another)
 * params   = user supplied parameters to operator (void *).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 */

EXPORT void rgrid_operate_one_product(rgrid *gridc, rgrid *gridb, rgrid *grida, REAL (*operator)(REAL a, void *params), void *params) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    if(gridc != gridb && gridc != grida) cuda_remove_block(cvalue, 0);
    cuda_remove_block(bvalue, 1);
    if(grida != gridb) cuda_remove_block(avalue, 1);
  }
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,operator,params) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = bvalue[ijnz + k] * operator(avalue[ijnz + k], params);
  }
}

/* 
 * Operate on two grids and place the result in third: gridc = O(grida, gridb).
 * where O is the operator.
 *
 * gridc    = destination grid (rgrid *; output).
 * grida    = 1st source grid (rgrid *; input).
 * gridb    = 2nd source grid (rgrid *; input).
 * operator = operator mapping grida and gridb (REAL (*)(REAL, REAL); input).
 *
 * No return value.
 *
 * Note: source and destination grids may be the same.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void rgrid_operate_two(rgrid *gridc, rgrid *grida, rgrid *gridb, REAL (*operator)(REAL, REAL)) {

  INT ij, k, ijnz, nxy = gridc->nx * gridc->ny, nz = gridc->nz, nzz = gridc->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  REAL *cvalue = gridc->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    if(gridc != gridb && gridc != grida) cuda_remove_block(cvalue, 0);
    cuda_remove_block(bvalue, 1);
    if(grida != gridb) cuda_remove_block(avalue, 1);
  }
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,cvalue,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = operator(avalue[ijnz + k], bvalue[ijnz + k]);
  }
}

/*
 * Operate on a grid by a given operator.
 *
 * grid     = grid to be operated (rgrid *; input/output).
 * operator = operator (void (*)(REAL *); input).
 * 
 * No return value.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void rgrid_transform_one(rgrid *grid, void (*operator)(REAL *a)) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz, nzz = grid->nz2;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 1);
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,value,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      operator(&value[ijnz + k]);
  }
}

/*
 * Operate on two separate grids by a given operator.
 *
 * grida    = grid to be operated (rgrid *; input/output).
 * gridb    = grid to be operated (rgrid *; input/output).
 * operator = operator (void (*)(REAL *); input).
 * 
 * No return value.
 *
 * TODO: Allow parameter passing.
 *
 */

EXPORT void rgrid_transform_two(rgrid *grida, rgrid *gridb, void (*operator)(REAL *a, REAL *b)) {

  INT ij, k, ijnz, nxy = grida->nx * grida->ny, nz = grida->nz, nzz = grida->nz2;
  REAL *avalue = grida->value;
  REAL *bvalue = gridb->value;
  
#ifdef USE_CUDA
  if(cuda_status()) {
    cuda_remove_block(bvalue, 1);
    cuda_remove_block(avalue, 1);
  }
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,operator) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      operator(&avalue[ijnz + k], &bvalue[ijnz + k]);
  }
}

/*
 * Integrate over a grid.
 *
 * grid = grid to be integrated (rgrid *; input).
 *
 * Returns the integral value (REAL).
 *
 * NOTE: This will integrate also over the missing points due to BC
 *       such that the symmetry is preserved.
 *
 */

EXPORT REAL rgrid_integral(rgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL sum, step = grid->step;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_integral(grid, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(nx,ny,nz,grid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += rgrid_value_at_index(grid, i, j, k);
  
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * Integrate over a grid with limits.
 *
 * grid = grid to be integrated (rgrid *; input).
 * xl   = lower limit for x (REAL; input).
 * xu   = upper limit for x (REAL; input).
 * yl   = lower limit for y (REAL; input).
 * yu   = upper limit for y (REAL; input).
 * zl   = lower limit for z (REAL; input).
 * zu   = upper limit for z (REAL; input).
 *
 * Returns the integral value (REAL).
 *
 */

EXPORT REAL rgrid_integral_region(rgrid *grid, REAL xl, REAL xu, REAL yl, REAL yu, REAL zl, REAL zu) {

  INT iu, il, i, ju, jl, j, ku, kl, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x0 = grid->x0, y0 = grid->y0, z0 = grid->z0, sum;
  REAL step = grid->step;
  
  il = grid->nx / 2 + (INT) ((xl + x0) / step);
  iu = grid->nx / 2 + (INT) ((xu + x0) / step);
  jl = grid->ny / 2 + (INT) ((yl + y0) / step);
  ju = grid->ny / 2 + (INT) ((yu + y0) / step);
  kl = grid->nz / 2 + (INT) ((zl + z0) / step);
  ku = grid->nz / 2 + (INT) ((zu + z0) / step);

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_integral_region(grid, il, iu, jl, ju, kl, ku, &sum)) return sum;
#endif
  
  sum = 0.0;
#pragma omp parallel for firstprivate(il,iu,jl,ju,kl,ku,grid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = il; i <= iu; i++)
    for (j = jl; j <= ju; j++)
      for (k = kl; k <= ku; k++)
	sum += rgrid_value_at_index(grid, i, j, k);
  
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}
 
/* 
 * Integrate over the grid squared (int grid^2).
 *
 * grid = grid to be integrated (rgrid *; input).
 *
 * Returns the integral (REAL).
 *
 */

EXPORT REAL rgrid_integral_of_square(rgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL sum, step = grid->step, tmp;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_integral_of_square(grid, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(nx,ny,nz,grid) private(i,j,k,tmp) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
        tmp = rgrid_value_at_index(grid, i, j, k);
	sum += tmp * tmp;
      }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * Calculate overlap between two grids (int grida gridb).
 *
 * grida = 1st grid (rgrid *; input).
 * gridb = 2nd grid (rgrid *; input).
 *
 * Returns the value of the overlap integral (REAL).
 *
 */

EXPORT REAL rgrid_integral_of_product(rgrid *grida, rgrid *gridb) {

  INT i, j, k, nx = grida->nx, ny = grida->ny, nz = grida->nz;
  REAL sum, step = grida->step;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_integral_of_product(grida, gridb, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(nx,ny,nz,grida,gridb) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
	sum += rgrid_value_at_index(grida, i, j, k) * rgrid_value_at_index(gridb, i, j, k);
      }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * Calculate the expectation value of a grid over a grid.
 * (int opgrid dgrid^2).
 *
 * dgrid  = grid giving the probability density (dgrid^2) (rgrid *; input).
 * opgrid = grid to be averaged (rgrid *; input).
 *
 * Returns the average value (REAL *).
 *
 */

EXPORT REAL rgrid_grid_expectation_value(rgrid *dgrid, rgrid *opgrid) {

  INT i, j, k, nx = opgrid->nx, ny = opgrid->ny, nz = opgrid->nz;
  REAL sum, step = opgrid->step, tmp;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_grid_expectation_value(dgrid, opgrid, &sum)) return sum;
#endif

  sum = 0.0;
#pragma omp parallel for firstprivate(nx,ny,nz,dgrid,opgrid) private(i,j,k,tmp) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
        tmp = rgrid_value_at_index(dgrid, i, j, k);
	sum += tmp * tmp * rgrid_value_at_index(opgrid, i, j, k);
      }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;

  return sum;
}
 
/*
 * Calculate the expectation value of a function over a grid.
 * (int grida func grida = int func grida^2).
 *
 * func  = function to be averaged (REAL (*)(void *, REAL, REAL, REAL, REAL); input).
 *         The arguments are: optional arg, grida(x,y,z), x, y, z.
 * grida = grid giving the probability (grida^2) (rgrid *; input).
 *
 * Returns the average value (REAL).
 *
 */
 
EXPORT REAL rgrid_grid_expectation_value_func(void *arg, REAL (*func)(void *arg, REAL val, REAL x, REAL y, REAL z), rgrid *grida) {
   
  INT i, j, k, nx = grida->nx, ny = grida->ny, nz = grida->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL sum = 0.0, tmp, step = grida->step, x0 = grida->x0, y0 = grida->y0, z0 = grida->z0, x, y, z;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grida->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,nx2,ny2,nz2,grida,x0,y0,z0,step,func,arg) private(x,y,z,i,j,k,tmp) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz2)) * step - z0;
	tmp = rgrid_value_at_index(grida, i, j, k);
	sum += tmp * tmp * func(arg, tmp, x, y, z);
      }
    }
  }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/* 
 * Integrate over the grid multiplied by weighting function (int grid w(x)).
 *
 * grid   = grid to be integrated over (rgrid *; input).
 * weight = function defining the weight (REAL (*)(REAL, REAL, REAL); input). The arguments are (x,y,z) coordinates.
 * farg   = argument to the weight function (void *; input).
 *
 * Returns the value of the integral (REAL).
 *
 */

EXPORT REAL rgrid_weighted_integral(rgrid *grid, REAL (*weight)(void *farg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL sum = 0.0, step = grid->step, x0 = grid->x0, y0 = grid->y0, z0 = grid->z0, x, y, z;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,nx2,ny2,nz2,grid,x0,y0,z0,step,weight,farg) private(x,y,z,i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz2)) * step - z0;
	sum += weight(farg, x, y, z) * rgrid_value_at_index(grid, i, j, k);
      }
    }
  }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/* 
 * Integrate over square of the grid multiplied by weighting function (int grid^2 w(x)).
 *
 * grid   = grid to be integrated over (rgrid *; input).
 * weight = function defining the weight (REAL (*)(REAL, REAL, REAL); input).
 *          The arguments are (x, y, z) coordinates.
 * farg   = argument to the weight function (void *; input).
 *
 * Returns the value of the integral (REAL).
 *
 */

EXPORT REAL rgrid_weighted_integral_of_square(rgrid *grid, REAL (*weight)(void *farg, REAL x, REAL y, REAL z), void *farg) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL sum = 0.0, step = grid->step, x0 = grid->x0, y0 = grid->y0, z0 = grid->z0, x, y, z, tmp;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

#pragma omp parallel for firstprivate(nx,ny,nz,nx2,ny2,nz2,grid,x0,y0,z0,step,weight,farg) private(x,y,z,i,j,k,tmp) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz2)) * step - z0;
        tmp = rgrid_value_at_index(grid, i, j, k);
	sum += weight(farg, x, y, z) * tmp * tmp;
      }
    }
  }

  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * Print the grid into file (ASCII format).
 *
 * grid = grid to be printed out (rgrid *; input).
 * out  = output file pointer (FILE *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_print(rgrid *grid, FILE *out) {

  INT i, j, k;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < grid->nx; i++) {
    for(j = 0; j < grid->ny; j++) {
      for(k = 0; k < grid->nz; k++)
        fprintf(out, FMT_R "   ", rgrid_value_at_index(grid, i, j, k));
      fprintf(out, "\n");
    }
    fprintf(out, "\n");
  }
}

/*
 * Perform Fast Fourier Transformation of a grid.
 *
 * grid = grid to be Fourier transformed (input/output) (rgrid *; input/output).
 *
 * No return value.
 *
 * Notes: - The input grid is overwritten with the output.
 *        - No normalization is performed.
 *
 */

EXPORT void rgrid_fft(rgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cufft_fft(grid)) return;
#endif

  rgrid_fftw(grid);
}

/*
 * Perform inverse Fast Fourier Transformation of a grid.
 *
 * grid = grid to be inverse Fourier transformed (input/output) (rgrid *; input/output).
 *
 * No return value.
 *
 * Notes: - The input grid is overwritten with the output.
 *        - No normalization.
 *
 */

EXPORT void rgrid_inverse_fft(rgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cufft_fft_inv(grid)) return;
#endif

  rgrid_fftw_inv(grid);
}
 
/*
 * Perform scaled inverse Fast Fourier Transformation of a grid.
 *
 * grid = grid to be inverse Fourier transformed (input/output) (rgrid *; input/output).
 * c    = scaling factor (i.e. the output is multiplied by this constant) (REAL; input).
 *
 * No return value.
 *
 * Note: The input grid is overwritten with the output.
 *
 */
 
EXPORT void rgrid_scaled_inverse_fft(rgrid *grid, REAL c) {
   
  rgrid_inverse_fft(grid);
  rgrid_multiply(grid, c);  
}

/*
 * Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm.
 *
 * grid = grid to be inverse Fourier transformed (rgrid *; input/output).
 *
 * No return value.
 *
 * Note: The input grid is overwritten with the output.
 *
 */

EXPORT void rgrid_inverse_fft_norm(rgrid *grid) {

  rgrid_scaled_inverse_fft(grid, grid->fft_norm);
}

/*
 * Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm (including spatial step).
 *
 * grid = grid to be inverse Fourier transformed (rgrid *; input/output).
 *
 * No return value.
 *
 * Note: The input grid is overwritten with the output.
 *
 */

EXPORT void rgrid_inverse_fft_norm2(rgrid *grid) {

  rgrid_scaled_inverse_fft(grid, grid->fft_norm2);
}

/*
 * Convolute FFT transformed grids (periodic). To apply this on grids grida and gridb and place the result in gridc:
 * rgrid_fft(grida);
 * rgrid_fft(gridb);
 * rgrid_convolute(gridc, grida, gridb);
 * rgrid_inverse_fft_norm2(gridc);    // note: must be norm2
 * gridc now contains the convolution of grida and gridb.
 *
 * grida = 1st grid to be convoluted (rgrid *; input).
 * gridb = 2nd grid to be convoluted (rgrid *; input).
 * gridc = output (rgrid *; output).
 *
 * No return value.
 *
 * Notes: - the input/output grids may be the same.
 *        - this no longer multiplies the result by the norm (use *_inverse_fft_norm2).
 *
 * Convert from FFT to Fourier integral:
 *
 * Forward: Multiply FFT result by step^3.
 * Inverse: Multiply FFT result by (1 / (step * N))^3.
 *
 */

EXPORT void rgrid_fft_convolute(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL complex *avalue, *bvalue, *cvalue;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_convolute(gridc, grida, gridb)) return;
#endif

  /* int f(r) g(r-r') d^3r' = iF[ F[f] F[g] ] = (step / N)^3 iFFT[ FFT[f] FFT[g] ] */
  
  nx = gridc->nx;
  ny = gridc->ny;
  nz = gridc->nz2 / 2;  // nz2 = 2 * (nz / 2 + 1)
  nxy = nx * ny;
  avalue = (REAL complex *) grida->value;
  bvalue = (REAL complex *) gridb->value;
  cvalue = (REAL complex *) gridc->value;
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,avalue,bvalue,cvalue) private(i,j,ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      /* if odd */
      if ((i + j + k) & 1)
	cvalue[ijnz + k] = -avalue[ijnz + k] * bvalue[ijnz + k];
      else
	cvalue[ijnz + k] = avalue[ijnz + k] * bvalue[ijnz + k];
    }
  }
}

/*
 * Add grids in Fourier space: gridc = grida + gridb.
 *
 * gridc = output (rgrid *; output).
 * grida = 1st grid (rgrid *; input).
 * gridb = 2nd grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fft_sum(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT k, ij, ijnz, nx, ny, nz, nxy;
  REAL complex *avalue, *bvalue, *cvalue;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_sum(gridc, grida, gridb)) return;
#endif

  nx = gridc->nx;
  ny = gridc->ny;
  nz = gridc->nz2 / 2;  // nz2 = 2 * (nz / 2 + 1)
  nxy = nx * ny;
  avalue = (REAL complex *) grida->value;
  bvalue = (REAL complex *) gridb->value;
  cvalue = (REAL complex *) gridc->value;
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] + bvalue[ijnz + k];
  }
}

/*
 * Multiply grids in Fourier space.
 *
 * gridc = output (rgrid *; output).
 * grida = 1st grid (rgrid *; input).
 * gridb = 2nd grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fft_product(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT k, ij, ijnz, nx, ny, nz, nxy;
  REAL complex *avalue, *bvalue, *cvalue;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_product(gridc, grida, gridb)) return;
#endif

  nx = gridc->nx;
  ny = gridc->ny;
  nz = gridc->nz2 / 2;  // nz2 = 2 * (nz / 2 + 1)
  nxy = nx * ny;
  avalue = (REAL complex *) grida->value;
  bvalue = (REAL complex *) gridb->value;
  cvalue = (REAL complex *) gridc->value;
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,avalue,bvalue,cvalue) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cvalue[ijnz + k] = avalue[ijnz + k] * bvalue[ijnz + k];
  }
}

/*
 * Multiply grid by a constant in Fourier space (grid->value is complex) 
 *
 * grid = Grid to be multiplied (rgrid *; input/output).
 * c    = Multiply by this value (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fft_multiply(rgrid *grid, REAL c) {

  INT ij, k, ijnz, nxy = grid->nx * grid->ny, nz = grid->nz2 / 2;
  REAL complex *value = (REAL complex *) grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_multiply(grid, c)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,value,c) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= c;
  }
}

/*
 * Access grid point at given index (follows boundary condition).
 *
 * grid = grid to be accessed (rgrid *; input).
 * i    = index along x (INT; input).
 * j    = index along y (INT; input).
 * k    = index along z (INT; input).
 *
 * Returns grid value at index (i, j, k) (REAL).
 *
 * NOTE: This is *very* slow on cuda as it transfers each element individually.
 *
 */

EXPORT inline REAL rgrid_value_at_index(rgrid *grid, INT i, INT j, INT k) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz)
    return grid->value_outside(grid, i, j, k);

#ifdef USE_CUDA
  REAL value;
  if(cuda_find_block(grid->value)) {
    INT nx = grid->nx, ngpu2 = cuda_ngpus(), ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, gpu, idx;
    gpu = i / nnx1;
    if(gpu >= ngpu1) {
      idx = i - ngpu1 * nnx1;
      gpu = idx / nnx2 + ngpu1;
      idx = idx % nnx2;
    } else idx = i % nnx1;
    cuda_get_element(grid->value, (int) gpu, (size_t) ((idx * grid->ny + j) * grid->nz2 + k), sizeof(REAL), (void *) &value);
    return value;
  } else
#endif
  return grid->value[(i * grid->ny + j) * grid->nz2 + k];
}

/*
 * Set value to a grid point at given index.
 *
 * grid  = grid to be accessed (rgrid *; output).
 * i     = index along x (INT; input).
 * j     = index along y (INT; input).
 * k     = index along z (INT; input).
 * value = value to be set at (i, j, k) (REAL; input).
 *
 * No return value.
 *
 * NOTE: This is *very* slow on cuda as it transfers each element individually.
 *
 */

EXPORT inline void rgrid_value_to_index(rgrid *grid, INT i, INT j, INT k, REAL value) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz) return;

#ifdef USE_CUDA
  if(cuda_find_block(grid->value)) {
    INT nx = grid->nx, ngpu2 = cuda_ngpus(), ngpu1 = nx % ngpu2, nnx2 = nx / ngpu2, nnx1 = nnx2 + 1, gpu, idx;
    gpu = i / nnx1;
    if(gpu >= ngpu1) {
      idx = i % (ngpu1 * nnx1);
      gpu = idx / nnx2 + ngpu1;
    } else idx = i % nnx1;
    cuda_set_element(grid->value, (int) gpu, (size_t) ((idx * grid->ny + j) * grid->nz2 + k), sizeof(REAL), (void *) &value);
  } else
#endif
  grid->value[(i * grid->ny + j) * grid->nz2 + k] = value;
}

/*
 * Access grid point in Fourier space at given index (returns zere outside the grid).
 *
 * grid = grid to be accessed (rgrid *; input).
 * i    = index along x (INT; input).
 * j    = index along y (INT; input).
 * k    = index along z (INT; input).
 *
 * Returns grid value at index (i, j, k) (REAL complex).
 *
 * NOTE: This is *very* slow on cuda as it transfers each element individually.
 *
 */

EXPORT inline REAL complex rgrid_cvalue_at_index(rgrid *grid, INT i, INT j, INT k) {

  REAL complex *val;
  INT nzz = grid->nz2 / 2;

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= nzz) return 0.0;

#ifdef USE_CUDA
  REAL complex value;
  if(cuda_find_block(grid->value)) {
    INT ny = grid->ny, ngpu2 = cuda_ngpus(), ngpu1 = ny % ngpu2, nny2 = ny / ngpu2, nny1 = nny2 + 1, gpu, idx;
    gpu = j / nny1;
    if(gpu >= ngpu1) {
      idx = j - ngpu1 * nny1;
      gpu = idx / nny2 + ngpu1;
      idx = idx % nny2;
    } else idx = j % nny1;
    cuda_get_element(grid->value, (int) gpu, (size_t) ((i * ny + idx) * nzz + k), sizeof(REAL complex), (void *) &value);
    return value;
  } else
#endif
  val = (REAL complex *) grid->value;
  return val[(i * grid->ny + j) * nzz + k];
}

/*
 * Access grid point at given (x,y,z) point using linear interpolation.
 *
 * grid = grid to be accessed (rgrid *; input).
 * x    = x value (REAL; input).
 * y    = y value (REAL; input).
 * z    = z value (REAL; input).
 *
 * Returns grid value at (x,y,z) (REAL).
 *
 */

EXPORT inline REAL rgrid_value(rgrid *grid, REAL x, REAL y, REAL z) {

  REAL f000, f100, f010, f001, f110, f101, f011, f111;
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

  /*
   * linear interpolation 
   *
   * f(x,y) = (1-x) (1-y) (1-z) f(0,0,0) + x (1-y) (1-z) f(1,0,0) + (1-x) y (1-z) f(0,1,0) + (1-x) (1-y) z f(0,0,1) 
   *          + x     y   (1-z) f(1,1,0) + x (1-y)   z   f(1,0,1) + (1-x) y   z   f(0,1,1) +   x     y   z f(1,1,1)
   */ 
  f000 = rgrid_value_at_index(grid, i, j, k);
  f100 = rgrid_value_at_index(grid, i + 1, j, k);
  f010 = rgrid_value_at_index(grid, i, j + 1, k);
  f001 = rgrid_value_at_index(grid, i, j, k + 1);
  f110 = rgrid_value_at_index(grid, i + 1, j + 1, k);
  f101 = rgrid_value_at_index(grid, i + 1, j, k + 1);
  f011 = rgrid_value_at_index(grid, i, j + 1, k + 1);
  f111 = rgrid_value_at_index(grid, i + 1, j + 1, k + 1);
  
  omx = 1.0 - x;
  omy = 1.0 - y;
  omz = 1.0 - z;

  return omx * omy * omz * f000 + x * omy * omz * f100 + omx * y * omz * f010 + omx * omy * z * f001
    + x * y * omz * f110 + x * omy * z * f101 + omx * y * z * f011 + x * y * z * f111;
}

/*
 * Extrapolate between two different grid sizes.
 *
 * dest = Destination grid (rgrid *; output).
 * src  = Source grid (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_extrapolate(rgrid *dest, rgrid *src) {

  INT i, j, k, nx = dest->nx, ny = dest->ny, nz = dest->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, nzz = dest->nz2;
  REAL x0 = dest->x0, y0 = dest->y0, z0 = dest->z0;
  REAL step = dest->step, x, y, z;

#ifdef USE_CUDA
  cuda_remove_block(src->value, 1);
  cuda_remove_block(dest->value, 0);
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nzz,nx2,ny2,nz2,step,dest,src,x0,y0,z0) private(i,j,k,x,y,z) default(none) schedule(runtime)
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz2)) * step - z0;
	dest->value[(i * ny + j) * nzz + k] = rgrid_value(src, x, y, z);
      }
    }
  }
}

/*
 * Rotate a grid by a given angle around the z-axis.
 *
 * in  = original grid (to be rotated) (rgrid *; input).
 * out = output grid (rotated) (rgrid *; output).
 * th  = rotation angle in radians (REAL; input).
 *
 * Note: The grids in and out CANNOT be the same.
 *
 * No return value.
 *
 */

EXPORT void rgrid_rotate_z(rgrid *out, rgrid *in, REAL th) {

  grid_rotation *r;

  if(in == out) {
    fprintf(stderr,"libgrid: in and out grids in rgrid_rotate_z must be different\n");
    abort();
  }

  if(!(r = malloc(sizeof(grid_rotation)))) {
    fprintf(stderr, "libgrid: cannot allocate rotation structure.\n");
    abort();
  }
  r->rgrid = in;
  r->sinth = SIN(-th);  // same direction of rotation as -wLz
  r->costh = COS(th);
  rgrid_map(out, rgrid_value_rotate_z, (void *) r);
  free(r);
}

/*
 * Get the largest value contained in a grid.
 *
 * grid = Grid from which the largest value is to be searched from (rgrid *; input).
 *
 * Returns the largest value found (REAL).
 *
 */

EXPORT REAL rgrid_max(rgrid *grid) {

  REAL max_val;
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_max(grid, &max_val)) return max_val;
#endif
  max_val = rgrid_value_at_index(grid, 0, 0, 0);

#pragma omp parallel for firstprivate(nx, ny, nz, grid) private(i, j, k) reduction(max: max_val) default(none) schedule(runtime)
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++)
        if(rgrid_value_at_index(grid, i, j, k) > max_val) max_val = rgrid_value_at_index(grid, i, j, k);

  return max_val;
}

/*
 * Get the smallest value in a grid.
 *
 * grid = Grid from which the smallest value is to be searched from (rgrid *; input).
 *
 * Returns the smallest value found (REAL).
 *
 */

EXPORT REAL rgrid_min(rgrid *grid) {

  REAL min_val;
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_min(grid, &min_val)) return min_val;
#endif
  min_val = rgrid_value_at_index(grid, 0, 0, 0);

#pragma omp parallel for firstprivate(nx, ny, nz, grid) private(i, j, k) reduction(min: min_val) default(none) schedule(runtime)
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++)
        if(rgrid_value_at_index(grid, i, j, k) < min_val) min_val = rgrid_value_at_index(grid, i, j, k);

  return min_val;
}

/*
 * Add random noise to grid.
 *
 * grid  = Grid where the noise will be added (rgrid *; input/output).
 * scale = Scaling for random numbers [-scale,+scale[ (REAL; input).
 *
 */

EXPORT void rgrid_random(rgrid *grid, REAL scale) {

  static char been_here = 0;
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nzz = grid->nz2;

  if(!been_here) {
    srand48(time(0));
    been_here = 1;
  }

#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++)
        grid->value[(i * ny + j) * nzz + k] += scale * 2.0 * (((REAL) drand48()) - 0.5);
}

/*
 * Add random noise to grid (normal distribution).
 *
 * grid  = Grid where the noise will be added (rgrid *; input/output).
 * scale = Scaling for random numbers: zero mean and std dev of "scale" (REAL; input).
 *
 */

EXPORT void rgrid_random_normal(rgrid *grid, REAL scale) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nzz = grid->nz2;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_random_normal(grid, scale)) return;
  cuda_remove_block(grid->value, 1);
#endif

  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++)
        grid->value[(i * ny + j) * nzz + k] += scale * grid_random_normal();
}

/*
 * Add random noise to grid to part of grid.
 *
 * grid  = Grid where the noise will be added (cgrid *; input/output).
 * scale = Scaling for random numbers [-scale,+scale[ (REAL; input).
 * lx    = Lower limit index for x (INT; input).
 * hx    = Upper limit index for x (INT; input).
 * ly    = Lower limit index for y (INT; input).
 * hy    = Upper limit index for y (INT; input).
 * lz    = Lower limit index for z (INT; input).
 * hz    = Upper limit index for z (INT; input).
 *
 */

EXPORT void rgrid_random_index(rgrid *grid, REAL scale, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  static char been_here = 0;
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  INT i, j, k;

  if(!been_here) {
    srand48(time(0));
    been_here = 1;
  }

#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  if(hx > nx) hx = nx;
  if(hy > ny) hy = ny;
  if(hz > nz) hz = nz;
  if(lx < 0) lx = 0;
  if(ly < 0) ly = 0;
  if(lz < 0) lz = 0;

  // drand48 is not thread safe.
  for (i = lx; i < hx; i++)
    for (j = ly; j <  hy; j++)
      for (k = lz; k < hz; k++)
        grid->value[(i * ny + j) * nz + k] += scale * 2.0 * (((REAL) drand48()) - 0.5);
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

EXPORT void rgrid_zero_index(rgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nynz = ny * nz;
  REAL *value = grid->value;

  if(hx > nx) hx = nx;
  if(hy > ny) hy = ny;
  if(hz > nz) hz = nz;
  if(lx < 0) lx = 0;
  if(ly < 0) ly = 0;
  if(lz < 0) lz = 0;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_zero_index(grid, lx, hx, ly, hy, lz, hz)) return;
#endif

#pragma omp parallel for firstprivate(lx,hx,nx,ly,hy,ny,lz,hz,nz,value,nynz) private(i,j,k) default(none) schedule(runtime)
  for(i = lx; i < hx; i++)
    for(j = ly; j < hy; j++)
      for(k = lz; k < hz; k++)
        value[i * nynz + j * nz + k] = 0.0;
}

/*
 * Raise grid to integer power (fast).
 *
 * dst = Destination grid (rgrid *; output).
 * src = Source grid (rgrid *; input).
 * exponent = Exponent to be used (INT; input). This value can be negative.
 *
 * No return value.
 *
 */

EXPORT void rgrid_ipower(rgrid *dst, rgrid *src, INT exponent) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = src->nz2;
  REAL *avalue = src->value;
  REAL *bvalue = dst->value;
  
  if(exponent == 1) {
    rgrid_copy(dst, src);
    return;
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_ipower(dst, src, exponent)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,nzz,avalue,bvalue,exponent) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      bvalue[ijnz + k] = ipow(avalue[ijnz + k], exponent);
  }
}

/*
 * Set a value to given grid based on upper/lower limit thresholds of another grid (possibly the same).
 *
 * dest = destination grid (rgrid *; input/output).
 * src  = source grid for evaluating the thresholds (rgrid *; input).
 * ul   = upper limit threshold for the operation (REAL; input).
 * ll   = lower limit threshold for the operation (REAL; input).
 * uval = value to set when the upper limit was exceeded (REAL; input).
 * lval = value to set when the lower limit was exceeded (REAL; input).
 *
 * Source and destination may be the same.
 *
 * No return value.
 *
 */

EXPORT void rgrid_threshold_clear(rgrid *dest, rgrid *src, REAL ul, REAL ll, REAL uval, REAL lval) {

  INT k, ij, nx = src->nx, ny = src->ny, nz = src->nz, nzz = src->nz2, nxy = nx * ny, ijnz;
  REAL *dval = dest->value, *sval = src->value;

  if(dest->nx != src->nx || dest->ny != src->ny || dest->nz != src->nz) {
    fprintf(stderr, "libgrid: Incompatible grids for rgrid_threshold_clear().\n");
    abort();
  }

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_threshold_clear(dest, src, ul, ll, uval, lval)) return;
#endif

#pragma omp parallel for firstprivate(nxy,nz,ny,nzz,dval,sval,ll,ul,lval,uval) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for (k = 0; k < nz; k++) {
      if(sval[ijnz + k] < ll) dval[ijnz + k] = lval;
      if(sval[ijnz + k] > ul) dval[ijnz + k] = uval;
    }
  }
}

/*
 * Decompose a vector field into "compressible (irrotational)" (u) and "incompressible (rotational)" (w) parts:
 * v = w + u = w + \nabla q where div w = 0 and u = \nabla q (Hodge's decomposition).
 *
 * One can also take rot of both sides: rot v = rot w + rot u = rot w + rot grad q = rot w. So, 
 * u = grad q is the irrotational part and w is the is the rotational part.
 * 
 * This is performed through solving the Poisson equation: \Delta q = div v. Then u = \nabla q.
 * The incompressible part is then w = v - u.
 *
 * vx = X component of the vector field to be decomposed (rgrid *; input).
 * vy = Y component of the vector field to be decomposed (rgrid *; input).
 * vz = Z component of the vector field to be decomposed (rgrid *; input).
 * ux = X component of the compressible vector field (rgrid *; output).
 * uy = Y component of the compressible vector field (rgrid *; output).
 * uz = Z component of the compressible vector field (rgrid *; output).
 * wx = X component of the incompressible vector field (rgrid *; output). May be NULL if not needed.
 * wy = Y component of the incompressible vector field (rgrid *; output). May be NULL if not needed.
 * wz = Z component of the incompressible vector field (rgrid *; output). May be NULL if not needed.
 *
 * No return value.
 *
 * Note: uses either FD or FFT based on grid_analyze_method.
 *
 */

EXPORT void rgrid_hodge(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *ux, rgrid *uy, rgrid *uz, rgrid *wx, rgrid *wy, rgrid *wz) {

  if(grid_analyze_method) { /* FFT */
    rgrid_div(wx, vx, vy, vz);
    rgrid_fft(wx);
    rgrid_poisson(wx);
    rgrid_fft_gradient_x(wx, ux);
    rgrid_fft_gradient_y(wx, uy);
    rgrid_fft_gradient_z(wx, uz);
    rgrid_inverse_fft_norm(ux);
    rgrid_inverse_fft_norm(uy);
    rgrid_inverse_fft_norm(uz);
    rgrid_difference(wx, vx, ux);
    rgrid_difference(wy, vy, uy);
    rgrid_difference(wz, vz, uz);
  } else { /* FD */
    rgrid_div(wx, vx, vy, vz);
    rgrid_fft(wx);
    rgrid_poisson(wx);
    rgrid_inverse_fft_norm(wx);
    rgrid_fd_gradient_x(wx, ux);
    rgrid_fd_gradient_y(wx, uy);
    rgrid_fd_gradient_z(wx, uz);
    rgrid_difference(wx, vx, ux);
    rgrid_difference(wy, vy, uy);
    rgrid_difference(wz, vz, uz);
  }
}

/*
 * Decompose a vector field into "compressible (irrotational)" (u) and "incompressible (rotational)" (w) parts:
 * v = w + u = w + \nabla q where div w = 0 and u = \nabla q (Hodge's decomposition).
 *
 * One can also take rot of both sides: rot v = rot w + rot u = rot w + rot grad q = rot w. So, 
 * u = grad q is the irrotational part and w is the is the rotational part.
 * 
 * This is performed through solving the Poisson equation: \Delta q = div v. Then u = \nabla q.
 * The incompressible part is then w = v - u.
 *
 * This is special version of rgrid_hodge() such that it only computes the compressible part.
 *
 * vx        = X component of the vector field to be decomposed (rgrid *; input). Output: X component of compressible part.
 * vy        = Y component of the vector field to be decomposed (rgrid *; input). Output: Y component of compressible part.
 * vz        = Z component of the vector field to be decomposed (rgrid *; input). Output: Z component of compressible part.
 * workspace = Additional workspace required (rgrid *; output).
 *
 * No return value.
 *
 * Note: uses either FD or FFT based on grid_analyze_method.
 *
 */

EXPORT void rgrid_hodge_comp(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *workspace) {

  if(grid_analyze_method) { /* FFT */
    rgrid_div(workspace, vx, vy, vz);
    rgrid_fft(workspace);
    rgrid_poisson(workspace);
    rgrid_fft_gradient_x(workspace, vx);
    rgrid_fft_gradient_y(workspace, vy);
    rgrid_fft_gradient_z(workspace, vz);
    rgrid_inverse_fft_norm(vx);
    rgrid_inverse_fft_norm(vy);
    rgrid_inverse_fft_norm(vz);
  } else { /* FD */
    rgrid_div(workspace, vx, vy, vz);
    rgrid_fft(workspace);
    rgrid_poisson(workspace);
    rgrid_inverse_fft_norm(workspace);
    rgrid_fd_gradient_x(workspace, vx);
    rgrid_fd_gradient_y(workspace, vy);
    rgrid_fd_gradient_z(workspace, vz);
  }
}

/*
 * Decompose a vector field into "compressible (irrotational)" (u) and "incompressible (rotational)" (w) parts:
 * v = w + u = w + \nabla q where div w = 0 and u = \nabla q (Hodge's decomposition).
 *
 * One can also take rot of both sides: rot v = rot w + rot u = rot w + rot grad q = rot w. So, 
 * u = grad q is the irrotational part and w is the is the rotational part.
 * 
 * This is performed through solving the Poisson equation: \Delta q = div v. Then u = \nabla q.
 * The incompressible part is then w = v - u.
 *
 * This is special version of rgrid_hodge() such that it only computes the incompressible part.
 *
 * vx         = X component of the vector field to be decomposed (rgrid *; input). Output: X component of incompressible part.
 * vy         = Y component of the vector field to be decomposed (rgrid *; input). Output: Y component of incompressible part.
 * vz         = Z component of the vector field to be decomposed (rgrid *; input). Output: Z component of incompressible part.
 * workspace  = Additional workspace required (rgrid *; output).
 * workspace2 = Additional workspace required (rgrid *; output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_hodge_incomp(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *workspace, rgrid *workspace2) {

  if(grid_analyze_method == 1) { /* FFT */
    rgrid_div(workspace, vx, vy, vz);
    rgrid_fft(workspace);
    rgrid_poisson(workspace);
    rgrid_fft_gradient_x(workspace, workspace2);
    rgrid_inverse_fft_norm(workspace2);
    rgrid_difference(vx, vx, workspace2);
    rgrid_fft_gradient_y(workspace, workspace2);
    rgrid_inverse_fft_norm(workspace2);
    rgrid_difference(vy, vy, workspace2);
    rgrid_fft_gradient_z(workspace, workspace2);
    rgrid_inverse_fft_norm(workspace2);
    rgrid_difference(vz, vz, workspace2);
  } else { /* FD */
    rgrid_div(workspace, vx, vy, vz);
    rgrid_fft(workspace);
    rgrid_poisson(workspace);
    rgrid_inverse_fft_norm(workspace);
    rgrid_fd_gradient_x(workspace, workspace2);
    rgrid_difference(vx, vx, workspace2);
    rgrid_fd_gradient_y(workspace, workspace2);
    rgrid_difference(vy, vy, workspace2);
    rgrid_fd_gradient_z(workspace, workspace2);
    rgrid_difference(vz, vz, workspace2);
  }
}

/*
 * Compute spherical shell average in real space with respect to the grid origin
 * (result 1-D grid).
 *
 * E_ave(r) = \int E(r, \theta, \phi) sin(\theta) d\theta d\phi / (4 pi r^2)
 * 
 * input1  = Input grid 1 for averaging (rgrid *; input).
 * input2  = Input grid 2 for averaging (rgrid *; input). Can be NULL if N/A.
 * input3  = Input grid 3 for averaging (rgrid *; input). Can be NULL if N/A.
 * bins    = 1-D array for the averaged values (REAL *; output). This is an array with dimenion equal to nbins.
 * binstep = Binning step length (REAL; input).
 * nbins   = Number of bins requested (INT; input).
 * volel   = 1: Include 4pi r^2 volume element or 0: just calculate average (char; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_spherical_average(rgrid *input1, rgrid *input2, rgrid *input3, REAL *bins, REAL binstep, INT nbins, char volel) {

  INT nx = input1->nx, ny = input1->ny, nz = input1->nz, nzz = input1->nz2, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, idx, nxy = nx * ny;
  REAL step = input1->step, *value1 = input1->value, x0 = input1->x0, y0 = input1->y0, z0 = input1->z0, r, x, y, z;
  REAL *value2, *value3;
  INT *nvals, ij, i, j, k, ijnz;

  if(input2) value2 = input2->value;
  else value2 = NULL;
  if(input3) value3 = input3->value;
  else value3 = NULL;

#ifdef USE_CUDA
  cuda_remove_block(value1, 1);
  if(value2) cuda_remove_block(value2, 1);
  if(value3) cuda_remove_block(value3, 1);
#endif

  if(!(nvals = (INT *) malloc(sizeof(INT) * (size_t) nbins))) {
    fprintf(stderr, "libgrid: Out of memory in rgrid_spherical_average().\n");
    abort();
  }
  bzero(nvals, sizeof(INT) * (size_t) nbins);
  bzero(bins, sizeof(REAL) * (size_t) nbins);

// TODO: Can't execute in parallel (reduction for bins[idx] needed
//#pragma omp parallel for firstprivate(nx,ny,nz,nzz,nx2,ny2,nz2,nxy,step,value1,value2,value3,x0,y0,z0,bins,nbins,binstep,nvals) private(i,j,ij,ijnz,k,x,y,z,r,idx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      r = SQRT(x * x + y * y + z * z);
      idx = (INT) (r / binstep);
      if(idx < nbins) {
        bins[idx] = bins[idx] + value1[ijnz + k];
        if(value2) bins[idx] = bins[idx] + value2[ijnz + k];
        if(value3) bins[idx] = bins[idx] + value3[ijnz + k];
        nvals[idx]++;
      }
    }
  }
  if(volel) {
    for(k = 0, z = 0.0; k < nbins; k++, z += binstep) {
      if(nvals[k]) bins[k] = bins[k] * 4.0 * M_PI * z * z / (REAL) nvals[k];
    }
  } else {
    for(k = 0; k < nbins; k++)
      if(nvals[k]) bins[k] = bins[k] / (REAL) nvals[k];
  }
  free(nvals);
}

/*
 * Compute spherical shell average in the reciprocal space of power spectrum with respect to the grid origin
 * (result 1-D grid). Note: Uses power spectrum (Fourier space)!
 *
 * E_ave(k) = \int |sqrt(grid(k, \theta_k, \phi_k))|^2 sin(\theta_k) d\theta_k d\phi_k / (4pi k^2)
 * 
 * input1  = Input grid 1 for averaging (rgrid *; input), but this complex data (i.e., *after* FFT).
 * input2  = Input grid 2 for averaging (rgrid *; input), but this complex data (i.e., *after* FFT). Can be NULL if N/A.
 * input3  = Input grid 3 for averaging (rgrid *; input), but this complex data (i.e., *after* FFT). Can be NULL if N/A.
 * bins    = 1-D array for the averaged values (REAL *; output). This is an array with dimenion equal to nbins.
 * binstep = Binning step length for k (REAL; input). 
 * nbins   = Number of bins requested (INT; input).
 * volel   = 1: Include 4\pi k^2 volume element or 0: just calculate average (char; input).
 *
 * No return value.
 *
 * Note to compute E(k), grid should correspond to flux / sqrt(rho) = \sqrt(rho) * v.
 *
 */

EXPORT void rgrid_spherical_average_reciprocal(rgrid *input1, rgrid *input2, rgrid *input3, REAL *bins, REAL binstep, INT nbins, char volel) {

  INT nx = input1->nx, ny = input1->ny, nz = input1->nz2 / 2, idx, nxy = nx * ny;
  REAL step = input1->step, r, kx, ky, kz, norm2;
  REAL complex *value1 = (REAL complex *) input1->value, *value2, *value3;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step), ly = 2.0 * M_PI / (((REAL) ny) * step), lz = M_PI / (((REAL) nz - 1) * step);
  INT *nvals, ij, i, j, k, ijnz, nz2 = nz / 2;

  if(input2) value2 = (REAL complex *) input2->value;
  else value2 = NULL;
  if(input3) value3 = (REAL complex *) input3->value;
  else value3 = NULL;

#ifdef USE_CUDA
  cuda_remove_block(value1, 1);
  if(value2) cuda_remove_block(value2, 1);
  if(value3) cuda_remove_block(value3, 1);
#endif

  if(!(nvals = (INT *) malloc(sizeof(INT) * (size_t) nbins))) {
    fprintf(stderr, "libgrid: Out of memory in rgrid_spherical_average().\n");
    abort();
  }
  bzero(nvals, sizeof(INT) * (size_t) nbins);
  bzero(bins, sizeof(REAL) * (size_t) nbins);

// TODO: Can't execute in parallel (reduction for bins[idx] needed
//#pragma omp parallel for firstprivate(nx,ny,nz,nz2,nxy,step,lx,ly,lz,value1,value2,value3,bins,nbins,binstep,nvals) private(i,j,ij,ijnz,k,kx,ky,kz,r,idx) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    if(i < nx/2) 
      kx = ((REAL) i) * lx;
    else
      kx = -((REAL) (nx - i)) * lx;
    if(j < ny/2)
      ky = ((REAL) j) * ly;
    else
      ky = -((REAL) (ny - j)) * ly;
    for(k = 0; k < nz; k++) {
      if(k < nz2)
        kz = ((REAL) k) * lz; /* - kz0; */
      else
        kz = -((REAL) (nz - k)) * lz; /* - kz0; */
      r = SQRT(kx * kx + ky * ky + kz * kz);
      idx = (INT) (r / binstep);
      if(idx < nbins) {
        bins[idx] = bins[idx] + 2.0 * sqnorm(value1[ijnz + k]);
        if(value2) bins[idx] = bins[idx] + 2.0 * sqnorm(value2[ijnz + k]);
        if(value3) bins[idx] = bins[idx] + 2.0 * sqnorm(value3[ijnz + k]);
        nvals[idx]++;
      }
    }
  }
  norm2 = input1->step * input1->step * input1->step; norm2 *= norm2;
  if(volel) {
    for(k = 0, kz = 0.0; k < nbins; k++, kz += binstep)
      if(nvals[k]) bins[k] = norm2 * bins[k] * 4.0 * M_PI * kz * kz / (REAL) nvals[k];
  } else {
    for(k = 0; k < nbins; k++)
      if(nvals[k]) bins[k] = norm2 * bins[k] / (REAL) nvals[k];
  }
  free(nvals);
}

/*
 * Calculate running average to smooth unwanted high freq. components.
 *
 * dest   = destination grid (rgrid *).
 * source = source grid (rgrid *).
 * npts   = number of points used in running average (int). This smooths over +-npts points (effectively 2 X npts).
 *
 * No return value.
 *
 * Note: dest and source cannot be the same array.
 * 
 */

EXPORT void rgrid_npoint_smooth(rgrid *dest, rgrid *source, INT npts) {

  INT i, ip, j, jp, k, kp, nx = source->nx, ny = source->ny, nz = source->nz, pts;
  INT li, ui, lj, uj, lk, uk;
  REAL ave;

  if(npts < 2) {
    rgrid_copy(dest, source);
    return; /* nothing to do */
  }
  if(dest == source) {
    fprintf(stderr, "libgrid: dft_driver_npoint_smooth() - dest and source cannot be equal.\n");
    abort();
  }

  for (i = 0; i < nx; i++) 
    for (j = 0; j < ny; j++) 
      for (k = 0; k < nz; k++) {
        ave = 0.0;
        pts = 0;
        if(i - npts < 0) li = 0; else li = i - npts;
        if(j - npts < 0) lj = 0; else lj = j - npts;
        if(k - npts < 0) lk = 0; else lk = k - npts;
        if(i + npts > nx) ui = nx; else ui = i + npts;
        if(j + npts > ny) uj = ny; else uj = j + npts;
        if(k + npts > nz) uk = nz; else uk = k + npts;
        for(ip = li; ip < ui; ip++)
          for(jp = lj; jp < uj; jp++)
            for(kp = lk; kp < uk; kp++) {
              pts++;
              ave += rgrid_value_at_index(source, ip, jp, kp);
            }
        ave /= (REAL) pts;
        rgrid_value_to_index(dest, i, j, k, ave);
      }
}

/*
 * Apply user defined filter in Fourier space.
 *
 * grid   = Grid in Fourier space to be filtered (rgrid *; input/output).
 * func   = Filter function (REAL complex (*func)(void *farg, REAL kx, REAL ky, REAL kz); input).
 * farg   = Arguments to be passed to the function (void *; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fft_filter(rgrid *grid, REAL complex (*func)(void *, REAL, REAL, REAL), void *farg) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step;
  REAL complex *value = (REAL complex *) grid->value;
  
#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz2 / 2;
  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
  nxy = nx * ny;
  step = grid->step;
  
  lx = 2.0 * M_PI / (((REAL) nx) * step);
  ly = 2.0 * M_PI / (((REAL) ny) * step);
  lz = M_PI / (((REAL) nz - 1) * step);

#pragma omp parallel for firstprivate(nx2,ny2,nz2,func,farg,nx,ny,nz,nxy,step,value,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
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
      value[ijnz + k] *= (*func)(farg, kx, ky, kz);
    }
  }
}

/*
 * Lock grid into host memory. This does nothing on pure CPU-based systems.
 * On GPU-based systems it forces a given grid to stay in host memory.
 *
 * grid = grid to be host-locked (rgrid *; input).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_host_lock(rgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 1;
#endif
}

/*
 * Unlock grid. This does nothing on pure CPU-based systems.
 * On GPU-based systems it allows again the grid to move to GPU.
 *
 * grid = grid to be host-locked (rgrid *; input).
 * 
 * No return value.
 *
 */

EXPORT void rgrid_host_unlock(rgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 0;
#endif
}

/*
 * Set space flag for grid. On CPU systems this does nothing.
 * On GPU systems it affects the data storage order (INPLACE vs. INPLACE_SHUFFLED).
 *
 * Since the real and complex (R2C and C2R) storage formats are already different
 * on CPU systems, this routine probably does not need to be called. If there is
 * a complaint that the data is in wrong space (real vs. fourier) then there is
 * likely something wrong with the program.
 *
 * grid = Grid for the operation (rgrid *; input).
 * flag = 0: Real data or 1: fourier space data (char; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_fft_space(rgrid *grid, char space) {

#ifdef USE_CUDA
  gpu_mem_block *ptr;

  if(!(ptr = cuda_block_address(grid->value))) return; // Not on GPU
  if(space) ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  else ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
#endif
}

/*
 * @DOC{rgrid_multiply_by_x, Multiply real grid by coordinate x.
 * 
 * grid  = Grid to be operated on (rgrid *; input/output).
 *
 * No return value.}
 *
 */

EXPORT void rgrid_multiply_by_x(rgrid *grid) {

  INT i, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, nx2 = nx / 2;
  REAL x, step = grid->step;
  REAL x0 = grid->x0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_multiply_by_x(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nzz,nx2,nxy,step,value,x0) private(i,ij,ijnz,k,x) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    x = ((REAL) (i - nx2)) * step - x0;
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= x;
  }
}

/*
 * Multiply real grid by coordinate y.
 * 
 * grid  = Grid to be operated on (rgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_multiply_by_y(rgrid *grid) {

  INT j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, ny2 = ny / 2;
  REAL y, step = grid->step;
  REAL y0 = grid->y0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_multiply_by_y(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nzz,ny2,nxy,step,value,y0) private(j,ij,ijnz,k,y) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    j = ij % ny;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= y;
  }
}

/*
 * Multiply real grid by coordinate z.
 * 
 * grid  = Grid to be operated on (rgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void rgrid_multiply_by_z(rgrid *grid) {

  INT k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nzz = grid->nz2, nz2 = nz / 2;
  REAL z, step = grid->step;
  REAL z0 = grid->z0;
  REAL *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_multiply_by_z(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nzz,nz2,nxy,step,value,z0) private(ij,ijnz,k,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      value[ijnz + k] *= z;
    }
  }
}

/* 
 * Natural logarithm of absolute value of grid.
 *
 * dst     = Destination grid (rgrid *; output).
 * src     = Source grid (rgrid *; input).
 * eps     = Small number to avoid zero (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void rgrid_log(rgrid *dst, rgrid *src, REAL eps) {

  INT ij, k, ijnz, nxy = dst->nx * dst->ny, nz = dst->nz, nzz = dst->nz2;
  REAL *DST = dst->value;
  REAL *SRC = src->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_log(dst, src, eps)) return;
#endif
#pragma omp parallel for firstprivate(nxy,nz,nzz,DST,SRC,eps) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    for(k = 0; k < nz; k++)
      DST[ijnz + k] = LOG(FABS(SRC[ijnz + k]) + eps);
  }
}
