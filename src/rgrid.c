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
 * @FUNC{rgrid_alloc, "Allocate real grid"}
 * @DESC{"Allocate memory for real grid. Note that the grid is kept in padded form, which allows
          the use of in-place FFT. Also, for CUDA the array dimensions must be powers of two for efficiency"}
 * @ARG1{INT nx, "Number of points on the grid along x"}
 * @ARG2{INT ny, "Number of points on the grid along y"}
 * @ARG3{INT nz, "Number of points on the grid along z"}
 * @ARG4{REAL step, "Spatial step length on the grid"}
 * @ARG5{REAL (*value_outside), "Condition for accessing boundary points: RGRID_DIRICHLET_BOUNDARY: Dirichlet boundary, RGRID_NEUMANN_BOUNDARY: Neumann boundary, RGRID_PERIODIC_BOUNDARY: Periodic boundary, or user supplied function with pointer to grid and grid index as parameters to provide boundary access"}
 * @ARG6{void *outside_params_ptr, "Pointer for passing parameters for the given boundary access function. Use 0 to with the predefined boundary functions"}
 * @ARG7{char *id, "String ID describing the grid"}
 * @RVAL{rgrid *, "Returns pointer to the allocated grid or NULL on error"}
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
    if((nx & (nx-1)) || (ny & (ny-1)) || (nz & (nz-1))) {
      fprintf(stderr, "libgrid(cuda): Grid dimensions must be powers of two (performance & reduction).\n");
      abort();
    }
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
 * @FUNC{rgrid_clone, "Clone real grid"}
 * @DESC{"Clone a real grid with the parameters identical to the given grid 
          (except new grid-$>$value is allocated)"}
 * @ARG1{rgrid *grid, "Grid to be cloned"}
 * @ARG2{char *id, "String describing the grid (for debugging)"}
 * @DESC{rgird *, "Returns pointer to the new grid"}
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

#ifdef USE_CUDA
  /* Clear host lock */
  ngrid->host_lock = 0;
#endif

  ngrid->flag = 0;

  return ngrid;
}

/*
 * @FUNC{rgrid_claim, "Claim real grid"}
 * @DESC{"Claim real grid (simple locking system for the workspace model)"}
 * @ARG1{rgrid *grid, "Grid to be claimed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_release, "Release real grid"}
 * @DESC{"Release grid (simple locking system for the workspace model)"}
 * @ARG1{rgrid *grid, "Grid to be claimed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_set_origin, "Set real grid origin"}
 * @DESC{"Set the grid origin. The grid coordinates will be evaluated as:\\
          x(i)  = (i - nx / 2) * step - x0\\
          y(j)  = (j - ny / 2) * step - y0\\
          z(k)  = (k - nz / 2) * step - z0"}
 * @ARG1{rgrid *grid, "Grid for which the origin is to be defined"}
 * @ARG2{REAL x0, "X coordinate for the origin"}
 * @ARG3{REAL y0, "Y coordinate for the origin"}
 * @ARG4{REAL z0, "Z coordinate for the origin"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_set_origin(rgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 = x0;
  grid->y0 = y0;
  grid->z0 = z0;
}

/* 
 * @FUNC{rgrid_shift_origin, "Shift real grid origin"}
 * @DESC{"Shift the grid origin"}
 * @ARG1{rgrid *grid, "Grid for which the origin is to be shifted"}
 * @ARG2{REAL x0, "Shift in X coordinate"}
 * @ARG3{REAL y0, "Shift in Y coordinate"}
 * @ARG4{REAL z0, "Shift in Z coordinate"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_shift_origin(rgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 += x0;
  grid->y0 += y0;
  grid->z0 += z0;
}

/*
 * @FUNC{rgrid_set_momentum, "Set real grid momentum origin"}
 * @DESC{"Set the grid origin in momentum space (or the velocity of the frame of reference).
        Arguments kx0, ky0 and kz0 can be any real numbers but keep in mind that the grid
        will only contain the component k = 0 if they are multiples of:\\
        kx0min = 2 * M_PI / (NX * STEP)\\
        ky0min = 2 * M_PI / (NY * STEP)\\
        kz0min = 2 * M_PI / (NZ * STEP)"}
 * @ARG1{rgrid *grid, "Grid for which the momentum origin is to be defined"}
 * @ARG2{REAL kx0, "Momentum origin along the X axis"}
 * @ARG3{REAL ky0, "Momentum origin along the Y axis"}
 * @ARG4{REAL kz0, "Momentum origin along the Z axis"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_set_momentum(rgrid *grid, REAL kx0, REAL ky0, REAL kz0) {

  grid->kx0 = kx0;
  grid->ky0 = ky0;
  grid->kz0 = kz0;
}

/*
 * @FUNC{rgrid_free, "Free real grid"}
 * @DESC{"Free real grid memory"}
 * @ARG1{rgrid *grid, "Pointer to grid to be freed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_write, "Write real grid to disk"}
 * @DESC{"Write grid to disk in binary format"}
 * @ARG1{rgrid *grid, "Grid to be written to disk"}
 * @ARG2{FILE *out, "File handle for the file"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_read, "Read real grid from disk"}
 * @DESC{"Read real grid from disk in binary format"}
 * @ARG1{rgrid *grid, "Grid to be read. If NULL, a grid with the correct dimensions will be allocated. Note that the boundary condition will assigned to PERIODIC by default"}
 * @ARG2{FILE *in, "File handle for reading the file"}
 * @RVAL{rgrid *, "Returns pointer to the grid or NULL on error"}
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
 * @FUNC{rgrid_read_compat, "Read real grid from disk (compatibility)"}
 * @DESC{"Read grid from disk in binary format. This is compatible with old libgrid binary grid format.
          Since in-place FFT is used, the new grids have holes in them"}
 * @ARG1{rgrid *grid, "Grid to be read. If NULL, a grid with the correct dimensions will be allocated. Note that the boundary condition will assigned to PERIODIC by default"}
 * @ARG2{FILE *in, "File handle for reading the file"}
 * @RVAL{rgrid *, "Returns pointer to the grid or NULL on error"}
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
 * @FUNC{rgrid_read_grid_compat, "Read real grid from disk (compatibility)"}
 * @DESC{"Read in real grid from a binary file (.grd). Compatibility with old libgrid binary grid files"}
 * @ARG1{rgrid *grid, "Pointer to grid storage"}
 * @ARG2{char *file, "File name for the file. Note that the .grd extension should not be included"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_read_grid, "Read real grid from disk"}
 * @DESC{"Read in real grid from a binary file (.grd)"}
 * @ARG1{rgrid *grid, "Pointer to grid storage"}
 * @ARG2{char *file, "File name for the file. Note: the .grd extension should not be included"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_write_grid, "Write real grid to disk"}
 * @DESC{"Write real grid to disk including cuts along x, y, and z axes. See also rgrid_write()"}
 * @ARG1{char *basename, "Base file name to which suffixes (.x, .y, .z, and .grd) are added"}
 * @ARG2{rgrid *grid, "Grid to be written to disk"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_write_grid_reciprocal, "Write real grid to disk (reciprocal space)"}
 * @DESC{"Write real grid to disk including cuts along x, y, and z axes (grid in reciprocal/Fourier space)"}
 * @ARG1{char *basename, "Base filename where suffixes (.x, .y, .z, and .grd) are added"}
 * @ARG2{rgrid *grid, "Grid to be written to disk"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_write_grid_reciprocal(char *base, rgrid *grid) {

  FILE *fp;
  char file[2048];
  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz2 / 2;
  REAL x, y, z, step = grid->step;
  REAL complex tmp;

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
  j = 0;
  k = 0;
  for(i = 0; i < nx; i++) { 
    if (i < nx / 2)
      x = 2.0 * M_PI * ((REAL) i) / (((REAL) nx) * step) - grid->kx0;
    else 
      x = 2.0 * M_PI * ((REAL) (i - nx)) / (((REAL) nx) * step) - grid->kx0;
    tmp = rgrid_cvalue_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", x, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);

  /* Write cut along y-axis */
  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  i = 0;
  k = 0;
  for(j = 0; j < ny; j++) {
    if (j < ny / 2)
      y = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - grid->ky0;
    else 
      y = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - grid->ky0;
    tmp = rgrid_cvalue_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", y, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);

  /* Write cut along z-axis */
  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s for writing.\n", file);
    abort();
  }
  i = 0;
  j = 0;
  for(k = 0; k < nz; k++) {
    if (k < nz / 2)
      z = 2.0 * M_PI * ((REAL) k) / (((REAL) nz) * step) - grid->kz0;
    else 
      z = 2.0 * M_PI * ((REAL) (k - nz)) / (((REAL) nz) * step) - grid->kz0;
    tmp = rgrid_cvalue_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", z, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);
}

/*
 * @FUNC{rgrid_copy, "Copy real grid"}
 * @DESC{"Copy grid from one grid to another"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_shift, "Shift real grid spatially"}
 * @DESC{"Shift real grid by a given amount spatially"}
 * @ARG1{rgrid *shifted, "Destination grid for the operation"}
 * @ARG2{rgrid *grid, "Source grid for the operation"}
 * @ARG3{REAL x, "Shift spatially by this amount in x"}
 * @ARG4{REAL y, "Shift spatially by this amount in y"}
 * @ARG5{REAL z, "Shift spatially by this amount in z"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_zero, "Zero real grid"}
 * @DESC{"Zero the contents of real grid"}
 * @ARG1{rgrid *grid, "Grid to be zeroed"}
 * @RVAL{void, "No return value"}
 * 
 */

EXPORT void rgrid_zero(rgrid *grid) { 

  rgrid_constant(grid, 0.0); 
}

/* 
 * @FUNC{rgrid_constant, "Set real grid to constant value"}
 * @DESC{"Set real grid to a constant value"}
 * @ARG1{rgrid *grid, "Grid to be set"}
 * @ARG2{REAL c, "Constant value"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_product_func, "Multiply real grid by function"}
 * @DESC{"Multiply real grid by a function. The use specified function takes the following arguments:
          user data (void *), value at the current grid point (REAL), and the x, y, z values of the
          current grid point (REAL)"}
 * @ARG1{rgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data (can be NULL if not needed)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_map, "Map function onto real grid"}
 * @DESC{"Map a given function onto real grid. The function to be mapped takes the following arguments:
          user data (void *) and x,y,z coordinates of the current grid point (REAL)"}
 * @ARG1{rgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data (can be NULL if not needed)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_smooth_map, "Map function to real grid (linear smoothing)"}
 * @DESC{"Map a given function onto grid with linear smoothing.
          This can be used to weight the values at grid points to produce more
          accurate integration over the grid. The user specified function takes
          the following arguments: user specified data (void *) and 
          x,y,z coordinates of the current grid point (REAL)"}
 * @ARG1{rgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data (NULL if not needed)"}
 * @ARG4{INT ns, "Number of intermediate points to be used in smoothing"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_adaptive_map, "Map function onto real grid (adaptive linear smoothing)"}
 * @DESC{"Map a given function onto grid with linear smoothing.
          This can be used to weight the values at grid points to produce more
          accurate integration over the grid. Limits for intermediate steps and
          tolerance can be given. The function to be mapped takes the following arguments:
          user data (void *) and x,y,z coordinates of the current grid point (REAL).
 * @ARG1{rgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data"}
 * @ARG4{INT min_ns, "Minimum number of intermediate points to be used in smoothing"}
 * @ARG5{INT max_ns, "Maximum number of intermediate points to be used in smoothing"}
 * @ARG6{REAL tol, "Tolerance for the converge of integral over the function"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_sum, "Sum of real grids"}
 * @DESC{"Add two grids: gridc = grida + gridb. Note that the source and destination grids
          may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_difference, "Subtract real grids"}
 * @DESC{"Subtract two grids: gridc = grida - gridb. Note that the source and destination grids
          may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_product, "Product of real grids"}
 * @DESC{"Calculate product of two grids: gridc = grida * gridb. Note that the source and
          destination grids may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_power, "Rise real grid to given power"}
 * @DESC{"Rise a grid to given power. Note that the source and destination grids may be the same.
          This routine uses pow() -- use rgrid_ipower() with integer exponents"}
 * @ARG1{rgrid *gridb, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{REAL exponent, "Exponent to be used"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_abs_power, "Rise absolute value of grid to given power"}
 * @DESC{"Rise absolute value of a grid to given power (gridb = |grida|$^{exponent}$).
          The source and destination grids may be the same"}
 * @ARG1{rgrid *gridb, "Destination grid"}
 * @ARG2{rgrid *grida, "Source grid"}
 * @ARG3{REAL exponent, "Exponent to be used"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_division, "Divide real grids"}
 * @DESC{"Divide two grids: gridc = grida / gridb. Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_division_eps, "Divide real grids (numerically stable)"}
 * @DESC{"Safely divide two grids: gridc = grida / (gridb + eps). Note that the source and the
          destination grids may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @ARG4{REAL eps, "Epsilon to add to the divisor"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_add, "Add constant to real grid"}
 * @DESC{"Add a constant to a grid"}
 * @ARG1{rgrid *grid, "Grid where the constant is added"}
 * @ARG2{REAL c, "Constant to be added"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_multiply, "Multiply real grid by constant"}
 * @DESC{"Multiply real grid by a constant"}
 * @ARG1{rgrid *grid, "Grid to be multiplied"}
 * @ARG2{REAL c, "Constant multiplier"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_add_and_multiply, "Add and multiply real grid"}
 * @DESC{"Add and multiply: grid = (grid + ca) * cm"}
 * @ARG1{rgrid *grid, "Grid to be operated"}
 * @ARG2{REAL ca, "Constant to be added"}
 * @ARG3{REAL cm, "Constant multiplier"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_multiply_and_add, "Multiply and add real grid"}
 * @DESC{"Multiply and add: grid = cm * grid + ca"}
 * @ARG1{rgrid *grid, "Grid to be operated"}
 * @ARG2{REAL cm, "Constant multiplier"}
 * @ARG3{REAL ca, "Constant to be added"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_add_scaled, "Add scaled real grids"}
 * @DESC{"Add scaled grids (multiply/add): gridc = gridc + d * grida.
         Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid for the operation"}
 * @ARG2{REAL d, "Multiplier for the operation"}
 * @ARG3{rgrid *grida, "Source grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_add_scaled_product, "Add scaled product of real grids"}
 * @DESC{"Perform the following operation: gridc = gridc + d * grida * gridb.
          Note that the source and destination grids may be the same"}x
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{REAL d, "Constant multiplier"}
 * @ARG3{rgrid *grida, "1st source grid"}
 * @ARG4{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_operate_one, "Operate on real grid"}
 * @DESC{"Operate on a grid by a given operator: gridc = O(grida). The source and destination
          grids may be the same. The operator function takes the following arguments:
          the grid value at the current point (REAL) and user parameters (void *)"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "Source grid"}
 * @ARG3{REAL (*operator), "User specified operator"}
 * @ARG4{void *params, "User parameters for operator (may be NULL if not used)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_operate_one_product, "Operate on real grid and multiply"}
 * @DESC{"Operate on a grid by a given operator and multiply: gridc = gridb * O(grida).
          Note that the source and destination grids may be the same.
          The operator function takes the following arguments:
          grid value at the current point (REAL) and user parameters (void *)"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *gridb, "Multiply with this grid"}
 * @ARG3{rgrid *grida, "Source grid"}
 * @ARG4{REAL (*operator), "Operator function"}
 * @ARG5{void *params, "User supplied parameters to operator (may be NULL)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_operate_two, "Operate on two real grids"}
 * @DESC{"Operate on two grids and place the result in third: gridc = O(grida, gridb)
          where O is the operator. Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *gridc, "Destination grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @ARG4{REAL (*operator), "Operator function"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_transform_one, "Transform real grid"}
 * @DESC{"Transform real grid by a given function. The transform function takes one argument
          that is a pointer to the grid data at the current point (REAL *)"}
 * @ARG1{rgrid *grid, "Grid to be operated"}
 * @ARG2{void (*operator) = Transform function"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_transform_two, "Transform two real grids"}
 * @DESC{"Transform two real grids by a given function. The transform function takes two arguments
          that are pointers to the data of the two grids at the current point (REAL *)"}
 * @ARG1{rgrid *grida, "1st grid to be operated"}
 * @ARG2{rgrid *gridb, "2nd grid to be operated"}
 * @ARG3{void (*operator) = Transform function"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_integral, "Integrate real grid"}
 * @DESC{"Integrate over real grid. Note that this may be missing
          some boundary points due to the boundary condition"}
 * @ARG1{rgrid *grid, "Grid to be integrated"}
 * @RVAL{REAL, "Returns the integral value"}
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
 * @FUNC{rgrid_integral_region, "Integrate real grid with limits"}
 * @DESC{"Integrate over a grid with limits"}
 * @ARG1{rgrid *grid, "Grid to be integrated"}
 * @ARG2{REAL xl, "Lower limit for x"}
 * @ARG3{REAL xu, "Upper limit for x"}
 * @ARG4{REAL yl, "Lower limit for y"}
 * @ARG5{REAL yu, "Upper limit for y"}
 * @ARG6{REAL zl, "Lower limit for z"}
 * @ARG7{REAL zu, "Upper limit for z"}
 * @RVAL{REAL, "Returns the integral value"}
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
 * @FUNC{rgrid_integral_of_square, "Integrate over real grid squared"}
 * @DESC{"Integrate over the grid squared ($\int grid^2$)"}
 * @ARG1{rgrid *grid, "Grid to be integrated"}
 * @RVAL{REAL, "Returns the integral"}
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
 * @FUNC{rgrid_integral_of_product, "Integral of product of two real grids"}
 * @DESC{"Calculate overlap between two grids ($\int grida gridb$)"}
 * @ARG1{rgrid *grida, "1st grid"}
 * @ARG2{rgrid *gridb, "2nd grid"}
 * @RVAL{REAL, "Returns the value of the integral"}
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
 * @FUNC{rgrid_grid_expectation_value, "Expectation value of real grid"}
 * @DESC{"Calculate the expectation value of a grid over a grid ($\int opgrid dgrid^2$)"}
 * @ARG1{rgrid *dgrid, "Grid giving the probability density (dgrid$^2$)"}
 * @ARG2{rgrid *opgrid, "Operator grid"}
 * @RVAL{REAL, "Returns the expectation value"}
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
 * @FUNC{rgrid_grid_expectation_value_func, "Expectation value of function over real grid squared"}
 * @DESC{"Calculate the expectation value of a function over a grid squared 
          ($\int grida func grida = \int func grida^2$).
          The function takes three arguments: user data (void *), value at the current grid point (REAL),
          and the x,y,z coordinates (REAL)"}
 * @ARG1{REAL (*func), "Function for which to calculate the expectation value"}
 * @ARG2{rgrid *grida, "Grid giving the probability (grida$^2$)"}
 * @RVAL{REAL, "Returns the expectation value"}
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
 * @FUNC{rgrid_weighted_integral, "Weighted integral over real grid"}
 * @DESC{"Integrate over the grid multiplied by weighting function ($\int grid w(x)$).
          The weighting function takes four arguments: user data (void *) and
          x,y,z coordinates of the current grid point (REAL)"}
 * @ARG1{rgrid *grid, "Grid to be integrated over"}
 * @ARG2{REAL (*weight), "Function defining the weight"}
 * @ARG3{void *farg, "User data for weighting function (may be NULL)"}
 * @RVAL{REAL, "Returns the value of the integral"}
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
 * @FUNC{rgrid_weighted_integral_of_square, "Integrate real grid squared weighted by function"}
 * @DESC{"Integrate over square of the grid multiplied by weighting function ($\int grid^2 w(x)$).
          The weighting function takes four arguments: user data (void *) and
          x,y,z coordinates of the current grid point (REAL)"}
 * @ARG1{rgrid *grid, "Grid to be integrated over"}
 * @ARG2{REAL (*weight), "Function defining the weight"}
 * @ARG3{void *farg, "User data for the weight function (may be NULL)"}
 * @RVAL{REAL, "Returns the value of the integral"}
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
 * @FUNC{rgrid_fft, "FFT of real grid"}
 * @DESC{"Perform Fast Fourier Transformation (FFT) of real grid (in-place). No normalization is performed"}
 * @ARG1{rgrid *grid, "Grid to be Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft(rgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cufft_fft(grid)) return;
#endif

  rgrid_fftw(grid);
}

/*
 * @FUNC{rgrid_inverse_fft, "Inverse FFT of real grid"}
 * @DESC{"Perform noninverse Fast Fourier Transformation of a grid (in-place). No normalization is performed"}
 * @ARG1{rgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_inverse_fft(rgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cufft_fft_inv(grid)) return;
#endif

  rgrid_fftw_inv(grid);
}
 
/*
 * @FUNC{rgrid_scaled_inverse_fft, "Scaled inverse FFT of real grid"}
 * @DESC{"Perform scaled inverse Fast Fourier Transformation of a grid"}
 * @ARG1{rgrid *grid, "Grid to be inverse Fourier transformed"}
 * @ARG2{REAL c, "Scaling factor"}
 * @RVAL{void, "No return value"}
 *
 */
 
EXPORT void rgrid_scaled_inverse_fft(rgrid *grid, REAL c) {
   
  rgrid_inverse_fft(grid);
  rgrid_multiply(grid, c);  
}

/*
 * @FUNC{rgrid_inverse_fft_norm, "Normalized inverse FFT of real grid"}
 * @DESC{"Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm (number of grid points)"}
 * @ARG1{rgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_inverse_fft_norm(rgrid *grid) {

  rgrid_scaled_inverse_fft(grid, grid->fft_norm);
}

/*
 * @FUNC{rgrid_inverse_fft_norm2, "Normalized inverse FFT of real grid (spatial step included)"}
 * @DESC{"Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm (including spatial step)"}
 * @ARG1{rgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_inverse_fft_norm2(rgrid *grid) {

  rgrid_scaled_inverse_fft(grid, grid->fft_norm2);
}

/*
 * @FUNC{rgrid_fft_convolute, "Convolute two real grids in reciprocal space"}
 * @DESC{"Convolute FFT transformed grids. To apply this on grids grida and gridb and 
          place the result in gridc (integral):\\
          rgrid_fft(grida);\\
          rgrid_fft(gridb);\\
          rgrid_convolute(gridc, grida, gridb);\\
          rgrid_inverse_fft_norm2(gridc); // includes multiplication by step$^3$\\
          gridc now contains the convolution of grida and gridb\\
          In general,to convert from FFT to Fourier integral:
          Forward: Multiply FFT result by step$^3$\\
          Inverse: Multiply FFT result by (1 / (step * N))$^3$"}
 * @ARG1{rgrid *grida, "1st grid to be convoluted"}
 * @ARG2{rgrid *gridb, "2nd grid to be convoluted"}
 * @ARG3{rgrid *gridc, "output"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fft_sum, "Sum real grids in reciprocal space"}
 * @DESC{"Sum grids in the reciprocal space: gridc = grida + gridb"}
 * @ARG1{rgrid *gridc, "Output grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fft_product, "Product of real grids in reciprocal space"}
 * @DESC{"Product of two grids in reciprocal space"}
 * @ARG1{rgrid *gridc, "Output grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fft_product_conj, "Conjugate product of real grids in reciprocal space"}
 * @DESC{"Multiply conj(grida) * gridb in reciprocal space"}
 * @ARG1{rgrid *gridc, "Output grid"}
 * @ARG2{rgrid *grida, "1st source grid"}
 * @ARG3{rgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_product_conj(rgrid *gridc, rgrid *grida, rgrid *gridb) {

  INT k, ij, ijnz, nx, ny, nz, nxy;
  REAL complex *avalue, *bvalue, *cvalue;

#ifdef USE_CUDA
  if(cuda_status() && !rgrid_cuda_fft_product_conj(gridc, grida, gridb)) return;
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
      cvalue[ijnz + k] = CONJ(avalue[ijnz + k]) * bvalue[ijnz + k];
  }
}

/*
 * @FUNC{rgrid_fft_multiply, "Multiply real grid by constant in reciprocal space"}
 * @DESC{"Multiply grid by a constant in Fourier space (grid-$>$value is complex!)"}
 * @ARG1{rgrid *grid, "Grid to be multiplied"}
 * @ARG2{REAL complex c, "Multiply by this value"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_multiply(rgrid *grid, REAL complex c) {

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
 * @FUNC{rgrid_value_at_index, "Access real grid point at given index"}
 * @DESC{"Access grid point at given index (follows boundary condition).
          Note that this is *very* slow on CUDA as it transfers each element individually"}
 * @ARG1{rgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @RVAL{REAL, "Returns grid value at index (i, j, k)"}
 *
 */

EXPORT inline REAL rgrid_value_at_index(rgrid *grid, INT i, INT j, INT k) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz)
    return grid->value_outside(grid, i, j, k);

#ifdef USE_CUDA
  // Too much cuda code here, move to cuda.c eventually */
  REAL value;
  INT nx = grid->nx, ngpu2 = cuda_ngpus(), ngpu1, nnx2, nnx1, gpu, idx;
  gpu_mem_block *ptr;
  if((ptr = cuda_find_block(grid->value))) {    
    if(ptr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
      fprintf(stderr, "libgrid: rgrid_value_at_index() GPU data in reciprocal space.\n");
      abort();
    }
    ngpu1 = nx % ngpu2;
    nnx2 = nx / ngpu2;
    nnx1 = nnx2 + 1;
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
 * @FUNC{rgrid_value_to_index, "Write value to real grid at given index"}
 * @DESC{"Set value to a grid point at given index. 
          Note that this is *very* slow on CUDA as it transfers each element individually"}
 * @ARG1{rgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @ARG5{REAL value, "Value to be set at (i, j, k)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT inline void rgrid_value_to_index(rgrid *grid, INT i, INT j, INT k, REAL value) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz) {
    fprintf(stderr, "libgrid: rgrid_value_to_index() access outside grid.\n");
    abort();
  }

#ifdef USE_CUDA
  // Too much cuda code here, move to cuda.c eventually */
  INT nx = grid->nx, ngpu2 = cuda_ngpus(), ngpu1, nnx2, nnx1, gpu, idx;
  gpu_mem_block *ptr;
  if((ptr = cuda_find_block(grid->value))) {    
    if(ptr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
      fprintf(stderr, "libgrid: rgrid_value_to_index() GPU data in reciprocal space.\n");
      abort();
    }
    ngpu1 = nx % ngpu2;
    nnx2 = nx / ngpu2;
    nnx1 = nnx2 + 1;
    gpu = i / nnx1;
    if(gpu >= ngpu1) {
      idx = i - ngpu1 * nnx1;
      gpu = idx / nnx2 + ngpu1;
      idx = idx % nnx2;
    } else idx = i % nnx1;
    cuda_set_element(grid->value, (int) gpu, (size_t) ((idx * grid->ny + j) * grid->nz2 + k), sizeof(REAL), (void *) &value);
    return;
  } else
#endif
  grid->value[(i * grid->ny + j) * grid->nz2 + k] = value;
}

/*
 * @FUNC{rgrid_cvalue_at_index, "Access complex grid value at given index (reciprocal space)"}
 * @DESC{"Access grid point in Fourier space at given index (returns zere outside the grid).
          This is very slow on CUDA as it transfers each element individually"}
 * @ARG1{rgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @RVAL{REAL complex, "Returns grid value at index (i, j, k)"}
 *
 */

EXPORT inline REAL complex rgrid_cvalue_at_index(rgrid *grid, INT i, INT j, INT k) {

  REAL complex *val;
  INT nzz = grid->nz2 / 2;

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= nzz) return 0.0;

#ifdef USE_CUDA
  // Too much cuda code here, move to cuda.c eventually */
  REAL complex value;
  INT ny = grid->ny, ngpu2 = cuda_ngpus(), ngpu1, nny2, nny1, gpu, idx;
  gpu_mem_block *ptr;
  if((ptr = cuda_find_block(grid->value))) {    
    if(ptr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
      fprintf(stderr, "libgrid: rgrid_cvalue_at_index() GPU data in real space.\n");
      abort();
    }
    ngpu1 = ny % ngpu2;
    nny2 = ny / ngpu2;
    nny1 = nny2 + 1;
    gpu = j / nny1;
    if(gpu >= ngpu1) {
      idx = j - ngpu1 * nny1;
      gpu = idx / nny2 + ngpu1;
      idx = idx % nny2;
      cuda_get_element(grid->value, (int) gpu, (size_t) ((i * nny2 + idx) * nzz + k), sizeof(REAL complex), (void *) &value);
    } else {
      idx = j % nny1;
      cuda_get_element(grid->value, (int) gpu, (size_t) ((i * nny1 + idx) * nzz + k), sizeof(REAL complex), (void *) &value);
    }
    return value;
  } else
#endif
  val = (REAL complex *) grid->value;
  return val[(i * grid->ny + j) * nzz + k];
}

/*
 * @FUNC{rgrid_cvalue_to_index, "Set complex value to grid point (reciprocal space)"}
 * @DESC{"Set grid point in Fourier space at given index.
          Note that this is very slow on CUDA as it transfers each element individually"}
 * @ARG1{rgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @ARG5{REAL complex value, "Value to be set at (i, j, k)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT inline void rgrid_cvalue_to_index(rgrid *grid, INT i, INT j, INT k, REAL complex value) {

  REAL complex *val;
  INT nzz = grid->nz2 / 2;

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= nzz) {
    fprintf(stderr, "libgrid: rgrid_cvalue_to_index() access outside grid.\n");
    abort();
  }

#ifdef USE_CUDA
  // Too much cuda code here, move to cuda.c eventually */
  INT ny = grid->ny, ngpu2 = cuda_ngpus(), ngpu1, nny2, nny1, gpu, idx;
  gpu_mem_block *ptr;
  if((ptr = cuda_find_block(grid->value))) {    
    if(ptr->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE_SHUFFLED) {
      fprintf(stderr, "libgrid: rgrid_cvalue_to_index() GPU data in real space.\n");
      abort();
    }
    ngpu1 = ny % ngpu2;
    nny2 = ny / ngpu2;
    nny1 = nny2 + 1;
    gpu = j / nny1;
    if(gpu >= ngpu1) {
      idx = j - ngpu1 * nny1;
      gpu = idx / nny2 + ngpu1;
      idx = idx % nny2;
      cuda_set_element(grid->value, (int) gpu, (size_t) ((i * nny2 + idx) * nzz + k), sizeof(REAL complex), (void *) &value);
    } else {
      idx = j % nny1;
      cuda_set_element(grid->value, (int) gpu, (size_t) ((i * nny1 + idx) * nzz + k), sizeof(REAL complex), (void *) &value);
    }
    return;
  } else
#endif
  val = (REAL complex *) grid->value;
  val[(i * grid->ny + j) * nzz + k] = value;
}

/*
 * @FUNC{rgrid_value, "Access real grid point at (x, y, z) with linear interpolation"}
 * @DESC{"Access grid point at given (x,y,z) point using linear interpolation"}
 * @ARG1{rgrid *grid, "Grid to be accessed"}
 * @ARG2{REAL x, "x value"}
 * @ARG3{REAL y, "y value"}
 * @ARG4{REAL z, "z value"}
 * @RVAL{REAL, "Returns grid value at (x,y,z)"}
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
 * @FUNC{rgrid_extrapolate, "Extrapolate between different size real grids"}
 * @DESC{"Extrapolate between two different grid sizes"}
 * @ARG1{rgrid *dest, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_rotate_z, "Rotate real grid around z-axis"}
 * @DESC{"Rotate a grid by a given angle around the z-axis. Note that the in and out grids
          CANNOT be the same"}
 * @ARG1{rgrid *in, "Original grid (to be rotated)"}
 * @ARG2{rgrid *out, "Output grid (rotated)"}
 * @ARG3{REAL th, "Rotation angle in radians"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_max, "Find maximum value in real grid"}
 * @DESC{"Get the largest value contained in a grid"}
 * @ARG1{rgrid *grid, "Grid from which the largest value is to be searched from"}
 * @RVAL{REAL, "Returns the largest value found"}
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
 * @FUNC{rgrid_min, "Find minimum value in real grid"}
 * @DESC{"Get the smallest value contained in a grid"}
 * @ARG1{rgrid *grid, "Grid from which the smallest value is to be searched from"}
 * @RVAL{REAL, "Returns the smallest value found"}
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
 * @FUNC{rgrid_zero_index, "Zero given range in real grid"}
 * @DESC{"Zero a given index range of a complex grid over $[lx, hx[$ X $[ly, hy[$ X $[lz, hz[$"}
 * @ARG1{rgrid *grid, "Grid to be operated on"}
 * @ARG2{INT lx, "Lower limit for x index"}
 * @ARG3{INT hx, "Upper limit for x index"}
 * @ARG4{INT ly, "Lower limit for y index"}
 * @ARG5{INT hy, "Upper limit for y index"}
 * @ARG6{INT lz, "Lower limit for z index"}
 * @ARG7{INT hz, "Upper limit for z index"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_ipower, "Rise real grid to integer power"}
 * @DESC{"Raise grid to integer power (fast)"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid"}
 * @ARG3{INT exponent, "Exponent to be used. This value can be negative"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_threshold_clear, "Clear part of real grid based on threshold values"}
 * @DESC{"Set a value to given grid based on upper/lower limit thresholds of another grid (possibly the same).
          Note that the source and destination grids may be the same"}
 * @ARG1{rgrid *dest, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid for evaluating the thresholds"}
 * @ARG3{REAL ul, "Upper limit threshold value"}
 * @ARG4{REAL ll, "Lower limit threshold value"}
 * @ARG5{REAL uval, "Value to set when the upper limit was exceeded"}
 * @ARG6{REAL lval, "Value to set when the lower limit was exceeded"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_hodge, "Hodge decomposition of vector field"}
 * @DESC{"Decompose a vector field into compressible (irrotational) ($u$) and incompressible (rotational)
          ($w$) parts: $v = w + u = w + \nabla q$ where $div w = 0$ and $u = \nabla q$ (Hodge's decomposition).
          One can also take rot of both sides: $rot v = rot w + rot u = rot w + rot grad q = rot w$. So, 
          $u = \nabla q$ is the irrotational part and $w$ is the is the rotational part.
          This is performed through solving the Poisson equation: $\Delta q = div v$. Then $u = \nabla q$.
          The incompressible part is then $w = v - u$.
          Note that this routine uses either FD or FFT based on grid_analyze_method setting"}
 * @ARG1{rgrid *vx, "X component of the vector field to be decomposed"}
 * @ARG2{rgrid *vy, "Y component of the vector field to be decomposed"}
 * @ARG3{rgrid *vz, "Z component of the vector field to be decomposed"}
 * @ARG4{rgrid *ux, "X component of the compressible vector field"}
 * @ARG5{rgrid *uy, "Y component of the compressible vector field"}
 * @ARG6{rgrid *uz, "Z component of the compressible vector field"}
 * @ARG7{rgrid *wx, "X component of the incompressible vector field. May be NULL if not needed"}
 * @ARG8{rgrid *wy, "Y component of the incompressible vector field. May be NULL if not needed"}
 * @ARG9{rgrid *wz, "Z component of the incompressible vector field. May be NULL if not needed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_hodge(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *ux, rgrid *uy, rgrid *uz, rgrid *wx, rgrid *wy, rgrid *wz) {

  if(grid_analyze_method) { /* FFT */
    rgrid_fft(vx);
    rgrid_fft(vy);
    rgrid_fft(vz);
    rgrid_fft_div(wx, vx, vy, vz);
    rgrid_inverse_fft_norm(vx);
    rgrid_inverse_fft_norm(vy);
    rgrid_inverse_fft_norm(vz);
    rgrid_fft_poisson(wx);
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
    rgrid_fft_poisson(wx);
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
 * @FUNC{rgrid_hodge_comp, "Hodge decomposition of vector field (compressible)"}
 * @DESC{"See documentation for rgrid_hodge(). This routine computes only the compressible portion"}
 * @ARG1{rgrid *vx, "X component of the vector field to be decomposed. On exit: X component of compressible part"}
 * @ARG2{rgrid *vy, "Y component of the vector field to be decomposed. On exit: Y component of compressible part"}
 * @ARG3{rgrid *vz, "Z component of the vector field to be decomposed. On exit: Z component of compressible part"}
 * @ARG4{rgrid *workspace, "Additional workspace required"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_hodge_comp(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *workspace) {

  if(grid_analyze_method) { /* FFT */
    rgrid_fft(vx);
    rgrid_fft(vy);
    rgrid_fft(vz);
    rgrid_fft_div(workspace, vx, vy, vz);
    rgrid_fft_poisson(workspace);
    rgrid_fft_gradient_x(workspace, vx);
    rgrid_fft_gradient_y(workspace, vy);
    rgrid_fft_gradient_z(workspace, vz);
    rgrid_inverse_fft_norm(vx);
    rgrid_inverse_fft_norm(vy);
    rgrid_inverse_fft_norm(vz);
  } else { /* FD */
    rgrid_div(workspace, vx, vy, vz);
    rgrid_fft(workspace);
    rgrid_fft_poisson(workspace);
    rgrid_inverse_fft_norm(workspace);
    rgrid_fd_gradient_x(workspace, vx);
    rgrid_fd_gradient_y(workspace, vy);
    rgrid_fd_gradient_z(workspace, vz);
  }
}

/*
 * @FUNC{rgrid_hodge_incomp, "Hodge decomposition of vector field (incompressible)"}
 * @DESC{"See documentation for rgrid_hodge(). This routine computes only the incompressible portion"}
 * @ARG1{rgrid *vx, "X component of the vector field to be decomposed. On exit: X component of incompressible part"}
 * @ARG2{rgrid *vy, "Y component of the vector field to be decomposed. On exit: Y component of incompressible part"}
 * @ARG3{rgrid *vz, "Z component of the vector field to be decomposed. On exit: Z component of incompressible part"}
 * @ARG4{rgrid *workspace, "Additional workspace required"}
 * @ARG5{rgrid *workspace2, "Additional workspace required"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_hodge_incomp(rgrid *vx, rgrid *vy, rgrid *vz, rgrid *workspace, rgrid *workspace2) {

  if(grid_analyze_method) { /* FFT */
    rgrid_fft(vx);
    rgrid_fft(vy);
    rgrid_fft(vz);
    rgrid_fft_div(workspace, vx, vy, vz);
    rgrid_fft_poisson(workspace);
    rgrid_inverse_fft_norm(vx);
    rgrid_inverse_fft_norm(vy);
    rgrid_inverse_fft_norm(vz);
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
    rgrid_fft_poisson(workspace);
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
 * @FUNC{rgrid_spherical_average, "Spherical average of real grid"}
 * @DESC{"Compute spherical shell average in real space with respect to the grid origin
          (result 1-D grid). Averaging can be done with up to three grids"}
 * @ARG1{rgrid *input1, "Input grid 1 for averaging"}
 * @ARG2{rgrid *input2, "Input grid 2 for averaging. Can be NULL if N/A"}
 * @ARG3{rgrid *input3, "Input grid 3 for averaging. Can be NULL if N/A"}
 * @ARG4{REAL *bins, "1-D array for the averaged values. This is an array with dimension equal to nbins"}
 * @ARG5{REAL binstep, "Binning step length"}
 * @ARG6{INT nbins, "Number of bins requested"}
 * @ARG7{char volel, "1: direct sum or 0: radial average"}
 * @RVAL{void, "No return value"}
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
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nzz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      r = SQRT(x * x + y * y + z * z);
      idx = (INT) (0.5 + r / binstep);
      if(idx < nbins) {
        bins[idx] = bins[idx] + value1[ijnz + k];
        if(value2) bins[idx] = bins[idx] + value2[ijnz + k];
        if(value3) bins[idx] = bins[idx] + value3[ijnz + k];
        nvals[idx]++;
      }
    }
  }
  switch(volel) {
    case 0: // radial average
      for(k = 0; k < nbins; k++)
        if(nvals[k]) bins[k] /= (REAL) nvals[k];
      break;
    case 1: // direct sum
      break;
    default:
      fprintf(stderr, "libgrid: illegal value for volel in spherial averaging.\n");
      exit(1);
  }
  free(nvals);
}

/*
 * @FUNC{rgrid_spherical_average_reciprocal, "Spherical average in reciprocal space"}
 * @DESC{"Compute spherical shell average in the reciprocal space of power spectrum 
          with respect to the grid origin (result 1-D grid). This can be done over three (or less)
          separate grids, This produces power spectrum, so the output is REAL.\\
          Notes: To compute $E(k)$, grid should correspond to $flux / \sqrt(\rho) = \sqrt(\rho) v$"}
 * @ARG1{rgrid *input1, "Input grid 1 for averaging, but this complex data (i.e., *after* FFT)"}
 * @ARG2{rgrid *input2, "Input grid 2 for averaging, but this complex data (i.e., *after* FFT). Can be NULL if N/A"}
 * @ARG3{rgrid *input3, "Input grid 3 for averaging, but this complex data (i.e., *after* FFT). Can be NULL if N/A"}
 * @ARG4{REAL *bins, "1-D array for the averaged values. This is an array with dimension equal to nbins"}
 * @ARG5{REAL binstep, "Binning step length for k"}
 * @ARG6{INT nbins, "Number of bins requested"}
 * @ARG7{char volel, "2: direct sum, 1: Include the volume element, or 0: just calculate radial average"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_spherical_average_reciprocal(rgrid *input1, rgrid *input2, rgrid *input3, REAL *bins, REAL binstep, INT nbins, char volel) {

  INT nx = input1->nx, ny = input1->ny, nz = input1->nz2 / 2, idx, nxy = nx * ny;
  REAL step = input1->step, r, kx, ky, kz, nrm = input1->fft_norm;
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
    fprintf(stderr, "libgrid: Out of memory in rgrid_spherical_average_reciprocal().\n");
    abort();
  }
  bzero(nvals, sizeof(INT) * (size_t) nbins);
  bzero(bins, sizeof(REAL) * (size_t) nbins);

// TODO: Can't execute in parallel (reduction for bins[idx] needed
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
      idx = (INT) (0.5 + r / binstep);
      if(idx < nbins) {
        bins[idx] += 2.0 * CREAL(value1[ijnz + k]) * CREAL(value1[ijnz + k]) + CIMAG(value1[ijnz + k]) * CIMAG(value1[ijnz + k]);  // 2x = conjugate side from z
        if(value2) bins[idx] += 2.0 * CREAL(value2[ijnz + k]) * CREAL(value2[ijnz + k]) + CIMAG(value2[ijnz + k]) * CIMAG(value2[ijnz + k]);
        if(value3) bins[idx] += 2.0 * CREAL(value3[ijnz + k]) * CREAL(value3[ijnz + k]) + CIMAG(value3[ijnz + k]) * CIMAG(value3[ijnz + k]);
        nvals[idx] += 2;
      }
    }
  }
  switch(volel) {
    case 0: // radial average
      for(k = 0; k < nbins; k++)
        if(nvals[k]) bins[k] *= nrm / (REAL) nvals[k];
      break;
    case 1: // direct sum
      for(k = 0; k < nbins; k++)
        bins[k] *= nrm;
      break;
    default:
      fprintf(stderr, "libgrid: illegal value for volel in spherial averaging.\n");
      exit(1);
  }
  free(nvals);
}

/*
 * @FUNC{rgrid_npoint_smooth, "Running average of grid"}
 * @DESC{"Calculate running average to smooth unwanted high frequency components.
          Note that the destination and source grids cannot be the same"}
 * @ARG1{rgrid *dest, "Destination grid"}
 * @ARG2{rgrid *source, "Source grid"}
 * @ARG3{INT npts, "Number of points used in running average"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_fft_filter, "Apply user defined filter in reciprocal space"}
 * @DESC{"Apply user defined filter in reciproca space.
          The user filter function takes four arguments:
          user data (void *) and kx,ky,kz coordinates (REAL)"}
 * @ARG1{rgrid *grid, "Grid in reciprocal space to be filtered"}
 * @ARG2{REAL complex *(func), "Filter function"}
 * @ARG3{void *farg, "Optional arguments to be passed to the function (can be NULL)"}
 * @RVAL{void, "No return value"}
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
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = M_PI / (step * (REAL) (nz - 1));

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
 * @FUNC{rgrid_host_lock, "Lock real grid in host memory"}
 * @DESC{"Lock grid into host memory. This does nothing on pure CPU-based systems
          but on GPU-based systems it forces a given grid to stay in the host memory"}
 * @ARG1{rgrid *grid, "Grid to be host-locked"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_host_lock(rgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 1;
#endif
}

/*
 * @FUNC{rgrid_host_unlock, "Unlock real grid from host memory"}
 * @DESC{"Unlock grid. This does nothing on pure CPU-based systems
         but on GPU-based systems it allows again the grid to move to GPU"}
 * @ARG1{rgrid *grid, "Grid to be host-unlocked"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_host_unlock(rgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 0;
#endif
}

/*
 * @FUNC{rgrid_fft_space, "Set real/reciprocal space flag for grid"}
 * @DESC{"Set space flag for grid. On CPU systems this does nothing.
          On GPU systems it affects the data storage order (INPLACE vs. INPLACE_SHUFFLED).
          Since the real and complex (R2C and C2R) storage formats are already different
          on CPU systems, this routine probably does not need to be called. If there is
          a complaint that the data is in wrong space (real vs. reciprocal) then there is
          likely something wrong with the program"}
 * @ARG1{rgrid *grid, "Grid for the operation"}
 * @ARG2{char flag, "0: Real data or 1: reciprocal space data"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_fft_space(rgrid *grid, char space) {

#ifdef USE_CUDA
  gpu_mem_block *ptr;

  if(!(ptr = cuda_find_block(grid->value))) return; // Not on GPU
  if(space) ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  else ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
#endif
}

/*
 * @FUNC{rgrid_multiply_by_x, "Multiply grid by coordinate x"}
 * @DESC{"Multiply real grid by coordinate x"}
 * @ARG1{rgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_multiply_by_y, "Multiply grid by coordinate y"}
 * @DESC{"Multiply real grid by coordinate y"}
 * @ARG1{rgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_multiply_by_z, "Multiply grid by coordinate z"}
 * @DESC{"Multiply real grid by coordinate z"}
 * @ARG1{rgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{rgrid_log, "Natural logarithm of real grid"}
 * @DESC{"Natural logarithm of absolute value of grid. Note that the source and destination may be
          the same grid"}
 * @ARG1{rgrid *dst, "Destination grid"}
 * @ARG2{rgrid *src, "Source grid"}
 * @ARG3{REAL eps, "Small number to add in order to avoid zero"}
 * @RVAL{void, "No return value"}
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

/*
 * @FUNC{rgrid_histogram, "Make histogram of grid"}x
 * @DESC{"Make histogram of the values in grid (from 0 to nbins * step)"}
 * @ARG1{rgrid *grid, "Grid of NON-NEGATIVE numbers"}
 * @ARG2{REAL *bins, "Historgram bins (length nbins)"}
 * @ARG3{INT nbins, "Number of bins"}
 * @ARG4{REAL step, "Bin step"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void rgrid_histogram(rgrid *grid, REAL *bins, INT nbins, REAL step) {

  REAL tmp;
  INT i, j, k, idx;

#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 1);
#endif
 
  bzero(bins, sizeof(REAL) * (size_t) nbins); 
  for(i = 0; i < grid->nx; i++) {
    for(j = 0; j < grid->ny; j++) {
      for(k = 0; k < grid->nz; k++) {
        tmp = rgrid_value_at_index(grid, i, j, k);
        if(tmp < 0.0) {
          fprintf(stderr, "libgrid: Negative values in rgrid_historgram().\n");
          abort();
        }
        idx = (INT) (tmp / step);
        if(idx < nbins) bins[idx] += 1.0;
      }
    }
  }
}

