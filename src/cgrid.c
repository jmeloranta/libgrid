 /*
 * Routines for complex grids.
 *
 * NX is major index and NZ is minor index (varies most rapidly).
 *
 * For 2-D grids use: (1, NY, NZ)
 * For 1-D grids use: (1, 1, NZ)
 *
 * Last reviewed: 26 Sep 2019.
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
 * @FUNC{cgrid_alloc, "Allocate complex grid"}
 * @DESC{"This function allocates a complex grid"}
 * @ARG1{INT nx, "Number of points on the grid along x"}
 * @ARG2{INT ny, "Number of points on the grid along y"}
 * @ARG3{INT nz, "Number of points on the grid along z"}
 * @ARG4{REAL step, "Spatial step length on the grid"}
 * @ARG5{value_outside, "Condition for accessing boundary points: CGRID_DIRICHLET_BOUNDARY, CGRID_NEUMANN_BOUNDARY, CGRID_PERIODIC_BOUNDARY, or user supplied function"}
 * @ARG6{void *outside_params_ptr, "Pointer for passing parameters for the given boundary access function. Use 0 to with the predefined boundary functions"}
 * @ARG7{char *id, "String ID describing the grid (for debugging)"}
 * @RVAL{cgrid *, "Pointer to the allocated grid. Returns NULL on error."}
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
  
#if defined(SINGLE_PREC)
  if (!(grid->value = (REAL complex *) fftwf_malloc(len))) {
#elif defined(DOUBLE_PREC)
  if (!(grid->value = (REAL complex *) fftw_malloc(len))) {
#elif defined(QUAD_PREC)
  if (!(grid->value = (REAL complex *) fftwl_malloc(len))) {
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

  /* FFT norm */
  grid->fft_norm = 1.0 / (REAL) (grid->nx * grid->ny * grid->nz);

  /* FFT integral norm */
  grid->fft_norm2 = grid->fft_norm;
  if(grid->nx > 1) grid->fft_norm2 *= grid->step;
  if(grid->ny > 1) grid->fft_norm2 *= grid->step;
  if(grid->nz > 1) grid->fft_norm2 *= grid->step;

  strncpy(grid->id, id, 32);
  grid->id[31] = 0;
  
  if (value_outside)
    grid->value_outside = value_outside;
  else
    grid->value_outside = cgrid_value_outside_dirichlet;
    
  if (outside_params_ptr)
    grid->outside_params_ptr = outside_params_ptr;
  else {
    grid->default_outside_params = 0.0;
    grid->outside_params_ptr = &grid->default_outside_params;
  }

  /* Forward and inverse plans - not allocated yet */  
  grid->plan = grid->iplan = NULL;

#ifdef USE_CUDA
  if(cuda_status()) {
    if((nx & (nx-1)) || (ny & (ny-1)) || (nz & (nz-1))) {
      fprintf(stderr, "libgrid(cuda): Grid dimensions must be powers of two (performance & reduction).\n");
      abort();
    }
    cgrid_cufft_alloc(grid); // We have to allocate these for cuda.c to work
  }
#endif
  
#ifdef USE_CUDA
  /* Allocate CUDA memory for reduction */
  cgrid_cuda_init(sizeof(REAL complex) 
     * ((((size_t) nx) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK)
     * ((((size_t) ny) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK) 
     * ((((size_t) nz) + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK));
#endif
  
  /* Initialize the grid values to zero */
  for(i = 0; i < nx * ny * nz; i++)
    grid->value[i] = 0.0;

  /* Mark the grid as unclaimed (not in use exclusively) */
  grid->flag = 0;

#ifdef USE_CUDA
  /* By default the grid is not locked into host memory */
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
 * @FUNC{cgrid_clone, "Clone a grid"}
 * @DESC{"Clone a complex grid with the parameters identical to the given grid (except new grid->value is allocated and new id is used)"}
 * @ARG1{cgrid *grid, "Grid to be cloned"}
 * @ARG2{char *id, "ID string describing the new grid"}
 * @RVAL{cgrid *, "Pointer to the new grid"}
 *
 */

EXPORT cgrid *cgrid_clone(cgrid *grid, char *id) {

  cgrid *ngrid;

  if(!(ngrid = (cgrid *) malloc(sizeof(cgrid)))) {
    fprintf(stderr, "libgrid: Out of memory in cgrid_clone().\n");
    abort();
  }

  bcopy((void *) grid, (void *) ngrid, sizeof(cgrid));
  strcpy(ngrid->id, id);

#if defined(SINGLE_PREC)
  if (!(ngrid->value = (REAL complex *) fftwf_malloc(ngrid->grid_len))) {
#elif defined(DOUBLE_PREC)
  if (!(ngrid->value = (REAL complex *) fftw_malloc(ngrid->grid_len))) {
#elif defined(QUAD_PREC)
  if (!(ngrid->value = (REAL complex *) fftwl_malloc(ngrid->grid_len))) {
#endif
    fprintf(stderr, "libgrid: Error in cgrid_clone(). Could not allocate memory for grid.\n");
    free(ngrid);
    abort();
  }

#ifdef USE_CUDA
  /* Make CUFFT plan */
  if(cuda_status()) cgrid_cufft_alloc(ngrid);
#endif

  /* No need to do FFTW plans yet */
  ngrid->plan = ngrid->iplan = NULL;

#ifdef USE_CUDA
  /* Clear host lock */
  ngrid->host_lock = 0;
#endif

  /* Mark as not claimed */
  ngrid->flag = 0;

  return ngrid;
}

/*
 * @FUNC{cgrid_claim, "Claim grid"}
 * @DESC{"Claim grid (simple locking system when using the workspace model)"}
 * @ARG1{cgrid *input, "Grid to be claimed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_claim(cgrid *grid) {

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
 * @FUNC{cgrid_release, "Release grid"}
 * @DESC{"Release grid (simple locking system for the workspace model)"}
 * @ARG1{cgrid *grid, "Grid to be claimed"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_set_origin, "Set grid origin"}
 * @DESC{"Set the origin of coordinates. The coordinates of the grid will be:\\
         x(i) = (i - nx/2) * step - x0,\\ 
         y(j) = (j - ny/2) * step - y0,\\ 
         z(k) = (k - nz/2) * step - z0"}
 * @ARG1{cgrid *grid, "Grid whose origin will be set"}
 * @ARG2{REAL x0, "X origin"}
 * @ARG3{REAL y0, "Y origin"}
 * @ARG4{REAL z0, "Z origin"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_set_origin(cgrid *grid, REAL x0, REAL y0, REAL z0) {

  grid->x0 = x0;
  grid->y0 = y0;
  grid->z0 = z0;
}

/*
 * @FUNC{cgrid_shift_origin, "Shift grid origin"}
 * @DESC{"Shift the origin to (x0, y0, z0)"}
 * @ARG1{cgrid *grid, "Grid whose origin is to be shifted"}
 * @ARG2{REAL x0, "X shift"}
 * @ARG3{REAL y0, "Y shift"}
 * @ARG4{REAL z0, "Z shift"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_shift_origin(cgrid *grid , REAL x0, REAL y0, REAL z0) {

  grid->x0 += x0;
  grid->y0 += y0;
  grid->z0 += z0;
}

/*
 * @FUNC{cgrid_set_momentum, "Set momentum origin of grid"}
 * @DESC{"Set the origin in the momentum space (or the moving frame of reference). Note that kx0, ky0 and kz0
         should be multiples of:\\
         kx0min = 2.0 * M_PI / (NX * STEP)\\
         ky0min = 2.0 * M_PI / (NY * STEP)\\
         kz0min = 2.0 * M_PI / (NZ * STEP)"}
 * @ARG1{REAL kx0, "Momentum origin along x"}
 * @ARG2{REAL ky0, "Momentum origin along y"}
 * @ARG3{REAL kz0, "Momentum origin along z"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_set_momentum(cgrid *grid, REAL kx0, REAL ky0, REAL kz0) {

  grid->kx0 = kx0;
  grid->ky0 = ky0;
  grid->kz0 = kz0;
}

/*
 * @FUNC{cgrid_set_rotation, "Set angular momentum constaint for grid"}
 * @DESC{"Set angular rotation constraing about the Z-axis."}
 * @ARG1{cgrid *grid, "Grid for operation"}
 * @ARG2{REAL omega, "Rotation frequency about the Z axis"}
 * RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_set_rotation(cgrid *grid, REAL omega) {

  grid->omega = omega;
}

/*
 * @FUNC{cgrid_free, "Free complex grid"}
 * @DESC{"Free complex grid."}
 * @ARG1{cgrid *grid, "Pointer to grid to be freed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_free(cgrid *grid) {

  if (grid) {
#ifdef USE_CUDA
    cuda_remove_block(grid->value, 0);
    if(grid->cufft_handle != -1) cufftDestroy(grid->cufft_handle);
#endif
#if defined(SINGLE_PREC)
    if (grid->value) fftwf_free(grid->value);
#elif defined(DOUBLE_PREC)
    if (grid->value) fftw_free(grid->value);
#elif defined(QUAD_PREC)
    if (grid->value) fftwl_free(grid->value);
#endif
    cgrid_fftw_free(grid);
    free(grid);
  }
}

/* 
 * @FUNC{cgrid_write, "Write grid to disk"}
 * @DESC{"Write grid to disk in binary format."}
 * @ARG1{cgrid *grid, "Grid to be written"}
 * @ARG2{FILE *out, "File handle for the file (FILE * as defined in stdio.h)."}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_read, "Read complex grid from disk"}
 * @DESC{"Read complex grid from disk in binary format. 
          If the allocated grid is different size than the one on disk,
          this will automatically interpolate the data"}
 * @ARG1{cgrid *grid, "Grid to be read"}
 * @ARG2{FILE *in, "File handle for reading the file (FILE * as defined in stdio.h)"}
 * @RVAL{cgrid *, "Returns pointer to the grid or NULL on error"}
 *
 */

EXPORT cgrid *cgrid_read(cgrid *grid, FILE *in) {

  INT nx, ny, nz;
  REAL step;
  
#ifdef USE_CUDA
  if(grid) cuda_remove_block(grid->value, 0);
#endif
  fread(&nx, sizeof(INT), 1, in);
  fread(&ny, sizeof(INT), 1, in);
  fread(&nz, sizeof(INT), 1, in);
  fread(&step, sizeof(REAL), 1, in);
  
  if (!grid) {
    if(!(grid = cgrid_alloc(nx, ny, nz, step, CGRID_PERIODIC_BOUNDARY, NULL, "read_grid"))) {
      fprintf(stderr, "libgrid: Failed to allocate grid in cgrid_read().\n");
      return NULL;
    }
  }

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
    return grid;
  }
  
  fread(grid->value, sizeof(REAL complex), (size_t) (grid->nx * grid->ny * grid->nz), in);
  return grid;
}

/*
 * @FUNC{cgrid_write_grid, "Write grid to disk with cuts along x, y, z"}
 * @DESC{"Write complex grid to disk (.grd) including cuts along x, y, and z axes. The latter files are
          in ASCII format and have .x, .y and .z extensions"}
 * @ARG1{char *basename, "Base filename"}
 * @ARG2{cgrid *grid, "Grid to be written to disk"}
 * @RVAL{void, "No return value"}
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
    abort();
  }
  cgrid_write(grid, fp);
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
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", x, CREAL(tmp), CIMAG(tmp));
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
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", y, CREAL(tmp), CIMAG(tmp));
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
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", z, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);
}

/*
 * @FUNC{cgrid_write_grid_reciprocal, "Write complex reciprocal space grid to disk"}
 * @DESC{"Write complex (momentum space) grid to disk including cuts along x, y, and z axes.
         Suffixes .x, .y, .z, and .grd are added to the given file name"}
 * @ARG1{char *basename, "Base filename"}
 * @ARG2{cgrid *grid, "Grid to be written to disk"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_write_grid_reciprocal(char *base, cgrid *grid) {

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
    abort();
  }
  cgrid_write(grid, fp);
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
    tmp = cgrid_value_at_index(grid, i, j, k);
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
    if (j <= ny / 2)
      y = 2.0 * M_PI * ((REAL) j) / (((REAL) ny) * step) - grid->ky0;
    else 
      y = 2.0 * M_PI * ((REAL) (j - ny)) / (((REAL) ny) * step) - grid->ky0;
    tmp = cgrid_value_at_index(grid, i, j, k);
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
    if (k <= nz / 2)
      z = 2.0 * M_PI * ((REAL) k) / (((REAL) nz) * step) - grid->kz0;
    else 
      z = 2.0 * M_PI * ((REAL) (k - nz)) / (((REAL) nz) * step) - grid->kz0;
    tmp = cgrid_value_at_index(grid, i, j, k);
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", z, CREAL(tmp), CIMAG(tmp));
  }
  fclose(fp);
}

/*
 * @FUNC{cgrid_read_grid, "Read grid from disk"}
 * @DESC{"Read in a grid from a binary file (.grd)"}
 * @ARG1{cgrid *grid, "Grid where the data is placed"}
 * @ARG2{char *file, "File name to be read. Note that the .grd extension must be included"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_read_grid(cgrid *grid, char *file) {

  FILE *fp;

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libgrid: Can't open complex grid file %s.\n", file);
    abort();
  }
  cgrid_read(grid, fp);
  fclose(fp);
}

/*
 * @FUNC{cgrid_copy, "Copy grid"}
 * @DESC{"Copy grid to another grid"}
 * @ARG1{cgrid *dst, "Destination grid"}
 * @ARG2{cgrid *src, "Source grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_copy(cgrid *dst, cgrid *src) {

  INT i, nx = src->nx, nyz = src->ny * src->nz;
  size_t bytes = ((size_t) nyz) * sizeof(REAL complex);
  REAL complex *svalue = src->value;
  REAL complex *dvalue = dst->value;

  if(src->nx != dst->nx || src->ny != dst->ny || src->nz != dst->nz) {
    fprintf(stderr, "libgrid: Different grid dimensions in cgrid_copy.\n");
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
  if(cuda_status() && !cgrid_cuda_copy(dst, src)) return;
#endif
  
#pragma omp parallel for firstprivate(nx,nyz,bytes,svalue,dvalue) private(i) default(none) schedule(runtime)
  for(i = 0; i < nx; i++)
    bcopy(&svalue[i * nyz], &dvalue[i * nyz], bytes);
}

/*
 * @FUNC{cgrid_conjugate, "Complex conjugate of grid"}
 * @DESC{"Take complex conjugate of grid. Note that the source and destination may be the same"}
 * @ARG1{cgrid *conjugate, "Destination for the operation"}
 * @ARG2{cgrid *grid, "Source grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_shift, "Spatial shift of grid"}
 * @DESC{"Shift grid by given amount spatially"}
 * @ARG1{cgrid *shifted, "Destination grid for the operation"}
 * @ARG2{cgrid *grid, "Source grid for the operation"}
 * @ARG3{REAL x, "Shift spatially by this amount in x"}
 * @ARG4{REAL y, "Shift spatially by this amount in y"}
 * @ARG5{REAL z, "Shift spatially by this amount in z"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_zerp, "Zero grid"}
 * @DESC{"Zero grid values"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @RVAL{void, "No return value"}
 * 
 */

EXPORT void cgrid_zero(cgrid *grid) { 

  cgrid_constant(grid, 0.0); 
}

/* 
 * @FUNC{cgrid_constant, "Set grid value to constant"}
 * @DESC{"Set grid values to constant"}
 * @ARG1{cgrid *grid, "Grid to be set"}
 * @ARG2{REAL complex c, "Constant value"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_product_func, "Multiple grid by function"}
 * @DESC{"Multiply grid by a given function. The arguments to the user supplied function are:
          (void *arg) is for external user specified data (may be NULL), 
          (REAL complex val) the grid value (REAL complex),
          and (REAL) x, y, z are the coordinates where the function is evaluated"}
 * @ARG1{cgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL complex (*func), "Function providing the mapping."}
 * @ARG3{void *farg, "Pointer to user specified data"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_map, "Map function onto grid"}
 * @DESC{"Map a given function onto a grid. The arguments to the user specified function are:
          (void *farg) pointer to user specified data (may be NULL)
          and x, y, z are the current coordinates where the function is evaluated"}
 * @ARG1{cgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL complex (*func), "Function providing the mapping"}
 * @ARG3{void *, "Pointer to user specified data"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_smooth_map, "Map function onto grid with linear smoothing"}
 * @DESC{"Map a given function onto grid with linear smoothing.
         This can be used to weight the values at grid points to produce more
         accurate integration over the grid. The arguments to the user specified function are:
         (void *farg) pointer to user specified data (may be NULL)
         and x, y, z are the current coordinates where the function is evaluated"}
 * @ARG1{cgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL complex (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data"}
 * @ARG4{INT ns, "Number of intermediate points to be used in smoothing"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_adaptive_map, "Map function onto grid with adaptive smoothing"}
 * @DESC{"Map a given function onto grid with adaptive linear smoothing.
          This can be used to weight the values at grid points to produce more
          accurate integration over the grid. Limits for intermediate steps and
          tolerance can be given. The arguments to the user specified function are:
          (void *farg) pointer to user specified data (may be NULL)
          and x, y, z are the current coordinates where the function is evaluated"}
 * @ARG1{cgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL complex (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data"}
 * @ARG4{INT min_ns, "Minimum number of intermediate points to be used in smoothing"}
 * @ARG5{INT max_ns, "Maximum number of intermediate points to be used in smoothing"}
 * @ARG6{REAL tol, "Tolerance for weighing"}
 * @RVAL{void, "No return value"}
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
        if (csqnorm(sum - sump) < tol2) break;
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
 * @FUNC{cgrid_sum, "Add two grids"}
 * @DESC{"Add two grids: gridc = grida + gridb. 
          Note that the source and destination grids may be the same grid"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st grid to be added"}
 * @ARG3{cgrid *gridb, "2nd grid to be added"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_difference, "Subtract two grids"}
 * @DESC{"Subtract two grids: gridc = grida - gridb. 
         Note that both source and destination may be the same grid."}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * FUNC{cgrid_product, "Product of two grids"}
 * @DESC{Calculate product of two grids: gridc = grida * gridb. 
         Note that the source and destination grids may be the same."}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_abs_power, "Power of absolute value of grid"}
 * @DESC{"Rise absolute value of a grid to a given power.
          Note that the source and destination grids may be the same grid.
          This routine uses pow() so that the exponent can be fractional."}
 * @ARG1{cgrid *gridb, "Destination grid"}
 * @ARG2{cgrid *grida, "Source grid"}
 * @ARG3{REAL exponent, "Exponent to be used"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_power, "Rise grid to given power"}
 * @DESC{"Rise grid to given power. The exponent can be fractional as this uses pow().
          Note that the source and destination grids may be the same grid"}
 * @ARG1{cgrid *gridb, "Destination grid"}
 * @ARG2{cgrid *grida, "Source grid"}
 * @ARG3{REAL exponent, "Exponent to be used"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_division, "Divide two grids"}
 * @DESC{"Divide two grids: gridc = grida / gridb. Note that the source and destination grids may be the same."}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_division_eps, "Numerically safe division of two grids"}
 * @DESC{"Numerically stable division of two grids: gridc = grida / (gridb + eps). 
          Note that the source and destination grids may be the same"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @ARG4{REAL eps, "Epsilon to add to the divisor"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_conjugate_product, "Product of two grids with conjugation"}
 * @DESC{"Conjugate product of two grids: gridc = CONJ(grida) * gridb.
          Note that the source and destination grids may be the same"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1st source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_add, "Add constant to grid"}
 * @DESC{"Add a constant to a grid."}
 * @ARG1{cgrid *grid, "Grid where the constant is added"}
 * @ARG2{REAL complex c, "Constant to be added"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_multiply, "Multiply grid by constant"}
 * @DESC{"Multiply grid by a constant"}
 * @ARG1{cgrid *grid, "Grid to be multiplied"}
 * @ARG2{REAL complex c, "Multiplier"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_add_and_multiply, "Add and multiply grid"}
 * @DESC{"Add and multiply: grid = (grid + ca) * cm"}
 * @ARG1{cgrid *grid, "Grid to be operated on"}
 * @ARG2{REAL complex ca, "Constant to be added"}
 * @ARG3{REAL complex cm, "Multiplier"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_multiply_and_add, "Multiply and add grid"}
 * @DESC{"Multiply and add: grid = cm * grid + ca"}
 * @ARG1{cgrid *grid, "Grid to be operated on"}
 * @ARG2{REAL complex cm, "Multiplier"}
 * @ARG3{REAL complex ca, "Constant to be added"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_add_scaled, "Add scaled grid"}
 * @DESC{"Add scaled grid: gridc = gridc + d * grida. 
          Note that source and destination grids may be the same grid"}
 * @ARG1{cgrid *gridc, "Destination grid for the operation"}
 * @ARG2{REAL complex d, "Multiplier for the operation"}
 * @ARG3{cgrid *grida, "Source grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_add_scaled_product, "Three grid product"}
 * @DESC{"Perform the following operation: gridc = gridc + d * grida * gridb.
          Note that the source and destination grids may be the same grid"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{REAL complex d, "Constant multiplier"}
 * @ARG3{cgrid *grida, "1st source grid"}
 * @ARG4{cgrid *gridb, "2nd source grid"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_operate_one, "Operate on grid"}
 * @DESC{"Operate on a grid by a given operator: gridc = O(grida).
         Note that the source and destination grids may be the same grid.
         The arguments to the use specified operator function are:
         (REAL complex) value at grid point and (void *) params.
         The latter argument is for external user supplied data"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "Source grid"}
 * @ARG3{(REAL complex) (*operator), "Operator function"}
 * @ARG4{void *params, "User supplied additional parameters (may be NULL)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_operate_one_product, "Operate on grid and multiply"}
 * @DESC{"Operate on a grid by a given operator and multiply: gridc = gridb * O(grida).
         Note that the source and destination grids may be the same.
         The operator function takes the following arguments:
         (REAL complex) value at the grid point, (void *) params. The latter is for
         user specified additional data"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *gridb, "Multiplication by this grid"}
 * @ARG3{cgrid *grida, "Source grid"}
 * @ARG4{(REAL complex (*operator), "Operator function"}
 * @ARG5{void *params, "User supplied data (may be NULL)"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_operate_two, "Operate on two grids"}
 * @DESC{"Operate on two grids and place the result in third: gridc = O(grida, gridb)
          where O is the operator. Note that the source and destination grids may be the same.
          The operator function has two parameters, the current grid values of grida and gridb"}
 * @ARG1{cgrid *gridc, "Destination grid"}
 * @ARG2{cgrid *grida, "1s source grid"}
 * @ARG3{cgrid *gridb, "2nd source grid"}
 * @ARG4{(REAL complex (*operator), "Operator"}
 * @RVAL{void, "No return value"}
 *
 * TODO: Allow user specified parameter passing.
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
 * @FUNC{cgrid_transform_one, "Transform grid"}
 * @DESC{"Transform a grid by a given operator. This can modify the grid values directly through
         the pointer (REAL complex *) that is given as argument to the operator"}
 * @ARG1{cgrid *grid, "Grid to be operated"}
 * @ARG2{void (*operator), "Operator function"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_transform_two, "Transform two grids"}
 * @DESC{"Transform two separate grids by a given operator. This can modify the grid values directly through
         the two pointers (REAL complex *) that are given as arguments to the operator"}
 * @ARG1{cgrid *grida, "Grid 1 to be operated"}
 * @ARG2{cgrid *gridb, "Grid 2 to be operated"}
 * @ARG3{void (*operator), "Operator acting on the two grids"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_integral, "Integrate over grid"}
 * @DESC{"Integrate over a grid"}
 * @ARG1{cgrid *grid, "Grid to be integrated"}
 * @RVAL{REAL complex, "Numerical value for the integral"}
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
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * @FUNC{cgrid_integral_region, "Integrate over region"}
 * @DESC{"Integrate over a grid with limits"}
 * @ARG1{cgrid *grid, "Grid to be integrated"}
 * @ARG2{REAL xl, "Lower limit for x"}
 * @ARG3{REAL xu, "Upper limit for x"}
 * @ARG4{REAL yl, "Lower limit for y"}
 * @ARG5{REAL yu, "Upper limit for y"}
 * @ARG6{REAL zl, "Lower limit for z"}
 * @ARG7{REAL zu, "Upper limit for z"}
 * @RVAL{REAL complex, "Numerical value for the integral"}
 *
 */

EXPORT REAL complex cgrid_integral_region(cgrid *grid, REAL xl, REAL xu, REAL yl, REAL yu, REAL zl, REAL zu) {

  INT iu, il, i, ju, jl, j, ku, kl, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
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
  if(nz != 1) sum *= step;
  return sum;
}
 
/* 
 * @FUNC{cgrid_integral_of_square, "Integrate over square of grid"}
 * @DESC{"Integrate over the grid squared ($\int |grid|^2$)"}
 * @ARG1{cgrid *grid, "Grid to be integrated"}
 * @RVAL{REAL complex, "Value for the integral"}
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
	sum += csqnorm(cgrid_value_at_index(grid, i, j, k));
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * @FUNC{cgrid_integral_of_conjugate_product, "Overlap between two grids"}
 * @DESC{"Calculate overlap between two grids ($\int grida^*gridb$)"}
 * @ARG1{cgrid *grida, "1st grid (complex conjugated)"}
 * @ARG2{cgrid *gridb, "2nd grid (no complex conjugation)"}
 * @RVAL{REAL complex, "Returns the overlap integral"}
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
  if(nz != 1) sum *= step;
  return sum;
}

/*
 * @FUNC{cgrid_grid_expectation_value, "Expectation value of operator over grid"}
 * @DESC{"Calculate expectation value of opgrid over dgrid (i.e., $\int opgrid |dgrid|^2$)"}
 * @ARG1{cgrid *dgrid, "Grid yielding the probability"}
 * @ARG2{cgrid *opgrid, "Grid to be averaged"}
 * @RVAL{REAL complex, "Returns the expectation value integral"}
 *
 */

EXPORT REAL complex cgrid_grid_expectation_value(cgrid *dgrid, cgrid *opgrid) {

  INT i, j, k, nx = dgrid->nx, ny = dgrid->ny , nz = dgrid->nz;
  REAL complex sum = 0.0, step = dgrid->step;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_grid_expectation_value(dgrid, opgrid, &sum)) return sum;
#endif
  
#pragma omp parallel for firstprivate(nx,ny,nz,dgrid,opgrid) private(i,j,k) reduction(+:sum) default(none) schedule(runtime)
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	sum += csqnorm(cgrid_value_at_index(dgrid, i, j, k)) * cgrid_value_at_index(opgrid, i, j, k);
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;

  return sum;
}
 
/*
 * @FUNC{cgrid_grid_expectation_value_func, "Calculate expectation value of function over grid"}
 * @DESC{"Calculate expectation value of a function over a grid 
         (i.e., $\int grida^* func grida = \int func |grida|^2$).
         Arguments to the function are: arg (optional; may be NULL), (REAL complex) grid value at (x, y, z),
         and coordinates x, y, z."}
 * @ARG1{REAL complex (*func), "Function to be averaged over the grid"}
 * @ARG2{cgrid *grida, "Grid yielding the probability (i.e., $|grida|^2$)"}
 * @RVAL{REAL complex, "Returns the expectation value integral"}
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
	sum += csqnorm(tmp) * func(arg, tmp, x, y, z);
      }
    }
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;
  return sum;
}

/* 
 * @FUNC{cgrid_weighted_integral, "Integrate over grid with weight function"}
 * @DESC{"Integrate over the grid multiplied by weighting function (i.e., $\int grid w(x)$).
         The weight function takes three arguments (REAL): x, y, z coordinates"}
 * @ARG1{cgrid *grid, "Grid to be integrated over"}
 * @ARG2{REAL complex (*weight), "Function defining the weight"}
 * @ARG3{void *farg, "Argument to the weight function"}
 * @RVAL{REAL complex, "Returns the value of the integral"}
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
  if(nz != 1) sum *= step;
  return sum;
}

/* 
 * @FUNC{cgrid_weighted_integral_of_square, "Weighted integral of grid squared"}
 * @DESC{"Integrate over square of the grid multiplied by weighting function ($\int grid^2 w(x)$).
         The weighing function takes three arguments: x, y, z coordinates of the current grid point"}
 * @ARG1{cgrid *grid, "Grid to be integrated over"}
 * @ARG2{REAL complex (*weight), "Function defining the weight"}
 * @ARG3{void *farg, "Argument to the weight function (may be NULL)"}
 * @RVAL{REAL complex, "Returns the value of the integral"}
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
	sum += weight(farg, x, y, z) * csqnorm(cgrid_value_at_index(grid, i, j, k));
      }
    }
  }
 
  if(nx != 1) sum *= step;
  if(ny != 1) sum *= step;
  if(nz != 1) sum *= step;

  return sum;
}

/*
 * @FUNC{cgrid_fft, "FFT of grid"}
 * @DESC{"Perform Fast Fourier Transformation of a grid"}
 * @ARG1{cgrid *grid, "Grid to be Fourier transformed (without normalization)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft(cgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cufft_fft(grid)) return;
#endif

  cgrid_fftw(grid);
}

/*
 * @FUNC{cgrid_inverse_fft, "Inverse FFT of grid"}
 * @DESC{"Perform inverse Fast Fourier Transformation of a grid (unnormalized)"}
 * @ARG1{cgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_inverse_fft(cgrid *grid) {

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cufft_fft_inv(grid)) return;
#endif

  cgrid_fftw_inv(grid);
}

/*
 * @FUNC{cgrid_scaled_inverse_fft, "Scaled inverse FFT of grid"}
 * @DESC{"Perform scaled inverse Fast Fourier Transformation of a grid"}
 * @ARG1{cgrid *grid, "Grid to be inverse Fourier transformed"}
 * @ARG2{REAL complex c, "Scaling factor"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_scaled_inverse_fft(cgrid *grid, REAL complex c) {

  cgrid_inverse_fft(grid);
  cgrid_multiply(grid, c);  
}

/*
 * @FUNC{cgrid_inverse_fft_norm, "Normalized inverse FFT of grid"}
 * @DESC{"Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm"}
 * @ARG1{cgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 * Note: The input grid is overwritten with the output.
 *
 */

EXPORT void cgrid_inverse_fft_norm(cgrid *grid) {

  cgrid_scaled_inverse_fft(grid, grid->fft_norm);
}

/*
 * @FUNC{cgrid_inverse_fft_norm2, "Spatially normalized inverse FFT of grid"}
 * @DESC{"Perform inverse Fast Fourier Transformation of a grid scaled by FFT norm
          (including the spatial step)"}
 * @ARG1{cgrid *grid, "Grid to be inverse Fourier transformed"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_inverse_fft_norm2(cgrid *grid) {

  cgrid_scaled_inverse_fft(grid, grid->fft_norm2);
}


/*
 * @FUNC{cgrid_fft_convolute, "Convolute two grids in reciprocal space"}
 * @DESC{"Convolute two FFT transformed grids. To convolute grids grida and gridb 
          and place the result in gridc:\\
          cgrid_fft(grida);\\
          cgrid_fft(gridb);\\
          cgrid_convolue(gridc, grida, gridb);\\
          cgrid_inverse_fft_norm2(gridc);\\
          gridc now contains the convolution of grida and gridb.
          The input and output grids may be the same. To conver from FFT to Fourier integral:\\
          Forward: Multiply result by step$^3$\\
          Inverse: Multiply result by 1 / (step * N)$^3$"}
 * @ARG1{cgrid *gridc, "Output grid"}
 * @ARG2{cgrid *grida, "1st grid to be convoluted"}
 * @ARG3{cgrid *gridb, "2nd grid to be convoluted"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_convolute(cgrid *gridc, cgrid *grida, cgrid *gridb) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy;
  REAL complex *cvalue, *bvalue, *avalue;

#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_fft_convolute(gridc, grida, gridb)) return;
#endif

  /* int f(r) g(r-r') d^3r' = iF[ F[f] F[g] ] = (step / N)^3 iFFT[ FFT[f] FFT[g] ] */
  nx = gridc->nx;
  ny = gridc->ny;
  nz = gridc->nz;
  nxy = nx * ny;
  
  cvalue = gridc->value;
  bvalue = gridb->value;
  avalue = grida->value; 
 
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,cvalue,bvalue,avalue) private(i,j,ij,ijnz,k) default(none) schedule(runtime)
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
 * @FUNC{cgrid_value_at_index, "Access grid point at given index"}
 * @DESC{"Access grid point at given index. Note that this is very slow on CUDA
          as it accesses the elements individually"}
 * @ARG1{cgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @RVAL{REAL complex, "Returns grid value at index (i, j, k)"}
 *
 * TODO: There is too much CUDA stuff in here, needs to move to cuda.c eventually.
 *
 */

EXPORT inline REAL complex cgrid_value_at_index(cgrid *grid, INT i, INT j, INT k) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz)
    return grid->value_outside(grid, i, j, k);

#ifdef USE_CUDA
  REAL complex value;
  gpu_mem_block *ptr;
  INT nx = grid->nx, ny = grid->ny, ngpu2 = cuda_ngpus(), ngpu1, nnx2, nnx1, nny2, nny1, gpu, idx;
  if((ptr = cuda_find_block(grid->value))) {
    if(ptr->gpu_info->subFormat == CUFFT_XT_FORMAT_INPLACE) { // Real space
      ngpu1 = nx % ngpu2;
      nnx2 = nx / ngpu2;
      nnx1 = nnx2 + 1;
      gpu = i / nnx1;
      if(gpu >= ngpu1) {
        idx = i - ngpu1 * nnx1;
        gpu = idx / nnx2 + ngpu1;
        idx = idx % nnx2;
      } else idx = i % nnx1;
      cuda_get_element(grid->value, (int) gpu, (size_t) ((idx * grid->ny + j) * grid->nz + k), sizeof(REAL complex), (void *) &value);
    } else {  // Reciprocal space
      ngpu1 = ny % ngpu2;
      nny2 = ny / ngpu2;
      nny1 = nny2 + 1;
      gpu = j / nny1;
      if(gpu >= ngpu1) {
        idx = j - ngpu1 * nny1;
        gpu = idx / nny2 + ngpu1;
        idx = idx % nny2;
        cuda_get_element(grid->value, (int) gpu, (size_t) ((i * nny2 + idx) * grid->nz + k), sizeof(REAL complex), (void *) &value);
      } else {
        idx = j % nny1;
        cuda_get_element(grid->value, (int) gpu, (size_t) ((i * nny1 + idx) * grid->nz + k), sizeof(REAL complex), (void *) &value);
      }
    }
    return value;
  } else
#endif
  return grid->value[(i * grid->ny + j) * grid->nz + k];
}

/*
 * @FUNC{cgrid_value_to_index, "Set value to grid at given index"}
 * @DESC{"Set value to a grid point at given index.
          Note that this is very slow on CUDA as each element is transferred individually"}
 * @ARG1{cgrid *grid, "Grid to be accessed"}
 * @ARG2{INT i, "Index along x"}
 * @ARG3{INT j, "Index along y"}
 * @ARG4{INT k, "Index along z"}
 * @RVAL{void, "No return value"}
 *
 * TODO: There is too much CUDA stuff in here, needs to move to cuda.c eventually
 *
 */

EXPORT inline void cgrid_value_to_index(cgrid *grid, INT i, INT j, INT k, REAL complex value) {

  if (i < 0 || j < 0 || k < 0 || i >= grid->nx || j >= grid->ny || k >= grid->nz) return;

#ifdef USE_CUDA
  gpu_mem_block *ptr;
  INT nx = grid->nx, ny = grid->ny, ngpu2 = cuda_ngpus(), ngpu1, nnx2, nnx1, nny2, nny1, gpu, idx;
  if((ptr = cuda_find_block(grid->value))) {
    if(ptr->gpu_info->subFormat == CUFFT_XT_FORMAT_INPLACE) { // Real space
      ngpu1 = nx % ngpu2;
      nnx2 = nx / ngpu2;
      nnx1 = nnx2 + 1;
      gpu = i / nnx1;
      if(gpu >= ngpu1) {
        idx = i - ngpu1 * nnx1;
        gpu = idx / nnx2 + ngpu1;
        idx = idx % nnx2;
      } else idx = i % nnx1;
      cuda_set_element(grid->value, (int) gpu, (size_t) ((idx * grid->ny + j) * grid->nz + k), sizeof(REAL complex), (void *) &value);
    } else {  // Reciprocal space (shuffled)
      ngpu1 = ny % ngpu2;
      nny2 = ny / ngpu2;
      nny1 = nny2 + 1;
      gpu = j / nny1;
      if(gpu >= ngpu1) {
        idx = j - ngpu1 * nny1;
        gpu = idx / nny2 + ngpu1;
        idx = idx % nny2;
        cuda_set_element(grid->value, (int) gpu, (size_t) ((i * nny2 + idx) * grid->nz + k), sizeof(REAL complex), (void *) &value);
      } else {
        idx = j % nny1;
        cuda_set_element(grid->value, (int) gpu, (size_t) ((i * nny1 + idx) * grid->nz + k), sizeof(REAL complex), (void *) &value);
      }
    }
    return;
  } else
#endif
   grid->value[(i * grid->ny + j) * grid->nz + k] = value;
}

/*
 * @FUNC{cgrid_value, "Access grid point at given coorinates"}
 * @DESC{"Access grid point at given (x,y,z) point using linear interpolation"}
 * @ARG1{cgrid *grid, "Grid to be accessed"}
 * @ARG2{REAL x, "x coordinate"}
 * @ARG3{REAL y, "y coordinate"}
 * @ARG4{REAL z, "z coordinate"}
 * @RVAL{REAL complex, "Returns grid value at (x,y,z)"}
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
 * @FUNC{cgrid_extrapolate, "Extrapolate between two grids"}
 * @DESC{"Extrapolate between two grids of different sizes"}
 * @ARG1{cgrid *dest, "Destination grid"}
 * @ARG2{cgrid *src, "Source grid"}
 * @RVAL{void, "No return value"}
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
 * TODO: This should probably move to cprivate.c ...
 *
 */

static REAL complex cgrid_value_rotate_z(void *arg, REAL x, REAL y, REAL z) {

  /* Unpack the values in arg */ 
  cgrid *grid = ((grid_rotation *) arg)->cgrid;
  REAL sth = ((grid_rotation *) arg)->sinth, cth = ((grid_rotation *) arg)->costh, xp, yp;

  xp = -y * sth + x * cth;
  yp =  y * cth + x * sth;

  return cgrid_value(grid, xp, yp, z);
}

/*
 * @FUNC{cgrid_rotate_z, "Rotate grid around z-axis"}
 * @DESC{"Rotate a grid by a given angle around the z-axis"}
 * @ARG1{cgrid *in, "Input grid"}
 * @ARG2{cgrid *out, "Rotated grid"}
 * @ARG3{REAL th, "Rotation angle theta in radians"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_rotate_z(cgrid *out, cgrid *in, REAL th) {

  grid_rotation *r;

  if(in == out) {
    fprintf(stderr,"libgrid: in and out grids in cgrid_rotate_z must be different\n");
    abort();
  }
  if(!(r = malloc(sizeof(grid_rotation)))) {
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
 * @FUNC{cgrid_zero_re, "Clear real part of grid"}
 * @DESC{"Clear real part of complex grid"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_zero_im, "Zero imaginary part of grid"}
 * @DESC{"Clear imaginary part of complex grid"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{cgrid_phase, "Calculate phase factor of grid"}
 * @DESC{"Extract complex phase factors from a given grid"}
 * @ARG1{cgrid *dst, "Destination grid"}
 * @ARG2{cgrid *src, "Source grid"}
 * @RVAL{void, "No return value"}
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
    abort();
  }

#pragma omp parallel for firstprivate(src,dst) private(i) default(none) schedule(runtime)
  for (i = 0; i < dst->nx * dst->ny * dst->nz; i++) {
    if(CABS(src->value[i]) < GRID_EPS) dst->value[i] = 0.0;
    else dst->value[i] = CREAL(-I * CLOG(src->value[i] / CABS(src->value[i])));
    if(dst->value[i] < 0.0) dst->value[i] = 2.0 * M_PI + dst->value[i];
  }
}

/*
 * @FUNC{cgrid_zero_index, "Zero part of grid"}
 * @DESC{"Zero a given index range of a complex grid [lx, hx[ x [ly, hy[ x [lz, hz["}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @ARG2{INT lx, "Lower limit for x index"}
 * @ARG3{INT hx, "Upper limit for x index"}
 * @ARG4{INT ly, "Lower limit for y index"}
 * @ARG5{INT hy, "Upper limit for y index"}
 * @ARG6{INT lz, "Lower limit for z index"}
 * @ARG7{INT hz, "Upper limit for z index"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_zero_index(cgrid *grid, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nynz = ny * nz;
  REAL complex *value = grid->value;

  if(hx > nx) hx = nx;
  if(hy > ny) hy = ny;
  if(hz > nz) hz = nz;
  if(lx < 0) lx = 0;
  if(ly < 0) ly = 0;
  if(lz < 0) lz = 0;

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
 * @FUNC{cgrid_ipower, "Rise grid to integer power"}
 * @DESC{"Raise grid to integer power (fast): $|grid|^n$. Note that source and destination can be the
          same grid"}
 * @ARG1{cgrid *dst, "Destination grid"}
 * @ARG2{cgrid *src, "Source grid"}
 * @ARG3{INT exponent, "Exponent, which may benegative"}
 * @RVAL{void, "No return value"}
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

/*
 * @FUNC{cgrid_fft_filter, "Apply FFT filter to grid"}
 * @DESC{"Apply user defined filter in Fourier space.
          The filter function takes the following arguments: user data (void *farg), 
          kx (REAL), ky (REAL), and kz (REAL)"}
 * @ARG1{cgrid *grid, "Grid in reciprocal space to be filtered"}
 * @ARG2{REAL complex (*func), "Filter function"}
 * @ARG3{void *farg, "User data passed to the filter function (may be NULL)"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_filter(cgrid *grid, REAL complex (*func)(void *, REAL, REAL, REAL), void *farg) {

  INT i, j, k, ij, ijnz, nx, ny, nz, nxy, nx2, ny2, nz2;
  REAL kx, ky, kz, lx, ly, lz, step;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  cuda_remove_block(grid->value, 1);
#endif

  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  nxy = nx * ny;
  step = grid->step;
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);

  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
#pragma omp parallel for firstprivate(nx2,ny2,nz2,func,farg,nx,ny,nz,nxy,step,value,lx,ly,lz) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
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
      kx =((REAL) i) * lx;
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
       
      value[ijnz + k] *= (*func)(farg, kx, ky, kz);
    }
  }
}

/*
 * @FUNC{cgrid_host_lock, "Lock grid to host memory (CUDA)"}
 * @DESC{"Lock grid into host memory. This does nothing on pure CPU-based systems.
          On GPU-based systems it forces a given grid to stay in host memory"}
 * @ARG1{cgrid *grid, "Grid to be host-locked"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_host_lock(cgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 1;
#endif
}

/*
 * @FUNC{cgrid_host_unlock, "Unlock grid from host memory (CUDA)"}
 * @DESC{"Unlock grid from host memory. This does nothing on pure CPU-based systems.
          On GPU-based systems it allows again the grid to move to GPU."}
 * @ARG1{cgrid *grid, "Grid to be host-unlocked"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_host_unlock(cgrid *grid) {

#ifdef USE_CUDA
  grid->host_lock = 0;
#endif
}

/*
 * @FUNC{cgrid_fft_space, "Set space flag on grid (CUDA)"}
 * @DESC{"Set space flag for grid. On CPU systems this does nothing.
          On GPU systems it affects the data storage order (INPLACE vs. INPLACE_SHUFFLED).
          In C2C transform on CPU there is no difference in the storage format. However,
          on GPU forward and inverse transforms store things differently across GPUs.
          This routine may have to be called if a grid is taken to Fourier space and
          then it is operated afterwards for real space. For example:\\
          cgrid_fft(grid1);\\
          cgrid_fft(grid2);\\
          cgrid_fft_convolute(grid3, grid2, grid1);\\
          cgrid_inverse_fft(grid3);\\
          ....\\
          $<$both grid1 and grid2 are left in INPLACE_SHUFFLED format$>$\\
          To use them in real space at this point, this routine must be used"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @ARG2{char flag, "0: Real data or 1: reciprocal space data"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_fft_space(cgrid *grid, char space) {

#ifdef USE_CUDA
  gpu_mem_block *ptr;

  if(!(ptr = cuda_block_address(grid->value))) return; // Not on GPU
  if(space) ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
  else ptr->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
#endif
}

/*
 * @FUNC{cgrid_multiply_by_x, "Multiply grid by coordinate x"}
 * @DESC{"Multiply complex grid by coordinate x"}
 * @ARG1{cgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_multiply_by_x(cgrid *grid) {

  INT i, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nx2 = nx / 2;
  REAL x, step = grid->step;
  REAL x0 = grid->x0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_multiply_by_x(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nx2,nxy,step,value,x0) private(i,ij,ijnz,k,x) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    x = ((REAL) (i - nx2)) * step - x0;
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= x;
  }
}

/*
 * @FUNC{cgrid_multiply_by_y, "Multiply grid by coordinate y"}
 * @DESC{"Multiply complex grid by coordinate y"}
 * @ARG1{cgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_multiply_by_y(cgrid *grid) {

  INT j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, ny2 = ny / 2;
  REAL y, step = grid->step;
  REAL y0 = grid->y0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_multiply_by_y(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,ny2,nxy,step,value,y0) private(j,ij,ijnz,k,y) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    j = ij % ny;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++)
      value[ijnz + k] *= y;
  }
}

/*
 * @FUNC{cgrid_multiply_by_z, "Multiply grid by coordinate z"}
 * @DESC{"Multiply complex grid by coordinate z"}
 * @ARG1{cgrid *grid, "Grid to be operated on"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_multiply_by_z(cgrid *grid) {

  INT k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nz2 = nz / 2;
  REAL z, step = grid->step;
  REAL z0 = grid->z0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_multiply_by_z(grid)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nz2,nxy,step,value,z0) private(ij,ijnz,k,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      value[ijnz + k] *= z;
    }
  }
}

/*
 * @FUNC{cgrid_spherical_average, "Spherical average of grid squared"}
 * @DESC{"Compute spherical shell average of $|grid|^2$ with respect to the grid origin
         (result 1-D grid). Three grids can be averaged together but they can be given as NULL
         if not needed"}
 * @ARG1{cgrid *input1, "Input grid 1 for averaging"}
 * @ARG2{cgrid *input2, "Input grid 2 for averaging. Can be NULL if N/A"}
 * @ARG3{cgrid *input3, "Input grid 3 for averaging. Can be NULL if N/A"}
 * @ARG4{REAL *bins, "1-D array for the averaged values. This is an array with dimension equal to nbins"}
 * @ARG5{REAL binstep, "Binning step length"}
 * @ARG6{INT nbins, "Number of bins requested"}
 * @ARG7{char volel, "1: direct sum or 0: radial average"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_spherical_average(cgrid *input1, cgrid *input2, cgrid *input3, REAL *bins, REAL binstep, INT nbins, char volel) {

  INT nx = input1->nx, ny = input1->ny, nz = input1->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2, idx, nxy = nx * ny;
  REAL step = input1->step, x0 = input1->x0, y0 = input1->y0, z0 = input1->z0, r, x, y, z;
  REAL complex *value1 = input1->value, *value2, *value3;
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
    fprintf(stderr, "libgrid: Out of memory in cgrid_spherical_average().\n");
    abort();
  }
  bzero(nvals, sizeof(INT) * (size_t) nbins);
  bzero(bins, sizeof(REAL) * (size_t) nbins);

// TODO: Can't execute in parallel (reduction for bins[idx] needed
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;
      r = SQRT(x * x + y * y + z * z);
      idx = (INT) (0.5 + r / binstep);
      if(idx < nbins) {
        bins[idx] += csqnorm(value1[ijnz + k]);
        if(value2) bins[idx] += csqnorm(value2[ijnz + k]);
        if(value3) bins[idx] += csqnorm(value3[ijnz + k]);
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
 * @FUNC{cgrid_spherical_average_reciprocal, "Spherical average of grid in reciprocal space"}
 * @DESC{"Compute spherical average in the reciprocal space of power spectrum with respect to the grid origin
         (result 1-D grid). This can compute the average for three grids but any of them can be
         given as NULL if not needed."}
 * @ARG1{cgrid *input1, "Input grid 1 for averaging, but this complex data (i.e., *after* FFT)"}
 * @ARG2{cgrid *input2, "Input grid 2 for averaging, but this complex data (i.e., *after* FFT)"}
 * @ARG3{cgrid *input3, "Input grid 3 for averaging, but this complex data (i.e., *after* FFT)"}
 * @ARG4{REAL *bins, "1-D array for the averaged values. This is an array with dimension equal to nbins"}
 * @ARG5{REAL binstep, "Binning step length for k"}
 * @ARG6{INT nbins, "Number of bins requested"}
 * @ARG7{char volel, "1: radial sum or 0: radial average"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_spherical_average_reciprocal(cgrid *input1, cgrid *input2, cgrid *input3, REAL *bins, REAL binstep, INT nbins, char volel) {

  INT nx = input1->nx, ny = input1->ny, nz = input1->nz, idx, nxy = nx * ny;
  REAL step = input1->step, r, kx, ky, kz, nrm = input1->fft_norm;
  REAL complex *value1 = input1->value, *value2, *value3;
  REAL lx = 2.0 * M_PI / (((REAL) nx) * step), ly = 2.0 * M_PI / (((REAL) ny) * step), lz = 2.0 * M_PI / (((REAL) nz) * step);
  INT *nvals, ij, i, j, k, ijnz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;

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
    fprintf(stderr, "libgrid: Out of memory in cgrid_spherical_average_reciprocal().\n");
    abort();
  }
  bzero(nvals, sizeof(INT) * (size_t) nbins);
  bzero(bins, sizeof(REAL) * (size_t) nbins);

// TODO: Can't execute in parallel (reduction for bins[idx] needed
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
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
        kz = ((REAL) k) * lz; /* - kz0; */
      else
        kz = -((REAL) (nz - k)) * lz; /* - kz0; */
      r = SQRT(kx * kx + ky * ky + kz * kz);
      idx = (INT) (0.5 + r / binstep);
      if(idx < nbins) {
        bins[idx] += csqnorm(value1[ijnz + k]);
        if(value2) bins[idx] += csqnorm(value2[ijnz + k]);
        if(value3) bins[idx] += csqnorm(value3[ijnz + k]);
        nvals[idx]++;
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
 * @FUNC{cgrid_dealias, "FFT dealiasing"}
 * @DESC{"Apply dealias to grid by a given rule. Note that the grid must be in Fourier space"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @ARG2{char rule, "1: 2/3 rule, 2: 1/2 (or 2/4) rule"}
 * @RVAL{void, "No return value"}
 * 
 */

EXPORT void cgrid_dealias(cgrid *grid, char rule) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  switch (rule) {
    case 1:
      cgrid_zero_index(grid, nx/2 - nx/6, nx/2 + nx/6 + 1, ny/2 - ny/6, ny/2 + ny/6 + 1, nz/2 - nz/6, nz/2 + nz/6 + 1);
      break;
    case 2:
      cgrid_zero_index(grid, nx/2 - nx/4, nx/2 + nx/4 + 1, ny/2 - ny/4, ny/2 + ny/4 + 1, nz/2 - nz/4, nz/2 + nz/4 + 1);
      break;
    default:
      fprintf(stderr, "libgrid: Illegal dealias rule in cgrid_dealias().\n");
      exit(1);
  }
}

/*
 * @FUNC{cgrid_dealias2, "Apply dealias with limit"}
 * @DESC{"Apply dealias to grid by a given $k_max$ value (i.e., zero when $|k| > k_max$).
          Note that the grid must be in Fourier space"}
 * @ARG1{cgrid *grid, "Grid for the operation"}
 * @ARG2{REAL kmax, "Maximum value for k"}
 * @RVAL{void, "No return value"}
 * 
 */

EXPORT void cgrid_dealias2(cgrid *grid, REAL kmax) {

  INT nx = grid->nx, ny = grid->ny, nz = grid->nz, nxy, i, j, k, ij, ijnz, nx2, ny2, nz2;
  REAL kx, ky, kz, r, lx, ly, lz, step = grid->step;
  REAL complex *value = grid->value;

  if(kmax < 0.0) {
    fprintf(stderr, "libgrid: Negative kmax in cgrid_dealias2().\n");
    abort();
  }
#ifdef USE_CUDA
  if(cuda_status() && !cgrid_cuda_dealias2(grid, kmax)) return;
#endif

  nx2 = nx / 2;
  ny2 = ny / 2;
  nz2 = nz / 2;
  nxy = nx * ny;
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
#pragma omp parallel for firstprivate(nx,ny,nz,nx2,ny2,nz2,nxy,value,lx,ly,lz,kmax) private(ij,ijnz,k,r,i,j,kx,ky,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
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
        kz = ((REAL) k) * lz; /* - kz0; */
      else
        kz = -((REAL) (nz - k)) * lz; /* - kz0; */
      r = SQRT(kx * kx + ky * ky + kz * kz);
      if(r > kmax) value[ijnz + k] = 0.0;
    }
  }
}

/*
 * @FUNC{cgrid_mapk, "Map function to grid in reciprocal space"}
 * @DESC{"Map function onto grid in the reciprocal space. 
          The arguments to the user specified function are:
          user data (void *) and kx,ky,kz (REAL) coordinates"}
 * @ARG1{cgrid *grid, "Destination grid for the operation"}
 * @ARG2{REAL complex (*func), "Function providing the mapping"}
 * @ARG3{void *farg, "Pointer to user specified data"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void cgrid_mapk(cgrid *grid, REAL complex (*func)(void *arg, REAL kx, REAL ky, REAL kz), void *farg) {

  INT i, j, k, ij, ijnz, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL kx, ky, kz, step = grid->step, lx, ly, lz;
  REAL kx0 = grid->kx0, ky0 = grid->ky0, kz0 = grid->kz0;
  REAL complex *value = grid->value;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(value, 0);
#endif
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
#pragma omp parallel for firstprivate(farg,nx,ny,nz,nxy,step,func,value,kx0,ky0,kz0,lx,ly,lz,nx2,ny2,nz2) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    i = ij / ny;
    j = ij % ny;
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
      value[ijnz + k] = func(farg, kx, ky, kz);
    }
  }
}
