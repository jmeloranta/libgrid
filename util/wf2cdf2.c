/*
 * Convert wavefunction grid to: real and imaginary parts (to be processed further)
 *
 * Usage: wf2cdf2 wf-grid cdffile
 *
 * See comments in wf2cdf.c
 *
 */

#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <netcdf.h>
#include <grid/grid.h>
#include <grid/au.h>

#ifdef SINGLE_PREC
#define DATA_NC NC_FLOAT
#else
#define DATA_NC NC_DOUBLE
#endif

/* FIXME: This should not be hard coded to helium mass */
#define MASS (4.002602 / GRID_AUTOAMU)

REAL *x, *y, *z;

void reverse_xz(rgrid *in, rgrid *out) {

  INT i, j, k;
  INT nx = in->nx, ny = in->ny, nz = in->nz;
  
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	out->value[k * nx * ny + j * nx + i] = rgrid_value_at_index(in, i, j, k);
}

wf *read_grid(char *file) {

  FILE *fp;
  INT i, nx, ny, nz;
  REAL step;
  static int been_here = 0;
  wf *wf;
  
  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "Can't open grid file %s.\n", file);
    exit(1);
  }

  nx = ny = nz = 1;
  fread(&nx, sizeof(INT), 1, fp);
  fread(&ny, sizeof(INT), 1, fp);
  fread(&nz, sizeof(INT), 1, fp);
  fread(&step, sizeof(REAL), 1, fp);
  wf = grid_wf_alloc(nx, ny, nz, step, MASS, WF_NEUMANN_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "wf");
  
  if(!been_here) {
    if(!(x = (REAL *) malloc(sizeof(REAL) * (size_t) nx))) {
      fprintf(stderr, "Out of memory.\n");
      exit(1);
    }
    if(!(y = (REAL *) malloc(sizeof(REAL) * (size_t) ny))) {
      fprintf(stderr, "Out of memory.\n");
      exit(1);
    }
    if(!(z = (REAL *) malloc(sizeof(REAL) * (size_t) nz))) {
      fprintf(stderr, "Out of memory.\n");
      exit(1);
    }
    // TODO: the grid files do not store the grid center x0,y0,z0...
    // origin assumed to be at the center
    for (i = 0; i < nx; i++) x[i] = ((REAL) (i - nx/2)) * step;
    for (i = 0; i < ny; i++) y[i] = ((REAL) (i - ny/2)) * step;
    for (i = 0; i < nz; i++) z[i] = ((REAL) (i - nz/2)) * step;
    been_here = 1;
  }

  fprintf(stderr, "File %s: nx = " FMT_I ", ny = " FMT_I ", nz = " FMT_I ", step = " FMT_R "\n", file, nx, ny, nz, step);
  fread(wf->grid->value, sizeof(REAL complex), (size_t) (nx * ny * nz), fp);
  fclose(fp);
  return wf;
}

int main(int argc, char **argv) {

  int ncid, varid1, varid2, varid5, varid6, varid7;
  int dimids[3], retval;
  INT nx, ny, nz;
  REAL step;
  rgrid *re_part, *im_part, *tmp;
  wf *wf;
  
  if(argc != 3) {
    fprintf(stderr, "Usage: wf2cdf wf-file2 cdffile\n");
    exit(1);
  }
  wf = read_grid(argv[1]);
  nx = wf->grid->nx;
  ny = wf->grid->ny;
  nz = wf->grid->nz;
  step = wf->grid->step;
  re_part = rgrid_alloc(nx, ny, nz, step, RGRID_NEUMANN_BOUNDARY, NULL, "real part");
  im_part = rgrid_alloc(nx, ny, nz, step, RGRID_NEUMANN_BOUNDARY, NULL, "imag part");
  tmp =  rgrid_alloc(nx, ny, nz, step, RGRID_NEUMANN_BOUNDARY, NULL, "tmp");

  grid_complex_re_to_real(re_part, wf->grid);

  reverse_xz(re_part, tmp);
  bcopy(tmp->value, re_part->value, sizeof(REAL) * (size_t) (nx * ny * nz));

  grid_complex_im_to_real(im_part, wf->grid);

  reverse_xz(im_part, tmp);
  bcopy(tmp->value, im_part->value, sizeof(REAL) * (size_t) (nx * ny * nz));
  
  if((retval = nc_create(argv[2], NC_CLOBBER | NC_64BIT_OFFSET, &ncid))) {
    puts(nc_strerror(retval));
    fprintf(stderr, "Error in nc_open().\n");
    exit(1);
  }
  nc_def_dim(ncid, "z", (size_t) nz, &dimids[0]);
  nc_def_var(ncid, "z", DATA_NC, 1, &dimids[0], &varid5);
  nc_def_dim(ncid, "y", (size_t) ny, &dimids[1]);
  nc_def_var(ncid, "y", DATA_NC, 1, &dimids[1], &varid6);
  nc_def_dim(ncid, "x", (size_t) nx, &dimids[2]);
  nc_def_var(ncid, "x", DATA_NC, 1, &dimids[2], &varid7);
  nc_def_var(ncid, "re", DATA_NC, 3, dimids, &varid1);
  nc_def_var(ncid, "im", DATA_NC, 3, dimids, &varid2);
  nc_enddef(ncid);
#ifdef SINGLE_PREC
  nc_put_var_float(ncid, varid1, re_part->value);
  nc_put_var_float(ncid, varid2, im_part->value);
  nc_put_var_float(ncid, varid5, z);
  nc_put_var_float(ncid, varid6, y);
  nc_put_var_float(ncid, varid7, x);
#else
  nc_put_var_double(ncid, varid1, re_part->value);
  nc_put_var_double(ncid, varid2, im_part->value);
  nc_put_var_double(ncid, varid5, z);
  nc_put_var_double(ncid, varid6, y);
  nc_put_var_double(ncid, varid7, x);
#endif
  nc_close(ncid);
  return 0;
}

