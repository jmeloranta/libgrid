/*
 * Convert real grid format to netcdf (3D only).
 *
 * Usage: grd2cdf gridfile cdffile
 *
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <netcdf.h>
#include <grid/grid.h>

#ifdef SINGLE_PREC
#define DATA_NC NC_FLOAT
#else
#define DATA_NC NC_DOUBLE
#endif

void reverse_xz(rgrid *in, rgrid *out) {

  INT i, j, k;
  INT nx = in->nx, ny = in->ny, nz = in->nz;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	out->value[k * nx * ny + j * nx + i] = rgrid_value_at_index(in, i, j, k);
}

int main(int argc, char **argv) {

  char *cdffile = argv[3], *gridfile = argv[2];
  FILE *fp;
 INT i;
  int ncid, varid0, varid1, varid2, varid3, dimids[3], retval, dim;
  REAL step, *x, *y, *z;
  INT nx, ny, nz;
  rgrid *orig, *final;

  if(argc != 3) {
    fprintf(stderr, "Usage: grd2cdf gridfile cdffile\n");
    exit(1);
  }
  if(!(fp = fopen(gridfile, "r"))) {
    fprintf(stderr, "Can't open grid file %s.\n", gridfile);
    exit(1);
  }

  nx = ny = nz = 1;
  fread(&nx, sizeof(INT), 1, fp);
  fread(&ny, sizeof(INT), 1, fp);
  fread(&nz, sizeof(INT), 1, fp);
  fread(&step, sizeof(REAL), 1, fp);

  fprintf(stderr, "nx = " FMT_I ", ny = " FMT_I ", nz = " FMT_I ", step = " FMT_R "\n", nx, ny, nz, step);
  
  orig = rgrid_alloc(nx, ny, nz, step, RGRID_NEUMANN_BOUNDARY, NULL, "orig");
  final = rgrid_alloc(nx, ny, nz, step, RGRID_NEUMANN_BOUNDARY, NULL, "final");

  if(!(x = (REAL *) malloc(sizeof(REAL) * (size_t) nx))) {
    fprintf(stderr, "Out of memory.\n");
    exit(1);
  }
  for (i = 0; i < nx; i++) x[i] = ((REAL) (i - nx/2)) * step;

  if(!(y = (REAL *) malloc(sizeof(REAL) * (size_t) ny))) {
    fprintf(stderr, "Out of memory.\n");
    exit(1);
  }
  for (i = 0; i < ny; i++) y[i] = ((REAL) (i - ny/2)) * step;

  if(!(z = (REAL *) malloc(sizeof(REAL) * (size_t) nz))) {
    fprintf(stderr, "Out of memory.\n");
    exit(1);
  }
  for (i = 0; i < nz; i++) z[i] = ((REAL) (i - nz/2)) * step;
  
  fread(orig->value, sizeof(REAL), (size_t) (nx * ny * nz), fp);
  fclose(fp);

  reverse_xz(orig, final);

  if((retval = nc_create(cdffile, NC_CLOBBER| NC_64BIT_OFFSET, &ncid))) {
    puts(nc_strerror(retval));
    fprintf(stderr, "Error in nc_open().\n");
    exit(1);
  }
  
  nc_def_dim(ncid, "z", (size_t) nz, &dimids[0]);
  nc_def_var(ncid, "z", DATA_NC, 1, &dimids[0], &varid0);
  nc_def_dim(ncid, "y", (size_t) ny, &dimids[1]);
  nc_def_var(ncid, "y", DATA_NC, 1, &dimids[1], &varid1);
  nc_def_dim(ncid, "x", (size_t) nx, &dimids[2]);
  nc_def_var(ncid, "x", DATA_NC, 1, &dimids[2], &varid2);

  nc_def_var(ncid, "density", DATA_NC, dim, dimids, &varid3);
  nc_enddef(ncid);
#ifdef SINGLE_PREC
  nc_put_var_float(ncid, varid3, final->value);
#else
  nc_put_var_double(ncid, varid3, final->value);
#endif
#ifdef SINGLE_PREC
  nc_put_var_float(ncid, varid2, x);
  nc_put_var_float(ncid, varid1, y);
  nc_put_var_float(ncid, varid0, z);
#else
  nc_put_var_double(ncid, varid2, x);
  nc_put_var_double(ncid, varid1, y);
  nc_put_var_double(ncid, varid0, z);
#endif

  nc_close(ncid);
  return 0;
}

