/*
 * Convert flux vector field to netcdf (3D only).
 *
 * Usage: vector2cdf density flux_x flux_y flux_z cdffile
 *
 * In visit, convert the fx, fy and fz fields to a vector field by:
 * 1) Controls -> Expressions
 * 2) Enter name: vecfield
 * 3) Type: Vector Mesh Variable
 * 4) Standard Editor: {fx,fy,fz}
 * 5) Apply.
 * 
 * Use vecfield variable for plotting the vector field.
 * Note that density is also included and can be overlaid
 * with the vector field in visit.
 *
 */

#include <stdlib.h>
#include <strings.h>
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

rgrid *tmp_grid = NULL;
REAL *x, *y, *z;

void reverse_xz(rgrid *in, rgrid *out) {

  INT i, j, k;
  INT nx = in->nx, ny = in->ny, nz = in->nz;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++)
	out->value[k * nx * ny + j * nx + i] = rgrid_value_at_index(in, i, j, k);
}

rgrid *read_grid(char *file) {

  FILE *fp;
  REAL step;
  INT i, nx, ny, nz;
  rgrid *grid;
  static int been_here = 0;

  if(!(fp = fopen(file, "r"))) { 
    fprintf(stderr, "Can't open grid file %s.\n", file);
    exit(1);
  }
  grid = rgrid_read(NULL, fp);
  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  step = grid->step;

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
    
  fclose(fp);

  if(tmp_grid == NULL)
    tmp_grid = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "tmp");

  reverse_xz(grid, tmp_grid);
  bcopy(tmp_grid->value, grid->value, sizeof(double) * (size_t) (nx * ny * nz));
  return grid;
}

int main(int argc, char **argv) {

  int ncid, varid1, varid2, varid3, varid4, varid5, varid6, varid7;
  int dimids[3], retval;
  char *density, *flux1, *flux2, *flux3, *cdffile;
  rgrid *de, *fl1, *fl2, *fl3;
  INT nx, ny, nz;

  if(argc != 6) {
    fprintf(stderr, "Usage: vector2cdf density flux_x flux_y flux_z cdffile\n");
    exit(1);
  }
  density = argv[1];
  flux1 = argv[2];
  flux2 = argv[3];
  flux3 = argv[4];
  cdffile = argv[5];
  
  fl1 = read_grid(flux1);  // x (or z)
  fl2 = read_grid(flux2);  // y (or r)
  fl3 = read_grid(flux3);  // z
  de = read_grid(density);  // density
  nx = de->nx;
  ny = de->ny;
  nz = de->nz;
 
  if((retval = nc_create(cdffile, NC_CLOBBER | NC_64BIT_OFFSET, &ncid))) {
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
  nc_def_var(ncid, "fx", DATA_NC, 3, dimids, &varid1);
  nc_def_var(ncid, "fy", DATA_NC, 3, dimids, &varid2);
  nc_def_var(ncid, "fz", DATA_NC, 3, dimids, &varid3);
  nc_def_var(ncid, "rho", DATA_NC, 3, dimids, &varid4);
  nc_enddef(ncid);
#ifdef SINGLE_PREC
  nc_put_var_float(ncid, varid1, fl1->value);
  nc_put_var_float(ncid, varid2, fl2->value);
  nc_put_var_float(ncid, varid3, fl3->value);
  nc_put_var_float(ncid, varid4, de->value);
  nc_put_var_float(ncid, varid5, z);
  nc_put_var_float(ncid, varid6, y);
  nc_put_var_float(ncid, varid7, x);
#else
  nc_put_var_double(ncid, varid1, fl1->value);
  nc_put_var_double(ncid, varid2, fl2->value);
  nc_put_var_double(ncid, varid3, fl3->value);
  nc_put_var_double(ncid, varid4, de->value);
  nc_put_var_double(ncid, varid5, z);
  nc_put_var_double(ncid, varid6, y);
  nc_put_var_double(ncid, varid7, x);
#endif
  
  nc_close(ncid);
  return 0;
}

