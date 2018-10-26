/*
 * Convert wavefunction grid to: density and (flux_x, flux_y, flux_z).
 *
 * Usage: wf2cdf wf-grid cdffile
 *
 * In visit, convert the fx, fy and fz current density components to a 
 * vector field:
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
 * In paraview:
 * 1) Make sure that the extension is .nc (do not use .cdf or anything else!)
 *    Choose the generic format (last choice).
 * 2) rho = density, {fx,fy,fz} are the current density compoenents.
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

REAL *x, *y, *z, step;

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
  INT i;
  static char been_here = 0;
  wf *wf;
  INT nx, ny, nz;
  
  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "Can't open grid file %s.\n", file);
    exit(1);
  }

  nx = ny = nz = 1;
  fread(&nx, sizeof(INT), 1, fp);
  fread(&ny, sizeof(INT), 1, fp);
  fread(&nz, sizeof(INT), 1, fp);
  fread(&step, sizeof(REAL), 1, fp);
  wf = grid_wf_alloc(nx, ny, nz, step, MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "wf");
  
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

  int ncid, varid1, varid2, varid3, varid4, varid5, varid6, varid7;
  int dimids[3], retval;
  rgrid *flux_x, *flux_y, *flux_z, *density, *tmp;
  wf *wf;
  INT nx, ny, nz;
  
  if(argc != 3) {
    fprintf(stderr, "Usage: wf2cdf wf-file cdffile\n");
    exit(1);
  }
  wf = read_grid(argv[1]);
  nx = wf->grid->nx;
  ny = wf->grid->ny;
  nz = wf->grid->nz;
  flux_x = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "flux_x");
  flux_y = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "flux_y");
  flux_z = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "flux_z");
  density = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "density");
  tmp = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, NULL, "tmp");
  grid_wf_probability_flux(wf, flux_x, flux_y, flux_z);
  grid_wf_density(wf, density);

  // Need to have x the fastest running index - reverse X and Z
  reverse_xz(flux_x, tmp);
  bcopy(tmp->value, flux_x->value, sizeof(REAL) * (size_t) (nx * ny * nz));

  reverse_xz(flux_y, tmp);
  bcopy(tmp->value, flux_y->value, sizeof(REAL) * (size_t) (nx * ny * nz));

  reverse_xz(flux_z, tmp);
  bcopy(tmp->value, flux_z->value, sizeof(REAL) * (size_t) (nx * ny * nz));

  reverse_xz(density, tmp);
  bcopy(tmp->value, density->value, sizeof(REAL) * (size_t) (nx * ny * nz));
  
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

  nc_def_var(ncid, "fz", DATA_NC, 3, dimids, &varid3);
  nc_def_var(ncid, "fy", DATA_NC, 3, dimids, &varid2);
  nc_def_var(ncid, "fx", DATA_NC, 3, dimids, &varid1);
  nc_def_var(ncid, "rho", DATA_NC, 3, dimids, &varid4);
  nc_enddef(ncid);
#ifdef SINGLE_PREC
  nc_put_var_float(ncid, varid3, flux_z->value);
  nc_put_var_float(ncid, varid2, flux_y->value);
  nc_put_var_float(ncid, varid1, flux_x->value);
  nc_put_var_float(ncid, varid4, density->value);
  nc_put_var_float(ncid, varid5, z);
  nc_put_var_float(ncid, varid6, y);
  nc_put_var_float(ncid, varid7, x);
#else
  nc_put_var_double(ncid, varid3, flux_z->value);
  nc_put_var_double(ncid, varid2, flux_y->value);
  nc_put_var_double(ncid, varid1, flux_x->value);
  nc_put_var_double(ncid, varid4, density->value);
  nc_put_var_double(ncid, varid5, z);
  nc_put_var_double(ncid, varid6, y);
  nc_put_var_double(ncid, varid7, x);
#endif
  nc_close(ncid);
  return 0;
}

