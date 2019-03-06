/*
 * Routines for analyzing wavefunctions.
 *
 */

#include "grid.h"
#include "au.h"
#include "private.h"

/*
 * Calculate the velocity field.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * vy     = y output grid containing the velocity (rgrid *).
 * vz     = z output grid containing the velocity (rgrid *).
 * cutoff = cutoff value for velocity (|v| <= cutoff) (REAL).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz, REAL cutoff) {
  
  grid_wf_velocity_x(gwf, vx, cutoff);
  grid_wf_velocity_y(gwf, vy, cutoff);
  grid_wf_velocity_z(gwf, vz, cutoff);
}

/*
 * Calculate the velocity field x component using:
 * v = -\frac{i\hbar}{2m} (d/dx) ln(\psi/\psi^*).
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * cutoff = cutoff value for velocity (|v| <= cutoff) (REAL).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_x(wf *gwf, rgrid *vx, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_x(gwf, vx, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vx,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i+1, j, k);
      pm = cgrid_value_at_index(grid, i-1, j, k);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp);
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vx, i, j, k, tmp);      
    }
  }
}

/*
 * Calculate the velocity field y component using:
 * v = -\frac{i\hbar}{2m} (d/dy) ln(\psi/\psi^*).
 *
 * gwf    = wavefunction for the operation (wf *).
 * vy     = y output grid containing the velocity (rgrid *).
 * cutoff = cutoff value for velocity (|v| <= cutoff) (REAL).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_y(wf *gwf, rgrid *vy, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_y(gwf, vy, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vy,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j+1, k);
      pm = cgrid_value_at_index(grid, i, j-1, k);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp);
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vy, i, j, k, tmp);
    }
  }
}

/*
 * Calculate the velocity field z component using:
 * v = -\frac{i\hbar}{2m} (d/dz) ln(\psi/\psi^*).
 *
 * gwf    = wavefunction for the operation (wf *).
 * vz     = z output grid containing the velocity (rgrid *).
 * cutoff = cutoff value for velocity (|v| <= cutoff) (REAL).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_z(wf *gwf, rgrid *vz, REAL cutoff) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_velocity_z(gwf, vz, inv_delta, cutoff)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vz,grid,inv_delta,cutoff) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j, k+1);
      pm = cgrid_value_at_index(grid, i, j, k-1);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp); // was CIMAG(CLOG(pp))
      if(tmp > cutoff) tmp = cutoff;
      else if(tmp < -cutoff) tmp = -cutoff;
      rgrid_value_to_index(vz, i, j, k, tmp);
    }
  }
}

/*
 * Calculate the probability flux x component (m^-2 s^-1). This is related to liquid momentum:
 * rho_mass * velocity = mass * flux.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_x    = x output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_x(wf *gwf, rgrid *flux_x) {

  cgrid *grid = gwf->grid;
  INT i, j, ij, k, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_x(gwf, flux_x, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_x,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_x, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k))
        * (cgrid_value_at_index(grid, i+1, j, k) - cgrid_value_at_index(grid, i-1, j, k))) * inv_delta);
  }
}

/*
 * Calculate the probability flux y component (m^-2 s^-1). This is related to liquid momentum:
 * rho_mass * velocity = mass * flux.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_y    = y output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_y(wf *gwf, rgrid *flux_y) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_y(gwf, flux_y, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_y,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_y, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k))
        * (cgrid_value_at_index(grid, i, j+1, k) - cgrid_value_at_index(grid, i, j-1, k))) * inv_delta);
  }
}

/*
 * Calculate the probability flux z component (m^-2 s^-1). This is related to liquid momentum:
 * rho_mass * velocity = mass * flux.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_z    = z output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_probability_flux_z(wf *gwf, rgrid *flux_z) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_probability_flux_z(gwf, flux_z, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,flux_z,grid,inv_delta) private(i,j,ij,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(flux_z, i, j, k, CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) 
        * (cgrid_value_at_index(grid, i, j, k+1) - cgrid_value_at_index(grid, i, j, k-1))) * inv_delta);
  }
}

/*
 * Calculate the probability flux (m^-2 s^-1). This is related to liquid momentum:
 * rho_mass * velocity = mass * flux.
 *
 * gwf        = wavefunction for the operation (wf *).
 * flux_x     = x output grid containing the flux (rgrid *).
 * flux_y     = y output grid containing the flux (rgrid *).
 * flux_z     = z output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 * NOTES: - This is not the liquid velocity! Divide by density (rho) to get v (velocity):
 *          v_i = flux_i / rho (i = x, y, z).
 *        - This is in units of # of particles. Multiply by gwf->mass to get this in terms of mass.
 */

EXPORT void grid_wf_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {
  
  /*
   * J(r) = -i (hbar/2m) ( psi^* grad psi - psi grad psi^* )
   *      = (hbar/m) Im[ psi^* grad psi ] 
   */
  grid_wf_probability_flux_x(gwf, flux_x);
  grid_wf_probability_flux_y(gwf, flux_y);
  grid_wf_probability_flux_z(gwf, flux_z);
}

/*
 * Calculate linear momentum expectation value <p_x>.
 *
 * wf        = Wafecuntion (wf *; input).
 * workspace = Workspace (rgrid *; input).
 *
 * Return <p_x>.
 *
 */

EXPORT REAL grid_wf_px(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_x(gwf, workspace);
  return rgrid_integral(workspace);
}

/*
 * Calculate linear momentum expectation value <p_y>.
 *
 * wf        = Wafecuntion (wf *; input).
 * workspace = Workspace (rgrid *; input).
 *
 * Return <p_y>.
 *
 */

EXPORT REAL grid_wf_py(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_y(gwf, workspace);
  return rgrid_integral(workspace);
}

/*
 * Calculate linear momentum expectation value <p_z>.
 *
 * wf        = Wafecuntion (wf *; input).
 * workspace = Workspace (rgrid *; input).
 *
 * Return <p_z>.
 *
 */

EXPORT REAL grid_wf_pz(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_z(gwf, workspace);
  return rgrid_integral(workspace);
}

/*
 * Calculate angular momentum expectation value <L_x>.
 *
 * wf         = Wavefunction (wf *; input).
 * workspace  = Workspace required for the operation (rgrid *; input).
 *
 * Return <L_x> (L_x = y p_z - z p_y).
 *
 */
 
EXPORT REAL grid_wf_lx(wf *wf, rgrid *workspace) {

  cgrid *grid = wf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, ny2 = ny / 2, nz2 = nz / 2;
  REAL step = grid->step, inv_delta = HBAR / (2.0 * wf->mass * step), y, z, y0 = grid->y0, z0 = grid->z0;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_lx(wf, workspace, inv_delta)) return rgrid_integral(workspace);
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,ny2,nz2,nxy,workspace,grid,inv_delta,y0,z0,step) private(i,j,ij,k,y,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    y = ((REAL) (j - ny2)) * step - y0;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;    
      rgrid_value_to_index(workspace, i, j, k, 
        (y * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i, j, k+1) - cgrid_value_at_index(grid, i, j, k-1))) /* y * p_z */
        -z * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i, j+1, k) - cgrid_value_at_index(grid, i, j-1, k))) /*-z * p_y */
        ) * inv_delta);
    }
  }
  return rgrid_integral(workspace);
}

/*
 * Calculate angular momentum expectation value <L_y>.
 *
 * wf         = Wavefunction (gwf *).
 * workspace  = Workspace required for the operation (rgrid *).
 *
 * Return <L_y> (L_y = z * p_x - x * p_z).
 *
 */
 
EXPORT REAL grid_wf_ly(wf *wf, rgrid *workspace) {

  cgrid *grid = wf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nx2 = nx / 2, nz2 = nz / 2;
  REAL step = grid->step, inv_delta = HBAR / (2.0 * wf->mass * step), x, z, x0 = grid->x0, z0 = grid->z0;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_ly(wf, workspace, inv_delta)) return rgrid_integral(workspace);
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nx2,nz2,nxy,workspace,grid,inv_delta,x0,z0,step) private(i,j,ij,k,x,z) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz2)) * step - z0;    
      rgrid_value_to_index(workspace, i, j, k, 
        (z * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i+1, j, k) - cgrid_value_at_index(grid, i-1, j, k))) /* z * p_x */
        -x * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i, j, k+1) - cgrid_value_at_index(grid, i, j, k-1))) /*-x * p_z */
        ) * inv_delta);
    }
  }
  return rgrid_integral(workspace);
}

/*
 * Calculate angular momentum expectation value <L_z>.
 *
 * wf         = Wavefunction (gwf *).
 * workspace  = Workspace required for the operation (rgrid *).
 *
 * Return <L_z> (L_z = x p_y - y p_x).
 *
 */
 
EXPORT REAL grid_wf_lz(wf *wf, rgrid *workspace) {

  cgrid *grid = wf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz, nx2 = nx / 2, ny2 = ny / 2;
  REAL step = grid->step, inv_delta = HBAR / (2.0 * wf->mass * step), x, y, x0 = grid->x0, y0 = grid->y0;
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_lz(wf, workspace, inv_delta)) return rgrid_integral(workspace);
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nx2,ny2,nxy,workspace,grid,inv_delta,x0,y0,step) private(i,j,ij,k,x,y) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    x = ((REAL) (i - nx2)) * step - x0;
    y = ((REAL) (j - ny2)) * step - y0;    
    for(k = 0; k < nz; k++)
      rgrid_value_to_index(workspace, i, j, k, 
        (x * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i, j+1, k) - cgrid_value_at_index(grid, i, j-1, k))) /* x * p_y */
        -y * CIMAG(CONJ(cgrid_value_at_index(grid, i, j, k)) * (cgrid_value_at_index(grid, i+1, j, k) - cgrid_value_at_index(grid, i-1, j, k))) /*-y * p_x */
        ) * inv_delta);
  }
  return rgrid_integral(workspace);
}

/*
 * Calculate angular momentum expectation values <L_x>, <L_y>, <L_z>.
 *
 * wf         = Wavefunction (gwf *).
 * lx         = Value of l_x (REAL *).
 * ly         = Value of l_y (REAL *).
 * lz         = Value of l_z (REAL *).
 * workspace  = Workspace required for the operation (rgrid *).
 *
 * NOTE: The old df_driver_L() routine returned angular momentum * mass.
 *       This routine does not include the mass.
 * 
 * No return value.
 *
 */
 
EXPORT void grid_wf_l(wf *wf, REAL *lx, REAL *ly, REAL *lz, rgrid *workspace) {

  *lx = grid_wf_lx(wf, workspace);
  *ly = grid_wf_ly(wf, workspace);
  *lz = grid_wf_lz(wf, workspace);
}

/*
 * Calculate the energy from the rotation constraint, -<omega*L>.
 *
 * gwf       = wavefunction for the system (wf *; input).
 * omega_x   = angular frequency in a.u., x-axis (REAL, input)
 * omega_y   = angular frequency in a.u., y-axis (REAL, input)
 * omega_z   = angular frequency in a.u., z-axis (REAL, input)
 * workspace = Workspace required for the operation (rgrid *).
 *
 * Returns the rotational energy.
 *
 */

EXPORT REAL grid_wf_rotational_energy(wf *gwf, REAL omega_x, REAL omega_y, REAL omega_z, rgrid *workspace) {

  REAL lx, ly, lz;

  grid_wf_l(gwf, &lx, &ly, &lz, workspace);
  return -(omega_x * lx * gwf->mass) - (omega_y * ly * gwf->mass) - (omega_z * lz * gwf->mass);
}

/*
 * Calculate incompressible kinetic energy density as a function of wave vector k (atomic unis).
 *
 * E(k) = (m/2) * (2pi)^3 (k^2/(4pi)) X "spherical average of the incompressible part in k-space"
 *
 * gwf        = Wave function to be analyzed (wf *; input).
 * bins       = Averages in k-space (REAL *; output). The array length is nbins.
 * binstep    = Step length in k-space in atomic units (REAL; input).
 * nbins      = Number of bins to use (INT; input).
 * workspace1 = Workspace 1 (rgrid *; input/output).
 * workspace2 = Workspace 2 (rgrid *; input/output).
 * workspace3 = Workspace 3 (rgrid *; input/output).
 * workspace4 = Workspace 4 (rgrid *; input/output).
 * workspace5 = Workspace 5 (rgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_incomp_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, rgrid *workspace5) {

  INT i;
  REAL step = gwf->grid->step, step3 = step * step * step;

  /* workspace1 = flux_x / sqrt(rho) = sqrt(rho) * v_x */
  /* workspace2 = flux_y / sqrt(rho) = sqrt(rho) * v_y */
  /* workspace3 = flux_z / sqrt(rho) = sqrt(rho) * v_z */
  grid_wf_probability_flux(gwf, workspace1, workspace2, workspace3);
  grid_wf_density(gwf, workspace4);
  rgrid_power(workspace4, workspace4, 0.5);
  rgrid_division_eps(workspace1, workspace1, workspace4, GRID_EPS2);
  rgrid_division_eps(workspace2, workspace2, workspace4, GRID_EPS2);
  rgrid_division_eps(workspace3, workspace3, workspace4, GRID_EPS2);
  rgrid_hodge_incomp(workspace1, workspace2, workspace3, workspace4, workspace5);

  /* FFT each component */
  rgrid_fft(workspace1); rgrid_multiply(workspace1, step3);
  rgrid_fft(workspace2); rgrid_multiply(workspace2, step3);
  rgrid_fft(workspace3); rgrid_multiply(workspace3, step3);
  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);
  
  for (i = 0; i < nbins; i++)
    bins[i] = bins[i] * (2.0 * M_PI) * (2.0 * M_PI) * (2.0 * M_PI) * 0.5 * gwf->mass / (4.0 * M_PI);
}

/*
 * Calculate compressible kinetic energy density as a function of wave vector k (atomic unis).
 *
 * E(k) = (m/2) * (2pi)^3 (k^2/(4pi)) X "spherical average of the compressible part in k-space"
 *
 * gwf        = Wave function to be analyzed (wf *; input).
 * bins       = Averages in k-space (REAL *; output). The array length is nbins.
 * binstep    = Step length in k-space in atomic units (REAL; input).
 * nbins      = Number of bins to use (INT; input).
 * workspace1 = Workspace 1 (rgrid *; input/output).
 * workspace2 = Workspace 2 (rgrid *; input/output).
 * workspace3 = Workspace 3 (rgrid *; input/output).
 * workspace4 = Workspace 4 (rgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_comp_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  INT i;
  REAL step = gwf->grid->step, step3 = step * step * step;

  /* workspace1 = flux_x / sqrt(rho) = sqrt(rho) * v_x */
  /* workspace2 = flux_y / sqrt(rho) = sqrt(rho) * v_y */
  /* workspace3 = flux_z / sqrt(rho) = sqrt(rho) * v_z */
  grid_wf_probability_flux(gwf, workspace1, workspace2, workspace3);
  grid_wf_density(gwf, workspace4);
  rgrid_power(workspace4, workspace4, 0.5);
  rgrid_division_eps(workspace1, workspace1, workspace4, GRID_EPS);
  rgrid_division_eps(workspace2, workspace2, workspace4, GRID_EPS);
  rgrid_division_eps(workspace3, workspace3, workspace4, GRID_EPS);
  rgrid_hodge_comp(workspace1, workspace2, workspace3, workspace4);

  /* FFT each component & multiply by step length^3 (Fourier series to Fourier integral) */
  rgrid_fft(workspace1); rgrid_multiply(workspace1, step3);
  rgrid_fft(workspace2); rgrid_multiply(workspace2, step3);
  rgrid_fft(workspace3); rgrid_multiply(workspace3, step3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);
  
  for (i = 0; i < nbins; i++)
    bins[i] = bins[i] * 0.5 * gwf->mass * (2.0 * M_PI) * (2.0 * M_PI) * (2.0 * M_PI) / (4.0 * M_PI);
}

/*
 * Calculate total classical kinetic energy minus the quantum pressure: int 1/2 * mass * rho * |v|^2 d^3
 * This is the kinetic energy due to classical flow / motion.
 *
 * wf         = Wavefunction (wf *; input).
 * workspace1 = Workspace (rgrid *; input).
 * workspace2 = Workspace (rgrid *; input).
 *
 * Returns the kinetic energy (REAL).
 *
 */

EXPORT REAL grid_wf_kinetic_energy_classical(wf *gwf, rgrid *workspace1, rgrid *workspace2) {

#if 0
  // brute force
  rgrid_zero(workspace2);
  grid_wf_probability_flux_x(gwf, workspace1);  // = rho * v_x
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);  // 1/2 * mass * rho^2 * v_x^2
  grid_wf_probability_flux_y(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  grid_wf_probability_flux_z(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  
  grid_wf_density(gwf, workspace1);
  rgrid_division_eps(workspace2, workspace2, workspace1, GRID_EPS2);  // divide out extra rho
  val = rgrid_integral(workspace2);
  return val;
#else
  // total - quantum pressure K.E.
  return grid_wf_kinetic_energy_cn(gwf) - grid_wf_kinetic_energy_qp(gwf, workspace1, workspace2);
#endif
}

/*
 * Calculate quantum pressure ( = -(hbar * hbar / (2m)) sqrt(rho) laplace sqrt(rho)).
 *
 * gwf        = Wavefunction (wf *; input).
 * workspace1 = Workspace (rgrid *; input).
 * workspace2 = Workspace (rgrid *; input).
 * 
 * Returns the kinetic energy (REAL).
 *
 */

EXPORT REAL grid_wf_kinetic_energy_qp(wf *gwf, rgrid *workspace1, rgrid *workspace2) {

  grid_wf_density(gwf, workspace1);
  rgrid_power(workspace1, workspace1, 0.5);
  rgrid_fd_laplace(workspace1, workspace2);
  rgrid_product(workspace1, workspace1, workspace2);
  return -(HBAR * HBAR / (2.0 * gwf->mass)) * rgrid_integral(workspace1);
}

/*
 * Calculate liquid circulation.
 *
 * wf         = Wave function (wf *; input).
 * nn         = Exponent (REAL; input). Usually nn = 1.0. See below.
 * workspace1 = Workspace (rgrid *; input).
 * workspace2 = Workspace (rgrid *; input).
 * workspace3 = Workspace (rgrid *; input).
 * workspace4 = Workspace (rgrid *; input).
 *
 * returns int |circulation|^nn
 *
 */

EXPORT REAL grid_wf_circulation(wf *gwf, REAL nn, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  grid_wf_probability_flux(gwf, workspace1, workspace2, workspace3);
  rgrid_abs_rot(workspace4, workspace1, workspace2, workspace3);
  if(nn != 1.0) rgrid_power(workspace4, workspace4, nn);
  return rgrid_integral(workspace4);
}
