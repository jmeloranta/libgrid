/*
 * Routines for analyzing wavefunctions.
 *
 */

#include "grid.h"
#include "au.h"
#include "private.h"

char grid_analyze_method = 0; // 0 = FD and 1 = FFT

/*
 * Generic frontend for calculating velocity. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * vy     = y output grid containing the velocity (rgrid *).
 * vz     = z output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz) {

  if(grid_analyze_method) grid_wf_fft_velocity(gwf, vx, vy, vz);
  else grid_wf_fd_velocity(gwf, vx, vy, vz);
}

/*
 * Generic frontend for calculating velocity. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_x(wf *gwf, rgrid *vx) {

  if(grid_analyze_method) grid_wf_fft_velocity_x(gwf, vx);
  else grid_wf_fd_velocity_x(gwf, vx);
}

/*
 * Generic frontend for calculating velocity. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vy     = y output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_y(wf *gwf, rgrid *vy) {

  if(grid_analyze_method) grid_wf_fft_velocity_y(gwf, vy);
  else grid_wf_fd_velocity_x(gwf, vy);
}

/*
 * Generic frontend for calculating velocity. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vz     = z output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_velocity_z(wf *gwf, rgrid *vz) {

  if(grid_analyze_method) grid_wf_fft_velocity_z(gwf, vz);
  else grid_wf_fd_velocity_z(gwf, vz);
}

/*
 * Calculate the velocity field using finite difference.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * vy     = y output grid containing the velocity (rgrid *).
 * vz     = z output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fd_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz) {
  
  grid_wf_fd_velocity_x(gwf, vx);
  grid_wf_fd_velocity_y(gwf, vy);
  grid_wf_fd_velocity_z(gwf, vz);
}

/*
 * Calculate the velocity field x component using:
 * v = -\frac{i\hbar}{2m} (d/dx) ln(\psi/\psi^*).
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fd_velocity_x(wf *gwf, rgrid *vx) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_velocity_x(gwf, vx, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vx,grid,inv_delta) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i+1, j, k);
      pm = cgrid_value_at_index(grid, i-1, j, k);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp);
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
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fd_velocity_y(wf *gwf, rgrid *vy) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_velocity_y(gwf, vy, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vy,grid,inv_delta) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j+1, k);
      pm = cgrid_value_at_index(grid, i, j-1, k);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp);
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
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fd_velocity_z(wf *gwf, rgrid *vz) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (4.0 * gwf->mass * grid->step), tmp;
  REAL complex pp, pm;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_velocity_z(gwf, vz, inv_delta)) return;
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vz,grid,inv_delta) private(pm,pp,i,j,ij,k,tmp) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      pp = cgrid_value_at_index(grid, i, j, k+1);
      pm = cgrid_value_at_index(grid, i, j, k-1);
      pp = pp * CONJ(pm) / (GRID_EPS + CONJ(pp) * pm);
      if(CABS(pp) < GRID_EPS) tmp = 0.0;
      else tmp = inv_delta * CARG(pp); // was CIMAG(CLOG(pp))
      rgrid_value_to_index(vz, i, j, k, tmp);
    }
  }
}

/*
 * Calculate the velocity field (derivatives in the Fourier space).
 *
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 * vy     = y output grid containing the velocity (rgrid *).
 * vz     = z output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fft_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL c = HBAR / (2.0 * gwf->mass);
  REAL complex tmp;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fft_velocity_setup(gwf, vz, c)) {
    rgrid_fft(vz);
    rgrid_fft_gradient_x(vz, vx);
    rgrid_fft_gradient_y(vz, vy);
    rgrid_fft_gradient_z(vz, vz);
    rgrid_inverse_fft(vx);
    rgrid_inverse_fft(vy);
    rgrid_inverse_fft(vz);
    return;
  }
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vx,vy,vz,c) private(i,j,ij,k,tmp) default(none) schedule(runtime)
  /* Prepare ln(psi/psi*) */
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      tmp = rgrid_value_at_index(vx, i, j, k);
      rgrid_value_to_index(vz, i, j, k, c * CARG(tmp / CONJ(tmp))); // - or + 
    }
  }
  rgrid_fft(vz);
  rgrid_fft_gradient_x(vz, vx);
  rgrid_fft_gradient_y(vz, vy);
  rgrid_fft_gradient_z(vz, vz);
  rgrid_inverse_fft(vx);
  rgrid_inverse_fft(vy);
  rgrid_inverse_fft(vz);
}

/*
 * Calculate the velocity field x component using:
 * v = -\frac{i\hbar}{2m} (d/dx) ln(\psi/\psi^*)
 * where d/dx is evaluated in the Fourier space.  
 * 
 * gwf    = wavefunction for the operation (wf *).
 * vx     = x output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fft_velocity_x(wf *gwf, rgrid *vx) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL c = HBAR / (2.0 * gwf->mass);
  REAL complex tmp;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fft_velocity_setup(gwf, vx, c)) {
    rgrid_fft(vx);
    rgrid_fft_gradient_x(vx, vx);
    rgrid_inverse_fft(vx);
    return;
  }
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vx,c) private(i,j,ij,k,tmp) default(none) schedule(runtime)
  /* Prepare ln(psi/psi*) */
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      tmp = rgrid_value_at_index(vx, i, j, k);
      rgrid_value_to_index(vx, i, j, k, c * CARG(tmp / CONJ(tmp)));
    }
  }
  rgrid_fft(vx);
  rgrid_fft_gradient_x(vx, vx);
  rgrid_inverse_fft(vx);
}

/*
 * Calculate the velocity field y component using:
 * v = -\frac{i\hbar}{2m} (d/dy) ln(\psi/\psi^*)
 * where d/dx is evaluated in the Fourier space.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vy     = y output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fft_velocity_y(wf *gwf, rgrid *vy) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL c = HBAR / (2.0 * gwf->mass);
  REAL complex tmp;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fft_velocity_setup(gwf, vy, c)) {
    rgrid_fft(vy);
    rgrid_fft_gradient_y(vy, vy);
    rgrid_inverse_fft(vy);
    return;
  }
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vy,c) private(i,j,ij,k,tmp) default(none) schedule(runtime)
  /* Prepare ln(psi/psi*) */
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      tmp = rgrid_value_at_index(vy, i, j, k);
      rgrid_value_to_index(vy, i, j, k, c * CARG(tmp / CONJ(tmp)));
    }
  }
  rgrid_fft(vy);
  rgrid_fft_gradient_y(vy, vy);
  rgrid_inverse_fft(vy);
}

/*
 * Calculate the velocity field z component using:
 * v = -\frac{i\hbar}{2m} (d/dz) ln(\psi/\psi^*)
 * where d/dx is evaluated in the Fourier space.
 *
 * gwf    = wavefunction for the operation (wf *).
 * vz     = z output grid containing the velocity (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fft_velocity_z(wf *gwf, rgrid *vz) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL c = HBAR / (2.0 * gwf->mass);
  REAL complex tmp;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fft_velocity_setup(gwf, vz, c)) {
    rgrid_fft(vz);
    rgrid_fft_gradient_y(vz, vz);
    rgrid_inverse_fft(vz);
    return;
  }
#endif
#pragma omp parallel for firstprivate(nx,ny,nz,nxy,vz,c) private(i,j,ij,k,tmp) default(none) schedule(runtime)
  /* Prepare ln(psi/psi*) */
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;
    for(k = 0; k < nz; k++) {
      tmp = rgrid_value_at_index(vz, i, j, k);
      rgrid_value_to_index(vz, i, j, k, c * CARG(tmp / CONJ(tmp)));
    }
  }
  rgrid_fft(vz);
  rgrid_fft_gradient_z(vz, vz);
  rgrid_inverse_fft(vz);
}

/*
 * Generic frontend for calculating probability flux. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf       = wavefunction for the operation (wf *).
 * flux_x    = x output grid containing the flux (rgrid *).
 * flux_y    = y output grid containing the flux (rgrid *).
 * flux_z    = z output grid containing the flux (rgrid *).
 *
 * No return value.
 */

EXPORT void grid_wf_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {

  if(grid_analyze_method) grid_wf_fft_probability_flux(gwf, flux_x, flux_y, flux_z);
  else grid_wf_fd_probability_flux(gwf, flux_x, flux_y, flux_z);
}

/*
 * Generic frontend for calculating probability flux. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf       = wavefunction for the operation (wf *).
 * flux_x    = x output grid containing the flux (rgrid *).
 *
 * No return value.
 */

EXPORT void grid_wf_probability_flux_x(wf *gwf, rgrid *flux_x) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_x(gwf, flux_x);
  else grid_wf_fd_probability_flux_x(gwf, flux_x);
}

/*
 * Generic frontend for calculating probability flux. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf       = wavefunction for the operation (wf *).
 * flux_y    = y output grid containing the flux (rgrid *).
 *
 * No return value.
 */

EXPORT void grid_wf_probability_flux_y(wf *gwf, rgrid *flux_y) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_y(gwf, flux_y);
  else grid_wf_fd_probability_flux_y(gwf, flux_y);
}

/*
 * Generic frontend for calculating probability flux. This may use FD or FFT
 * depending on the grid_analyze_method setting.
 *
 * gwf       = wavefunction for the operation (wf *).
 * flux_z    = z output grid containing the flux (rgrid *).
 *
 * No return value.
 */

EXPORT void grid_wf_probability_flux_z(wf *gwf, rgrid *flux_z) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_z(gwf, flux_z);
  else grid_wf_fd_probability_flux_z(gwf, flux_z);
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

EXPORT void grid_wf_fd_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {
  
  /*
   * J(r) = -i (hbar/2m) ( psi^* grad psi - psi grad psi^* )
   *      = (hbar/m) Im[ psi^* grad psi ] 
   */
  grid_wf_fd_probability_flux_x(gwf, flux_x);
  grid_wf_fd_probability_flux_y(gwf, flux_y);
  grid_wf_fd_probability_flux_z(gwf, flux_z);
}

/*
 * Calculate the probability flux x component (m^-2 s^-1). This is related to liquid momentum:
 * rho_mass * velocity = mass * flux. Uses finite difference.
 * 
 * gwf       = wavefunction for the operation (wf *).
 * flux_x    = x output grid containing the flux (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_fd_probability_flux_x(wf *gwf, rgrid *flux_x) {

  cgrid *grid = gwf->grid;
  INT i, j, ij, k, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_probability_flux_x(gwf, flux_x, inv_delta)) return;
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

EXPORT void grid_wf_fd_probability_flux_y(wf *gwf, rgrid *flux_y) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_probability_flux_y(gwf, flux_y, inv_delta)) return;
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

EXPORT void grid_wf_fd_probability_flux_z(wf *gwf, rgrid *flux_z) {

  cgrid *grid = gwf->grid;
  INT i, j, k, ij, nx = grid->nx, ny = grid->ny, nxy = nx * ny, nz = grid->nz;
  REAL inv_delta = HBAR / (2.0 * gwf->mass * grid->step);
  
#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_fd_probability_flux_z(gwf, flux_z, inv_delta)) return;
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
 * Calculate probability flux using FFT: J = rho * flux.
 *
 * gwf    = Wavefunction (gwf *; input).
 * flux_x = Flux x component (rgrid *; output).
 * flux_y = Flux y component (rgrid *; output).
 * flux_z = Flux z component (rgrid *; output).
 *
 * No return value.
 *
 * TODO: This is probably not the best way to do this.
 *
 */

EXPORT void grid_wf_fft_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {

  grid_wf_fft_velocity(gwf, flux_x, flux_y, flux_z);
  grid_product_norm(flux_x, flux_x, gwf->grid);
  grid_product_norm(flux_y, flux_y, gwf->grid);
  grid_product_norm(flux_z, flux_z, gwf->grid);
}

/*
 * Calculate probability flux using FFT: J = rho * flux.
 *
 * gwf    = Wavefunction (gwf *; input).
 * flux_x = Flux x component (rgrid *; output).
 *
 * No return value.
 *
 * TODO: This is probably not the best way to do this.
 *
 */

EXPORT void grid_wf_fft_probability_flux_x(wf *gwf, rgrid *flux_x) {

  grid_wf_fft_velocity_x(gwf, flux_x);
  grid_product_norm(flux_x, flux_x, gwf->grid);
}

/*
 * Calculate probability flux using FFT: J = rho * flux.
 *
 * gwf    = Wavefunction (gwf *; input).
 * flux_y = Flux y component (rgrid *; output).
 *
 * No return value.
 *
 * TODO: This is probably not the best way to do this.
 *
 */

EXPORT void grid_wf_fft_probability_flux_y(wf *gwf, rgrid *flux_y) {

  grid_wf_fft_velocity_y(gwf, flux_y);
  grid_product_norm(flux_y, flux_y, gwf->grid);
}

/*
 * Calculate probability flux using FFT: J = rho * flux.
 *
 * gwf    = Wavefunction (gwf *; input).
 * flux_z = Flux z component (rgrid *; output).
 *
 * No return value.
 *
 * TODO: This is probably not the best way to do this.
 *
 */

EXPORT void grid_wf_fft_probability_flux_z(wf *gwf, rgrid *flux_z) {

  grid_wf_fft_velocity_z(gwf, flux_z);
  grid_product_norm(flux_z, flux_z, gwf->grid);
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
 * workspace1 = Workspace required for the operation (rgrid *; input).
 * workspace2 = Workspace required for the operation (rgrid *; input).
 *
 * Return <L_x> (L_x = y p_z - z p_y).
 *
 */

EXPORT REAL grid_wf_lx(wf *wf, rgrid *workspace1, rgrid *workspace2) {

  grid_wf_pz(wf, workspace1);       // p_z
  rgrid_multiply_by_y(workspace1);   // yp_z
  grid_wf_py(wf, workspace2);       // p_y
  rgrid_multiply_by_z(workspace2);   // zp_y    
  rgrid_difference(workspace1, workspace1, workspace2); // yp_z - zp_y
  return rgrid_integral(workspace1);
}

/*
 * Calculate angular momentum expectation value <L_y>.
 *
 * wf         = Wavefunction (gwf *).
 * workspace1 = Workspace required for the operation (rgrid *; input).
 * workspace2 = Workspace required for the operation (rgrid *; input).
 *
 * Return <L_y> (L_y = z * p_x - x * p_z).
 *
 */
 
EXPORT REAL grid_wf_ly(wf *wf, rgrid *workspace1, rgrid *workspace2) {

  grid_wf_px(wf, workspace1);       // p_x
  rgrid_multiply_by_z(workspace1);   // zp_x
  grid_wf_pz(wf, workspace2);       // p_z
  rgrid_multiply_by_x(workspace2);   // xp_z
  rgrid_difference(workspace1, workspace1, workspace2); // zp_x - xp_z
  return rgrid_integral(workspace1);
}

/*
 * Calculate angular momentum expectation value <L_z>.
 *
 * wf         = Wavefunction (gwf *).
 * workspace1 = Workspace required for the operation (rgrid *; input).
 * workspace2 = Workspace required for the operation (rgrid *; input).
 *
 * Return <L_z> (L_z = x p_y - y p_x).
 *
 */
 
EXPORT REAL grid_wf_lz(wf *wf, rgrid *workspace1, rgrid *workspace2) {

  grid_wf_py(wf, workspace1);       // p_y
  rgrid_multiply_by_x(workspace1);   // xp_y
  grid_wf_px(wf, workspace2);       // p_x
  rgrid_multiply_by_y(workspace2);   // yp_x
  rgrid_difference(workspace1, workspace1, workspace2); // xp_y - yp_x
  return rgrid_integral(workspace1);
}

/*
 * Calculate angular momentum expectation values <L_x>, <L_y>, <L_z>.
 *
 * wf         = Wavefunction (gwf *).
 * lx         = Value of l_x (REAL *).
 * ly         = Value of l_y (REAL *).
 * lz         = Value of l_z (REAL *).
 * workspace1 = Workspace required for the operation (rgrid *; input).
 * workspace2 = Workspace required for the operation (rgrid *; input).
 *
 * NOTE: The old df_driver_L() routine returned angular momentum * mass.
 *       This routine does not include the mass.
 * 
 * No return value.
 *
 */
 
EXPORT void grid_wf_l(wf *wf, REAL *lx, REAL *ly, REAL *lz, rgrid *workspace1, rgrid *workspace2) {

  *lx = grid_wf_lx(wf, workspace1, workspace2);
  *ly = grid_wf_ly(wf, workspace1, workspace2);
  *lz = grid_wf_lz(wf, workspace1, workspace2);
}

/*
 * Calculate the energy from the rotation constraint, -<omega*L>.
 *
 * gwf        = wavefunction for the system (wf *; input).
 * omega_x    = angular frequency in a.u., x-axis (REAL, input)
 * omega_y    = angular frequency in a.u., y-axis (REAL, input)
 * omega_z    = angular frequency in a.u., z-axis (REAL, input)
 * workspace1 = Workspace required for the operation (rgrid *).
 * workspace2 = Workspace required for the operation (rgrid *).
 *
 * Returns the rotational energy.
 *
 */

EXPORT REAL grid_wf_rotational_energy(wf *gwf, REAL omega_x, REAL omega_y, REAL omega_z, rgrid *workspace1, rgrid *workspace2) {

  REAL lx, ly, lz;

  grid_wf_l(gwf, &lx, &ly, &lz, workspace1, workspace2);
  return -(omega_x * lx * gwf->mass) - (omega_y * ly * gwf->mass) - (omega_z * lz * gwf->mass);
}

/*
 * Calculate kinetic energy density as a function of wave vector k (atomic unis).
 *
 * E(k) = 4pi (m/2) k^2 \int |(sqrt(rho)v)(k,theta,phi)|^2 sin(theta) dtheta dphi
 *
 * Total K.E. is then int E(k) dk.
 *
 * gwf        = Wave function to be analyzed (wf *; input).
 * bins       = Averages in k-space (REAL *; output). The array length is nbins.
 * binstep    = Step length in k-space in atomic units (REAL; input).
 * nbins      = Number of bins to use (INT; input).
 * workspace1 = Workspace 1 (rgrid *; input/output).
 * workspace2 = Workspace 2 (rgrid *; input/output).
 * workspace3 = Workspace 3 (rgrid *; input/output).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3) {

  INT i;

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass;
}

/*
 * Calculate incompressible kinetic energy density as a function of wave vector k (atomic unis).
 *
 * E(k) = 4pi (m/2) k^2 \int |(sqrt(rho)v)(k,theta,phi)|^2 sin(theta) dtheta dphi
 *
 * where (m/2) |sqrt(rho)v|^2 corresponds to the incompressible part of kinetic energy.
 * Total incompressible K.E. is then int E(k) dk.
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

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3);

  rgrid_hodge_incomp(workspace1, workspace2, workspace3, workspace4, workspace5);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass;
}

/*
 * Calculate compressible kinetic energy density as a function of wave vector k (atomic unis).
 *
 * E(k) = 4pi (m/2) \int |(sqrt(rho)v)(k,theta,phi)|^2 sin(theta) dtheta dphi
 *
 * where (m/2) |sqrt(rho)v|^2 is the compressible part of kinetic energy.
 * Total compressible K.E. is then int E(k) dk.
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

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3);


  rgrid_hodge_comp(workspace1, workspace2, workspace3, workspace4);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass;
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

  rgrid_zero(workspace2);
  grid_wf_probability_flux_x(gwf, workspace1);  // = rho * v_x
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);  // 1/2 * mass * rho^2 * v_x^2
  grid_wf_probability_flux_y(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  grid_wf_probability_flux_z(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  
  grid_wf_density(gwf, workspace1);
  rgrid_division_eps(workspace2, workspace2, workspace1, GRID_EPS2);

  return rgrid_integral(workspace2);
}

/*
 * Calculate quantum pressure ( = -(hbar * hbar / (2m)) sqrt(rho) laplace sqrt(rho)).
 *
 * gwf        = Wavefunction (wf *; input).
 * workspace1 = Workspace (rgrid *; input).
 * workspace2 = Workspace (rgrid *; input).
 * 
 * Returns the quantum kinetic energy (REAL).
 *
 */

EXPORT REAL grid_wf_kinetic_energy_qp(wf *gwf, rgrid *workspace1, rgrid *workspace2) {

  grid_wf_density(gwf, workspace1);
  rgrid_power(workspace1, workspace1, 0.5);
  if(grid_analyze_method) rgrid_fft_laplace(workspace1, workspace2);
  else rgrid_fd_laplace(workspace1, workspace2);
  rgrid_product(workspace1, workspace2, workspace1);
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
