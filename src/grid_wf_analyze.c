/*
 * Routines for analyzing wavefunctions.
 *
 */

#include "grid.h"
#include "au.h"
#include "cprivate.h"

char grid_analyze_method = (char) -1; // 0 = FD and 1 = FFT, -1 = not set

/*
 * @FUNC{grid_wf_analyze_method, "Set finite difference or FFT derivatives"}
 * @DESC{"Function to switch between FD and FFT based routines. Note that only FFT-based
          routines can be used on multi-GPU systems"}
 * @ARG1{char method, "0: Finite difference (FD) or 1 = FFT"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_analyze_method(char method) {

  grid_analyze_method = method;
}

/*
 * @FUNC{grid_wf_velocity, "Calculate velocity field of wavefunction"}
 * @DESC{"This function calculates the velocity field for a given wavefunction (Madelung representation). 
          This may use either FD or FFT depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *vx, "output grid containing x component of velocity"}
 * @ARG3{rgrid *vy, "output grid containing y component of velocity"}
 * @ARG4{rgrid *vz, "output grid containing z component of velocity"}
 * @ARG5{REAL eps, "Epsilon for (safely) dividing by $|psi|^2$"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_velocity(wf *gwf, rgrid *vx, rgrid *vy, rgrid *vz, REAL eps) {

  grid_wf_probability_flux(gwf, vx, vy, vz);  
  grid_division_norm(vx, vx, gwf->grid, eps);
  grid_division_norm(vy, vy, gwf->grid, eps);
  grid_division_norm(vz, vz, gwf->grid, eps);
}

/*
 * @FUNC{grid_wf_velocity_x, "Calculate velocity field x of wavefunction"}
 * @DESC{"This function calculates the velocity field x component for a given wavefunction (Madelung representation).
          This may use either FD or FFT depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *vx, "output grid containing x component of velocity"}
 * @ARG5{REAL eps, "Epsilon for (safely) dividing by $|psi|^2$"}
 *
 */

EXPORT void grid_wf_velocity_x(wf *gwf, rgrid *vx, REAL eps) {

  grid_wf_probability_flux_x(gwf, vx);
  grid_division_norm(vx, vx, gwf->grid, eps);
}

/*
 * @FUNC{grid_wf_velocity_y, "Calculate velocity field y of wavefunction"}
 * @DESC{"This function calculates the velocity field y component for a given wavefunction (Madelung representation).
          This may use either FD or FFT depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *vy, "output grid containing y component of velocity"}
 * @ARG5{REAL eps, "Epsilon for (safely) dividing by $|psi|^2$"}
 *
 */

EXPORT void grid_wf_velocity_y(wf *gwf, rgrid *vy, REAL eps) {

  grid_wf_probability_flux_y(gwf, vy);
  grid_division_norm(vy, vy, gwf->grid, eps);
}

/*
 * @FUNC{grid_wf_velocity_z, "Calculate velocity field z of wavefunction"}
 * @DESC{"This function calculates the velocity field z component for a given wavefunction (Madelung representation).
          This may use either FD or FFT depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *vz, "output grid containing z component of velocity"}
 * @ARG5{REAL eps, "Epsilon for (safely) dividing by $|psi|^2$"}
 *
 */

EXPORT void grid_wf_velocity_z(wf *gwf, rgrid *vz, REAL eps) {

  grid_wf_probability_flux_z(gwf, vz);
  grid_division_norm(vz, vz, gwf->grid, eps);
}

/*
 * @FUNC{grid_wf_probability_flux, "Calculate probability flux for wavefunction"}
 * @DESC{"Calculate probability flux for given wavefunction. This may use FD or FFT
          depending on the grid_analyze_method setting. This is related to liquid momentum:
          rho_mass * velocity = mass * flux\\
          Notes:\\
          - This is not the liquid velocity. Divide by density (rho) to get v (velocity);
            v_i = flux_i / rho (i = x, y, z).\\
          - This is in units of \# of particles. Multiply by gwf->mass to get this in terms of mass"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @ARG3{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @ARG4{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {

  if(grid_analyze_method) grid_wf_fft_probability_flux(gwf, flux_x, flux_y, flux_z);
  else grid_wf_fd_probability_flux(gwf, flux_x, flux_y, flux_z);
}

/*
 * @FUNC{grid_wf_probability_flux_x, "Calculate probability flux x for wavefunction"}
 * @DESC{"Calculate probability flux x component for given wavefunction. This may use FD or FFT
          depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_probability_flux_x(wf *gwf, rgrid *flux_x) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_x(gwf, flux_x);
  else grid_wf_fd_probability_flux_x(gwf, flux_x);
}

/*
 * @FUNC{grid_wf_probability_flux_y, "Calculate probability flux y for wavefunction"}
 * @DESC{"Calculate probability flux y component for given wavefunction. This may use FD or FFT
          depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_probability_flux_y(wf *gwf, rgrid *flux_y) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_y(gwf, flux_y);
  else grid_wf_fd_probability_flux_y(gwf, flux_y);
}

/*
 * @FUNC{grid_wf_probability_flux_z, "Calculate probability flux z for wavefunction"}
 * @DESC{"Calculate probability flux z component for given wavefunction. This may use FD or FFT
          depending on the grid_analyze_method setting"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_probability_flux_z(wf *gwf, rgrid *flux_z) {

  if(grid_analyze_method) grid_wf_fft_probability_flux_z(gwf, flux_z);
  else grid_wf_fd_probability_flux_z(gwf, flux_z);
}

/*
 * @FUNC{grid_wf_fd_probability_flux, "Calculate probability flux for wavefunction (finite difference)"}
 * @DESC{"Calculate probability flux for given wavefunction. This uses explicitly finite difference (FD).
          This is related to liquid momentum: rho_mass * velocity = mass * flux.\\
          Notes:\\
          - This is not the liquid velocity. Divide by density (rho) to get v (velocity);
            v_i = flux_i / rho (i = x, y, z).\\
          - This is in units of \# of particles. Multiply by gwf-$>$mass to get this in terms of mass"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @ARG3{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @ARG4{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
 *
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
 * @FUNC{grid_wf_fd_probability_flux_x, "Calculate probability flux x for wavefunction (finite difference)"}
 * @DESC{"Calculate probability flux x component for given wavefunction. This uses explicitly FD"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_wf_fd_probability_flux_y, "Calculate probability flux y for wavefunction (finite difference)"}
 * @DESC{"Calculate probability flux y component for given wavefunction. This uses explicitly FD"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_wf_fd_probability_flux_z, "Calculate probability flux z for wavefunction (finite difference)"}
 * @DESC{"Calculate probability flux z component for given wavefunction. This uses explicitly FD"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
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
 * @FUNC{grid_wf_fft_probability_flux, "Calculate probability flux for wavefunction (FFT)"}
 * @DESC{"Calculate probability flux for given wavefunction. This uses explicitly FFT.
   This evaluates: $(\hbar/m) * Im[\psi^* \nabla psi]$"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @ARG3{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @ARG4{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_fft_probability_flux(wf *gwf, rgrid *flux_x, rgrid *flux_y, rgrid *flux_z) {

  grid_wf_fft_probability_flux_x(gwf, flux_x);
  grid_wf_fft_probability_flux_y(gwf, flux_y);
  grid_wf_fft_probability_flux_z(gwf, flux_z);
}

/*
 * @FUNC{grid_wf_fft_probability_flux_x, "Calculate probability flux x for wavefunction (FFT)"}
 * @DESC{"Calculate probability flux x component for given wavefunction. This uses explicitly FFT"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_x, "Output grid containing x component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_fft_probability_flux_x(wf *gwf, rgrid *flux_x) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_clone(grid, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_x(cworkspace, cworkspace);
  cgrid_inverse_fft_norm(cworkspace);
  cgrid_conjugate_product(cworkspace, grid, cworkspace);
  grid_complex_im_to_real(flux_x, cworkspace);  
  rgrid_multiply(flux_x, HBAR / gwf->mass);
}

/*
 * @FUNC{grid_wf_fft_probability_flux_y, "Calculate probability flux y for wavefunction (FFT)"}
 * @DESC{"Calculate probability flux y component for given wavefunction. This uses explicitly FFT"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_y, "Output grid containing y component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_fft_probability_flux_y(wf *gwf, rgrid *flux_y) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_clone(grid, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_y(cworkspace, cworkspace);
  cgrid_inverse_fft_norm(cworkspace);
  cgrid_conjugate_product(cworkspace, grid, cworkspace);
  grid_complex_im_to_real(flux_y, cworkspace);  
  rgrid_multiply(flux_y, HBAR / gwf->mass);
}

/*
 * @FUNC{grid_wf_fft_probability_flux_z, "Calculate probability flux z for wavefunction (FFT)"}
 * @DESC{"Calculate probability flux z component for given wavefunction. This uses explicitly FFT"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{rgrid *flux_z, "Output grid containing z component of the flux"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_fft_probability_flux_z(wf *gwf, rgrid *flux_z) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_clone(grid, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_z(cworkspace, cworkspace);
  cgrid_inverse_fft_norm(cworkspace);
  cgrid_conjugate_product(cworkspace, grid, cworkspace);
  grid_complex_im_to_real(flux_z, cworkspace);  
  rgrid_multiply(flux_z, HBAR / gwf->mass);
}

/*
 * @FUNC{grid_wf_px, "Linear momentum x of wavefunction"}
 * @DESC{"Calculate linear momentum expectation value x component: $<p> = mass * <probability flux>$"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *workspace, "Workspace"}
 * @RVAL{REAL, "Returns $<p_x>$"}
 *
 */

EXPORT REAL grid_wf_px(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_x(gwf, workspace);
  return rgrid_integral(workspace) * gwf->mass;
}

/*
 * @FUNC{grid_wf_py, "Linear momentum y of wavefunction"}
 * @DESC{"Calculate linear momentum expectation value y component: $<p> = mass * <probability flux>$"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *workspace, "Workspace"}
 * @RVAL{REAL, "Returns $<p_y>$"}
 *
 */

EXPORT REAL grid_wf_py(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_y(gwf, workspace);
  return rgrid_integral(workspace) * gwf->mass;
}

/*
 * @FUNC{grid_wf_pz, "Linear momentum z of wavefunction"}
 * @DESC{"Calculate linear momentum expectation value z component: $<p> = mass * <probability flux>$"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *workspace, "Workspace"}
 * @RVAL{REAL, "Returns $<p_z>$"}
 *
 */

EXPORT REAL grid_wf_pz(wf *gwf, rgrid *workspace) {

  grid_wf_probability_flux_z(gwf, workspace);
  return rgrid_integral(workspace) * gwf->mass;
}

/*
 * @FUNC{grid_wf_lx_op, "Angular momentum x operator"}
 * @DESC{"Operator for angular momentum x component"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *dst, "Destination grid"}
 * @ARG3{rgrid *workspace, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_lx_op(wf *wf, cgrid *dst, cgrid *workspace) {

  grid_wf_momentum_z(wf, dst);       // p_z
  cgrid_multiply_by_y(dst);   // yp_z
  grid_wf_momentum_y(wf, workspace);       // p_y
  cgrid_multiply_by_z(workspace);   // zp_y    
  cgrid_difference(dst, dst, workspace); // yp_z - zp_y
}

/*
 * @FUNC{grid_wf_ly_op, "Angular momentum y operator"}
 * @DESC{"Operator for angular momentum y component"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *dst, "Destination grid"}
 * @ARG3{rgrid *workspace, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */
 
EXPORT void grid_wf_ly_op(wf *wf, cgrid *dst, cgrid *workspace) {

  grid_wf_momentum_x(wf, dst);       // p_x
  cgrid_multiply_by_z(dst);   // zp_x
  grid_wf_momentum_z(wf, workspace);       // p_z
  cgrid_multiply_by_x(workspace);   // xp_z
  cgrid_difference(dst, dst, workspace); // zp_x - xp_z
}

/*
 * @FUNC{grid_wf_lz_op, "Angular momentum z operator"}
 * @DESC{"Operator for angular momentum z component"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *dst, "Destination grid"}
 * @ARG3{rgrid *workspace, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */
 
EXPORT void grid_wf_lz_op(wf *wf, cgrid *dst, cgrid *workspace) {

  grid_wf_momentum_y(wf, dst);       // p_y
  cgrid_multiply_by_x(dst);   // xp_y
  grid_wf_momentum_x(wf, workspace);       // p_x
  cgrid_multiply_by_y(workspace);   // yp_x
  cgrid_difference(dst, dst, workspace); // xp_y - yp_x
}

/*
 * @FUNC{grid_wf_lx, "Angular momentum x expectation value"}
 * @DESC{"Calculate expectation value for angular momentum x component ($L_x = y p_z - z p_y$)"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{cgrid *workspace1, "Workspace 1"}
 * @ARG3{cgrid *workspace2, "Workspace 2"}
 * @RVAL{REAL, "Expectation value $<L_x>$"}
 *
 */

EXPORT REAL grid_wf_lx(wf *wf, cgrid *workspace1, cgrid *workspace2) {

  grid_wf_lx_op(wf, workspace1, workspace2);
  return CREAL(cgrid_integral_of_conjugate_product(wf->grid, workspace1));
}

/*
 * @FUNC{grid_wf_ly, "Angular momentum y expectation value"}
 * @DESC{"Calculate expectation value for angular momentum y component ($L_y = z * p_x - x * p_z$)"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{cgrid *workspace1, "Workspace 1"}
 * @ARG3{cgrid *workspace2, "Workspace 2"}
 * @RVAL{REAL, "Expectation value $<L_y>$"}
 *
 */
 
EXPORT REAL grid_wf_ly(wf *wf, cgrid *workspace1, cgrid *workspace2) {

  grid_wf_ly_op(wf, workspace1, workspace2);
  return CREAL(cgrid_integral_of_conjugate_product(wf->grid, workspace1));
}

/*
 * @FUNC{grid_wf_lz, "Angular momentum z expectation value"}
 * @DESC{"Calculate expectation value for angular momentum z component ($L_z = x p_y - y p_x$)"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{cgrid *workspace1, "Workspace 1"}
 * @ARG3{cgrid *workspace2, "Workspace 2"}
 * @RVAL{REAL, "Expectation value $<L_z>$"}
 *
 */
 
EXPORT REAL grid_wf_lz(wf *wf, cgrid *workspace1, cgrid *workspace2) {

  grid_wf_lz_op(wf, workspace1, workspace2);
  return CREAL(cgrid_integral_of_conjugate_product(wf->grid, workspace1));
}

/*
 * @FUNC{grid_wf_l, "Angular momentum expectation value"}
 * @DESC{"Calculate expectation value for angular momentum vector. Note that this does not include;5D
          multiplication by mass"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{REAL *lx, "Value for $L_x$"}
 * @ARG3{REAL *ly, "Value for $L_y$"}
 * @ARG4{REAL *lz, "Value for $L_z$"}
 * @ARG5{cgrid *workspace1, "Workspace 1"}
 * @ARG6{cgrid *workspace2, "Workspace 2"}
 * @RVAL{void, "No return value"}
 * 
 */
 
EXPORT void grid_wf_l(wf *wf, REAL *lx, REAL *ly, REAL *lz, cgrid *workspace1, cgrid *workspace2) {

  *lx = grid_wf_lx(wf, workspace1, workspace2);
  *ly = grid_wf_ly(wf, workspace1, workspace2);
  *lz = grid_wf_lz(wf, workspace1, workspace2);
}

/*
 * @FUNC{grid_wf_rotational_energy, "Energy from rotational constraint"}
 * @DESC{"Calculate the energy from the rotation constraint, $-<omega*L>$"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{REAL omega_x, "Angular frequency in a.u., x-axis"}
 * @ARG3{REAL omega_y, "Angular frequency in a.u., y-axis"}
 * @ARG4{REAL omega_z, "Angular frequency in a.u., z-axis"}
 * @ARG5{cgrid *workspace1, "Workspace 1"}
 * @ARG6{cgrid *workspace2, "Workspace 2"}
 * @RVAL{REAL, "Returns the rotational energy"}
 *
 */

EXPORT REAL grid_wf_rotational_energy(wf *gwf, REAL omega_x, REAL omega_y, REAL omega_z, cgrid *workspace1, cgrid *workspace2) {

  REAL lx, ly, lz;

  grid_wf_l(gwf, &lx, &ly, &lz, workspace1, workspace2);
  return -(omega_x * lx + omega_y * ly + omega_z * lz);
}

/*
 * @FUNC{grid_wf_KE, "Kinetic energy density in reciprocal space"}
 * @DESC{"Calculate kinetic energy density as a function of wave vector $|k|$ (atomic units):\\
          $E(k) = 4\pi (m/2) k^2 \int |(\sqrt(rho)v)(k,theta,phi)|^2 \sin(theta) dtheta dphi$\\
          The total K.E. is then $\int E(k) dk$"}
 * @ARG1{wf *gwf, "Wave function to be analyzed"}
 * @ARG2{REAL *bins, "Array for the averages in k-space. The array length is nbins"}
 * @ARG3{REAL binstep, "Step length in k-space in atomic units"}
 * @ARG4{INT nbins, "Number of bins to use"}
 * @ARG5{rgrid *workspace1, "Workspace 1"}
 * @ARG6{rgrid *workspace2, "Workspace 2"}
 * @ARG7{rgrid *workspace3, "Workspace 3"}
 * @ARG8{rgrid *workspace4, "Workspace 4"}
 * @ARG9{REAL eps, "Epislon for (safe) division by $|psi|^2$"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, REAL eps) {

  INT i;
  REAL step = gwf->grid->step;

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3, eps);

  grid_wf_density(gwf, workspace4);
  rgrid_power(workspace4, workspace4, 0.5);    // sqrt(rho)

  rgrid_product(workspace1, workspace1, workspace4); // Multiply velocity field by sqrt(rho)
  rgrid_product(workspace2, workspace2, workspace4);
  rgrid_product(workspace3, workspace3, workspace4);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass * step * step * step / binstep;
}

/*
 * @FUNC{grid_wf_incomp_KE, "Incompressible kinetic energy density in reciprocal space"}
 * @DESC{"Calculate incompressible kinetic energy density as a function of wave vector $|k|$ (atomic units):\\
          $E(k) = (m/2) k^2 \int |(\sqrt(rho)v)(k,theta,phi)|^2 \sin(theta) dtheta dphi$\\
          where $(m/2) |\sqrt(rho)v|^2$ corresponds to the incompressible part of kinetic energy.
          Total incompressible K.E. is then $\int E(k) dk$"}
 * @ARG1{wf *gwf, "Wavefunction to be analyzed"}
 * @ARG2{REAL *bins, "Averages in k-space. The array length is nbins"}
 * @ARG3{REAL binstep, "Step length in k-space in atomic units"}
 * @ARG4{INT nbins, "Number of bins to use"}
 * @ARG5{rgrid *workspace1, "Workspace 1"}
 * @ARG6{rgrid *workspace2, "Workspace 2"}
 * @ARG7{rgrid *workspace3, "Workspace 4"}
 * @ARG8{rgrid *workspace4, "Workspace 5"}
 * @ARG9{rgrid *workspace5, "Workspace 6"}
 * @ARG10{REAL eps, "Epislon for (safe) division by $|psi|^2$"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_incomp_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, rgrid *workspace5, REAL eps) {

  INT i;
  REAL step = gwf->grid->step;

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3, eps);

  rgrid_hodge_incomp(workspace1, workspace2, workspace3, workspace4, workspace5);

  grid_wf_density(gwf, workspace4);
  rgrid_power(workspace4, workspace4, 0.5);    // sqrt(rho)

  rgrid_product(workspace1, workspace1, workspace4); // Multiply velocity field by sqrt(rho)
  rgrid_product(workspace2, workspace2, workspace4);
  rgrid_product(workspace3, workspace3, workspace4);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass * step * step * step / binstep;
}

/*
 * @FUNC{grid_wf_comp_KE, "Compressible kinetic energy density in reciprocal space"}
 * @DESC{"Calculate compressible kinetic energy density as a function of wave vector $|k|$ (atomic units):\\
          $E(k) = 4pi (m/2) \int |(\sqrt(rho)v)(k,theta,phi)|^2 \sin(theta) dtheta dphi$\\
          where $(m/2) |\sqrt(rho)v|^2$ is the compressible part of kinetic energy.
          Total compressible K.E. is then $\int E(k) dk$"}
 * @ARG1{wf *gwf, "Wavefunction to be analyzed"}
 * @ARG2{REAL *bins, "Averages in k-space. The array length is nbins"}
 * @ARG3{REAL binstep, "Step length in k-space in atomic units"}
 * @ARG4{INT nbins, "Number of bins to use"}
 * @ARG5{rgrid *workspace1, "Workspace 1"}
 * @ARG6{rgrid *workspace2, "Workspace 2"}
 * @ARG7{rgrid *workspace3, "Workspace 3"}
 * @ARG8{rgrid *workspace4, "Workspace 4"}
 * @ARG9{REAL eps, "Epislon for (safe) division by $|psi|^2$"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_comp_KE(wf *gwf, REAL *bins, REAL binstep, INT nbins, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, REAL eps) {

  INT i;
  REAL step = gwf->grid->step;

  grid_wf_velocity(gwf, workspace1, workspace2, workspace3, eps);

  rgrid_hodge_comp(workspace1, workspace2, workspace3, workspace4);

  grid_wf_density(gwf, workspace4);
  rgrid_power(workspace4, workspace4, 0.5);    // sqrt(rho)

  rgrid_product(workspace1, workspace1, workspace4); // Multiply velocity field by sqrt(rho)
  rgrid_product(workspace2, workspace2, workspace4);
  rgrid_product(workspace3, workspace3, workspace4);

  rgrid_fft(workspace1);
  rgrid_fft(workspace2);
  rgrid_fft(workspace3);

  rgrid_spherical_average_reciprocal(workspace1, workspace2, workspace3, bins, binstep, nbins, 1);

  for(i = 0; i < nbins; i++)
    bins[i] *= 0.5 * gwf->mass * step * step * step / binstep;
}

/*
 * @FUNC{grid_wf_average_occupation, "Average occupation number in reciprocal space"}
 * @DESC{"Calculate average spherical occupation numbers in the Fourier space, $n(|k|)$"}
 * @ARG1{wf *gwf, "Wavefunction to be analyzed"}
 * @ARG2{REAL *bins, "Averages in k-space. The array length is nbins"}
 * @ARG3{REAL binstep, "Step length in k-space in atomic units"}
 * @ARG4{INT nbins, "Number of bins to use"}
 * @ARG5{cgrid *cworkspace, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_average_occupation(wf *gwf, REAL *bins, REAL binstep, INT nbins, cgrid *cworkspace) {

  INT i;
  REAL step = gwf->grid->step;

  cgrid_copy(cworkspace, gwf->grid);
  cgrid_fft(cworkspace);

  cgrid_spherical_average_reciprocal(cworkspace, NULL, NULL, bins, binstep, nbins, 0);
  for (i = 0; i < nbins; i++)
    bins[i] *= step * step * step;
}

/*
 * @FUNC{grid_wf_total_occupation, "Total occupation number in reciprocal space"}
 * @DESC{"Calculate total spherical occupation numbers in the Fourier space, $n(|k|)$"}
 * @ARG1{wf *gwf, "Wavefunction to be analyzed"}
 * @ARG2{REAL *bins, "Averages in k-space. The array length is nbins"}
 * @ARG3{REAL binstep, "Step length in k-space in atomic units"}
 * @ARG4{INT nbins, "Number of bins to use"}
 * @ARG5{cgrid *cworkspace, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_total_occupation(wf *gwf, REAL *bins, REAL binstep, INT nbins, cgrid *cworkspace) {

  INT i;
  REAL step = gwf->grid->step;

  cgrid_copy(cworkspace, gwf->grid);
  cgrid_fft(cworkspace);

  cgrid_spherical_average_reciprocal(cworkspace, NULL, NULL, bins, binstep, nbins, 1);
  for (i = 0; i < nbins; i++)
    bins[i] *= step * step * step;
}

/*
 * @FUNC{grid_wf_kinetic_energy_classical, "Classical kinetic energy"}
 * @DESC{"Calculate classical kinetic energy: $\int \frac{1}{2}  m \rho * |v|^2 d^3r$.
          This is the kinetic energy due to classical flow / motion"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{rgrid *workspace1, "Workspace 1"}
 * @ARG3{rgrid *workspace2, "Workspace 2"}
 * @ARG4{REAL eps, "Epsilon for (safe) dividing by $|psi|^2$"}
 * @RVAL{REAL, "Returns the kinetic energy"}
 *
 */

EXPORT REAL grid_wf_kinetic_energy_classical(wf *gwf, rgrid *workspace1, rgrid *workspace2, REAL eps) {

  rgrid_zero(workspace2);
  grid_wf_probability_flux_x(gwf, workspace1);  // = rho * v_x
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);  // 1/2 * mass * rho^2 * v_x^2
  grid_wf_probability_flux_y(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  grid_wf_probability_flux_z(gwf, workspace1);
  rgrid_add_scaled_product(workspace2, 0.5 * gwf->mass, workspace1, workspace1);
  
  grid_wf_density(gwf, workspace1);
  rgrid_division_eps(workspace2, workspace2, workspace1, eps);

  return rgrid_integral(workspace2);
}

/*
 * @FUNC{grid_wf_kinetic_energy_qp, "Quantum kinetic energy (quantum pressure)"}
 * @DESC{"Calculate quantum (pressure) energy: $\frac{\hbar^2}{2m} \int |\nabla \sqrt{\rho})|^2$\\
   Note that ideally, classical + quantum = total kinetic energy. 
   However, there are often numerical accuracy issues such that this does not hold well"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{rgrid *workspace1, "Workspace 1"}
 * @ARG3{rgrid *workspace2, "Workspace 2"}
 * @ARG4{rgrid *workspace3, "Workspace 3"}
 * @RVAL{REAL, "Returns the quantum kinetic energy"}
 *
 */

EXPORT REAL grid_wf_kinetic_energy_qp(wf *gwf, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3) {

  grid_wf_density(gwf, workspace1);
  rgrid_power(workspace1, workspace1, 0.5);
  rgrid_zero(workspace2);
  if(grid_analyze_method) {
    rgrid_fft(workspace1); // |grad sqrt(rho)|^2

    rgrid_fft_gradient_x(workspace1, workspace3);
    rgrid_inverse_fft_norm(workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);

    rgrid_fft_gradient_y(workspace1, workspace3);
    rgrid_inverse_fft_norm(workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);

    rgrid_fft_gradient_z(workspace1, workspace3);
    rgrid_inverse_fft_norm(workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);
  } else {
    rgrid_fd_gradient_x(workspace1, workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);
    rgrid_fd_gradient_y(workspace1, workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);
    rgrid_fd_gradient_z(workspace1, workspace3);
    rgrid_power(workspace3, workspace3, 2.0);
    rgrid_sum(workspace2, workspace2, workspace3);
  }
  return (HBAR * HBAR / (2.0 * gwf->mass)) * rgrid_integral(workspace2);
}

/*
 * @FUNC{grid_wf_circulation, "Calculate liquid circulation"}
 * @DESC{"Calculate total liquid circulation ($\int |rot (\rho v)|^nn$)"}
 * @ARG1{wf *wf, "Wavefunction"}
 * @ARG2{REAL nn, "Exponent $nn$. Usually $nn = 1.0$"}
 * @ARG3{rgrid *workspace1, "Workspace 1"}
 * @ARG4{rgrid *workspace2, "Workspace 2"}
 * @ARG5{rgrid *workspace3, "Workspace 3"}
 * @ARG6{rgrid *workspace4, "Workspace 4"}
 * @RVAL{REAL, "Returns total circulation"}
 *
 */

EXPORT REAL grid_wf_circulation(wf *gwf, REAL nn, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  grid_wf_probability_flux(gwf, workspace1, workspace2, workspace3);
  rgrid_abs_rot(workspace4, workspace1, workspace2, workspace3);
  if(nn != 1.0) rgrid_power(workspace4, workspace4, nn);
  return rgrid_integral(workspace4);
}

/*
 * @FUNC{grid_wf_temperature, "Calculate BEC temperature"}
 * @DESC{"Calculate $T_BEC$ for a given wavefunction according to:
          $T_BEC = T_{\lambda} * ((n - ngnd) / n)^{exponent}$"}
 * @ARG1{wf *wf, "Wavefunction for which the temperature is calculated"}
 * @ARG2{REAL tl, "Lambda temperature ($T_{\lambda}$)"}
 * @ARG3{REAL exponent, "Exponent (2/3 for BEC, approx. 1/6 for superfluid $^4$He with $T_{\lambda} \approx 2.19$"}
 * @RVAL{REAL, "Returns temperature in Kelvin"}
 *
 */

EXPORT REAL grid_wf_temperature(wf *gwf, REAL tl, REAL exponent) {

  REAL n, ngnd, tmp;
  cgrid *grid = gwf->grid;

  cgrid_fft(grid);
  ngnd = csqnorm(cgrid_value_at_index(grid, 0, 0, 0));
  tmp = grid->step;
  grid->step = 1.0;
  n = cgrid_integral_of_square(grid);
  grid->step = tmp;
  cgrid_inverse_fft_norm(grid);

  return tl * POW(1.0 - ngnd / n, exponent);  // All normalization constants cancel
}
  
/*
 * @FUNC{grid_wf_superfluid, "Calculate superfluid fraction"}
 * @DESC{"Calculate the superfluid fraction: $n_s = n_gnd / n$"}
 * @ARG1{wf *wf, "Wavefunction for which the temperature is calculated"}
 * @RVAL{REAL, "Returns the fraction between 0 and 1"}
 *
 */

EXPORT REAL grid_wf_superfluid(wf *gwf) {

  cgrid *grid = gwf->grid;

#if 1 // Real space
  // |\int\psi d^3r|^2 / (N * V)
  return (csqnorm(cgrid_integral(grid)) / (grid_wf_norm(gwf) * (grid->step * grid->step * grid->step * (REAL) (grid->nx * grid->ny * grid->nz))));
#else // Reciprocal space, amplitude of the DC component
  REAL n, ngnd, tmp;
  cgrid_fft(grid);
  ngnd = csqnorm(cgrid_value_at_index(grid, 0, 0, 0));
  tmp = grid->step;
  grid->step = 1.0;
  n = cgrid_integral_of_square(grid);
  grid->step = tmp;
  cgrid_inverse_fft_norm(grid);
  return ngnd / n;
#endif
}

/*
 * @FUNC{grid_wf_normalfluid, "Calculate normal fraction"}
 * @DESC{"Calculate the normal fraction: $n_n = 1 - n_gnd / n$"}
 * @ARG1{wf *wf, "Wavefunction for which the temperature is calculated"}
 * @RVAL{REAL, "Returns the fraction between 0 and 1"}
 * 
 */

EXPORT REAL grid_wf_normalfluid(wf *gwf) {

  return 1.0 - grid_wf_superfluid(gwf);
}

/*
 * @FUNC{grid_wf_entropy, "Calculate entropy"}
 * @DESC{"Calculate entropy: $S = -K_b\sum_i p_i\ln(p_i)$"}
 * @ARG1{wf *wf, "Wavefunction for the calculation"}
 * @ARG2{cgrid *cworkspace, "Workspace required for the operation"}
 * @RVAL{REAL, "Returns entropy"}
 *
 */

EXPORT REAL grid_wf_entropy(wf *wf, cgrid *cworkspace) {

  REAL S, tmp;
  INT ij, k, ijnz, nxy = wf->grid->nx * wf->grid->ny, nz = wf->grid->nz;
  REAL complex *cwrk = cworkspace->value;

  cgrid_copy(cworkspace, wf->grid);
  cgrid_fft(cworkspace);
  tmp = cworkspace->step;
  cworkspace->step = 1.0;  // use sum rather than integral
  cgrid_multiply(cworkspace, SQRT(1.0 / cgrid_integral_of_square(cworkspace)));
  cgrid_abs_power(cworkspace, cworkspace, 2.0);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_entropy(cworkspace)) {
    S = -GRID_AUKB * CREAL(cgrid_integral(cworkspace));
    cworkspace->step = tmp;  
    return S;
  }    
#endif

#pragma omp parallel for firstprivate(nxy,nz,cwrk) private(ij,ijnz,k) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    ijnz = ij * nz;
    for(k = 0; k < nz; k++)
      cwrk[ijnz + k] = cwrk[ijnz + k] * LOG(GRID_EPS + CREAL(cwrk[ijnz + k]));
  }

  S = -GRID_AUKB * CREAL(cgrid_integral(cworkspace));
  cworkspace->step = tmp;  
  return S;
}
