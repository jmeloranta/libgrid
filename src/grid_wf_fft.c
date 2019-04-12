/*
 * FFT-based routines for handling wavefunctions.
 *
 */

#include "grid.h"
#include "private.h"

/*
 * Calculate the momentum x component.
 * 
 * gwf        = wavefunction for the operation (wf *).
 * momentum_x = output grid containing the momentum (rgrid *).
 *
 * No return value.
 *
 * TODO: Could do just half-transform so that workspace would not have to be complex.
 *
 */

EXPORT void grid_wf_momentum_x(wf *gwf, rgrid *momentum_x) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_x(cworkspace, cworkspace);
  cgrid_inverse_fft(cworkspace);
  cgrid_multiply(cworkspace, -I * HBAR / (2.0 * gwf->mass));
  grid_complex_re_to_real(momentum_x, cworkspace);
}

/*
 * Calculate the momentum y component.
 * 
 * gwf        = wavefunction for the operation (wf *).
 * momentum_y = output grid containing the momentum (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_momentum_y(wf *gwf, rgrid *momentum_y) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_y(cworkspace, cworkspace);
  cgrid_inverse_fft(cworkspace);
  cgrid_multiply(cworkspace, -I * HBAR / (2.0 * gwf->mass));
  grid_complex_re_to_real(momentum_y, cworkspace);
}

/*
 * Calculate the probability momentum z component.
 * 
 * gwf        = wavefunction for the operation (wf *).
 * momentum_z = output grid containing the momentum (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_momentum_z(wf *gwf, rgrid *momentum_z) {

  cgrid *grid = gwf->grid, *cworkspace;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  cgrid_copy(cworkspace, grid);
  cgrid_fft(cworkspace);
  cgrid_fft_gradient_z(cworkspace, cworkspace);
  cgrid_inverse_fft(cworkspace);
  cgrid_multiply(cworkspace, -I*HBAR / (2.0 * gwf->mass));
  grid_complex_re_to_real(momentum_z, cworkspace);
}

/*
 * Calculate the momentum.
 *
 * gwf        = wavefunction for the operation (wf *).
 * momentum_x = x output grid containing the momentum (rgrid *).
 * momentum_y = y output grid containing the momentum (rgrid *).
 * momentum_z = z output grid containing the momentum (rgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_momentum(wf *gwf, rgrid *momentum_x, rgrid *momentum_y, rgrid *momentum_z) {
  
  grid_wf_momentum_x(gwf, momentum_x);
  grid_wf_momentum_y(gwf, momentum_y);
  grid_wf_momentum_z(gwf, momentum_z);
}


/*
 * Auxiliary routine for calculating the energy (FFT).
 * Users should rather call grid_wf_energy().
 *
 * gwf       = wavefunction for the energy calculation (wf *).
 * potential = Potential grid (rgrid *).
 * 
 * Returns the energy.
 *
 */

EXPORT REAL grid_wf_energy_fft(wf *gwf, rgrid *potential) {

  REAL en;

  en = grid_wf_kinetic_energy_fft(gwf);
  if(potential) en += grid_wf_potential_energy(gwf, potential);
  return en;
}

/*
 * Auxiliary routine for calculating kinetic energy (FFT).
 * This is used by grid_wf_energy_fft().
 * 
 * gwf       = wavefunction for the kinetic energy calculation (wf *).
 *
 * Returns the kinetic energy.
 *
 * NOTE: The moving background contribution is subtracted off (laplace expectation value routine includes it!).
 *
 */

EXPORT REAL grid_wf_kinetic_energy_fft(wf *gwf) {

  cgrid *cworkspace, *grid = gwf->grid;
  REAL mass = gwf->mass, kx = gwf->grid->kx0, ky = gwf->grid->ky0, kz = gwf->grid->kz0;
  REAL ekin = -HBAR * HBAR * (kx * kx + ky * ky + kz * kz) / (2.0 * mass);

  if(ekin != 0.0) ekin *= CREAL(cgrid_integral_of_square(gwf->grid));  // Remove moving background contribution

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  /* delta (- k^2) fft[f(x)] / N */
  cgrid_copy(cworkspace, gwf->grid);
  cgrid_fft(cworkspace);
  
  return -HBAR * HBAR / (2.0 * gwf->mass) * cgrid_fft_laplace_expectation_value(cworkspace, cworkspace) + ekin;
}

/*
 * Auxiliary routine to propagate kinetic energy using FFT.
 *
 * gwf  = wavefunction to be propagated (wf *).
 * time = time step (REAL complex).
 *
 * TODO: Dirichlet BC could be implemented using sin transformation and Neumann by cos transformation. This
 * would eliminate the current CN based routines completely. The FFT methods have the best accuracy (?).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_kinetic_fft(wf *gwf, REAL complex time) {

  INT i, j, k, ij, ijnz, nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz, nxy = nx * ny, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL kx, ky, kz, lx, ly, lz, step = gwf->grid->step, norm;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL complex *value = gwf->grid->value, time_mass = -I * time * HBAR / (gwf->mass * 2.0);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_fft(gwf, time_mass)) return;
#endif
  
  cgrid_fftw(gwf->grid);

  /* f(x) = ifft[fft[f(x)]] / N */
  norm = gwf->grid->fft_norm;

  if(gwf->boundary == WF_NEUMANN_BOUNDARY  ||
     gwf->boundary == WF_VORTEX_X_BOUNDARY ||
     gwf->boundary == WF_VORTEX_Y_BOUNDARY ||
     gwf->boundary == WF_VORTEX_Z_BOUNDARY) {
    lx = M_PI / (((REAL) nx) * step);
    ly = M_PI / (((REAL) ny) * step);
    lz = M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,lx,ly,lz,step,value,time_mass,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      kx = ((REAL) i) * lx - kx0;
      ky = ((REAL) j) * ly - ky0;
      
      for(k = 0; k < nz; k++) {
        kz = ((REAL) k) * lz - kz0;
        
        /* psi(k, t+dt) = psi(k, t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */	  
        value[ijnz + k] *= norm * CEXP(time_mass * (kx * kx + ky * ky + kz * kz));
      }
    }
  } else {
    lx = 2.0 * M_PI / (((REAL) nx) * step);
    ly = 2.0 * M_PI / (((REAL) ny) * step);
    lz = 2.0 * M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(lx,ly,lz,norm,nx,ny,nz,nx2,ny2,nz2,nxy,step,value,time_mass,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * if i <= N/2, k = 2pi i / L
       * else k = 2pi (i - N) / L
       *
       * 2nd derivative (laplacian):
       * multiply by -k^2
       *
       * first derivative (not used here):
       * multiply by I * k and by zero for i = n/2.
       *
       */

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
        
        /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
        value[ijnz + k] *= norm * CEXP(time_mass * (kx * kx + ky * ky + kz * kz));
      }
    } 
  }
  
  cgrid_fftw_inv(gwf->grid);
}

/*
 * Auxiliary routine to propagate kinetic energy using CFFT (Cayley's form).
 *
 * gwf  = wavefunction to be propagated (wf *).
 * time = time step (REAL complex).
 *
 * TODO: Dirichlet BC could be implemented using sin transformation and Neumann by cos transformation. This
 * would eliminate the current CN based routines completely. The FFT methods have the best accuracy (?).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_propagate_kinetic_cfft(wf *gwf, REAL complex time) {

  INT i, j, k, ij, ijnz, nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz, nxy = nx * ny, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL kx, ky, kz, lx, ly, lz, step = gwf->grid->step, norm;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL complex *value = gwf->grid->value, time_mass = -I * time * HBAR / (gwf->mass * 2.0), tmp;

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cfft(gwf, time_mass)) return;
#endif
  
  cgrid_fftw(gwf->grid);

  /* f(x) = ifft[fft[f(x)]] / N */
  norm = gwf->grid->fft_norm;

  if(gwf->boundary == WF_NEUMANN_BOUNDARY  ||
     gwf->boundary == WF_VORTEX_X_BOUNDARY ||
     gwf->boundary == WF_VORTEX_Y_BOUNDARY ||
     gwf->boundary == WF_VORTEX_Z_BOUNDARY) {
    lx = M_PI / (((REAL) nx) * step);
    ly = M_PI / (((REAL) ny) * step);
    lz = M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(norm,nx,ny,nz,nxy,lx,ly,lz,step,value,time_mass,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,tmp) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      kx = ((REAL) i) * lx - kx0;
      ky = ((REAL) j) * ly - ky0;
      
      for(k = 0; k < nz; k++) {
        kz = ((REAL) k) * lz - kz0;
        
        /* psi(k, t+dt) = psi(k, t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */	  
        /* exp ~ (1 + 0.5 * x) / (1 - 0.5 * x) */
        tmp = 0.5 * time_mass * (kx * kx + ky * ky + kz * kz);
        value[ijnz + k] *= norm * (1.0 + tmp) / (1.0 - tmp);
      }
    }
  } else {
    lx = 2.0 * M_PI / (((REAL) nx) * step);
    ly = 2.0 * M_PI / (((REAL) ny) * step);
    lz = 2.0 * M_PI / (((REAL) nz) * step);
#pragma omp parallel for firstprivate(lx,ly,lz,norm,nx,ny,nz,nx2,ny2,nz2,nxy,step,value,time_mass,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz,tmp) default(none) schedule(runtime)
    for(ij = 0; ij < nxy; ij++) {
      i = ij / ny;
      j = ij % ny;
      ijnz = ij * nz;
      
      /* 
       * if i <= N/2, k = 2pi i / L
       * else k = 2pi (i - N) / L
       *
       * 2nd derivative (laplacian):
       * multiply by -k^2
       *
       * first derivative (not used here):
       * multiply by I * k and by zero for i = n/2.
       *
       */

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
        
        /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
        /* exp ~ (1 + 0.5 * x) / (1 - 0.5 * x) */
        tmp = 0.5 * time_mass * (kx * kx + ky * ky + kz * kz);
        value[ijnz + k] *= norm * (1.0 + tmp) / (1.0 - tmp);
      }
    } 
  }
  
  cgrid_fftw_inv(gwf->grid);
}

/*
 * Calculate square of potential gradient.
 *
 * gwf         = wavefunction (wf *).
 * sq_grad_pot = output grid (cgrid *).
 * potential   = potental input grid (cgrid *).
 *
 * No return value.
 *
 */

EXPORT void grid_wf_square_of_potential_gradient(wf *gwf, cgrid *sq_grad_pot, cgrid *potential) {

  cgrid *cworkspace, *cworkspace2, *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace2 = gwf->cworkspace2;

  cgrid_copy(sq_grad_pot, potential);
  cgrid_fft(sq_grad_pot);
  cgrid_fft_gradient(sq_grad_pot, sq_grad_pot, cworkspace, cworkspace2);
  
  cgrid_inverse_fft(sq_grad_pot);
  cgrid_inverse_fft(cworkspace);
  cgrid_inverse_fft(cworkspace2);
  
  cgrid_conjugate_product(sq_grad_pot, sq_grad_pot, sq_grad_pot);
  cgrid_conjugate_product(cworkspace, cworkspace, cworkspace);
  cgrid_conjugate_product(cworkspace2, cworkspace2, cworkspace2);
  
  cgrid_sum(cworkspace, cworkspace, cworkspace2);
  cgrid_sum(sq_grad_pot, sq_grad_pot, cworkspace);
}
