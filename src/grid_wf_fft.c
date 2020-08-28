/*
 * FFT-based routines for handling wavefunctions.
 *
 */

#include "grid.h"

extern char grid_analyze_method;

/*
 * @FUNC{grid_wf_momentum_x, "Momentum x component of wavefunction"}
 * @DESC{"Calculate the momentum x component: $-i\hbar d/dx$"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{cgrid *momentum_x, "Output grid containing the momentum"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_momentum_x(wf *gwf, cgrid *momentum_x) {

  if(grid_analyze_method) {
    cgrid_copy(momentum_x, gwf->grid);
    cgrid_fft(momentum_x);
    cgrid_fft_gradient_x(momentum_x, momentum_x);
    cgrid_inverse_fft_norm(momentum_x);
  } else cgrid_fd_gradient_x(gwf->grid, momentum_x);

  cgrid_multiply(momentum_x, -I * HBAR);
}

/*
 * @FUNC{grid_wf_momentum_y, "Momentum y component of wavefunction"}
 * @DESC{"Calculate the momentum y component: $-i\hbar d/dy$"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{cgrid *momentum_y, "Output grid containing the momentum"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_momentum_y(wf *gwf, cgrid *momentum_y) {

  if(grid_analyze_method) {
    cgrid_copy(momentum_y, gwf->grid);
    cgrid_fft(momentum_y);
    cgrid_fft_gradient_y(momentum_y, momentum_y);
    cgrid_inverse_fft_norm(momentum_y);
  } else cgrid_fd_gradient_x(gwf->grid, momentum_y);

  cgrid_multiply(momentum_y, -I * HBAR);
}

/*
 * @FUNC{grid_wf_momentum_z, "Momentum z component of wavefunction"}
 * @DESC{"Calculate the momentum z component: $-i\hbar d/dz$"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{cgrid *momentum_z, "Output grid containing the momentum"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_momentum_z(wf *gwf, cgrid *momentum_z) {

  if(grid_analyze_method) {
    cgrid_copy(momentum_z, gwf->grid);
    cgrid_fft(momentum_z);
    cgrid_fft_gradient_z(momentum_z, momentum_z);
    cgrid_inverse_fft_norm(momentum_z);
  } else cgrid_fd_gradient_z(gwf->grid, momentum_z);

  cgrid_multiply(momentum_z, -I * HBAR);
}

/*
 * @FUNC{grid_wf_momentum, "Momentum of wavefunction"}
 * @DESC{"Calculate the momentum: $-i\hbar (d/dx, d/dy, d/dz)$"}
 * @ARG1{wf *gwf, "Wavefunction for the operation"}
 * @ARG2{cgrid *momentum_x, "Output grid containing the momentum x"}
 * @ARG3{cgrid *momentum_y, "Output grid containing the momentum y"}
 * @ARG4{cgrid *momentum_z, "Output grid containing the momentum z"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_momentum(wf *gwf, cgrid *momentum_x, cgrid *momentum_y, cgrid *momentum_z) {
  
  grid_wf_momentum_x(gwf, momentum_x);
  grid_wf_momentum_y(gwf, momentum_y);
  grid_wf_momentum_z(gwf, momentum_z);
}

/*
 * @FUNC{grid_wf_energy_fft, "Energy of wavefunction (FFT)"}
 * @DESC{"Auxiliary routine for calculating the energy (FFT).
          Users should rather call grid_wf_energy()"}
 * @ARG1{wf *gwf, "Wavefunction for the energy calculation"}
 * @ARG2{rgrid *potential, "Potential energy grid"}
 * @RVAL{REAL, "Returns the energy"}
 *
 */

EXPORT REAL grid_wf_energy_fft(wf *gwf, rgrid *potential) {

  REAL en;

  en = grid_wf_kinetic_energy_fft(gwf);
  if(potential) en += grid_wf_potential_energy(gwf, potential);
  return en;
}

/*
 * @FUNC{grid_wf_kinetic_energy_fft, "Kinetic energy of wavefunction (FFT)"}
 * @DESC{"Auxiliary routine for calculating kinetic energy (FFT). 
          This is used by grid_wf_energy_fft()"}
 * @ARG1{wf *gwf, "Wavefunction for the kinetic energy calculation"}
 * @RVAL{REAL, "Returns the kinetic energy"}
 *
 */

EXPORT REAL grid_wf_kinetic_energy_fft(wf *gwf) {

  cgrid *grid = gwf->grid;
  REAL tmp;

  /* delta (- k^2) fft[f(x)] / N */
  cgrid_fft(grid);  
  tmp = -HBAR * HBAR / (2.0 * gwf->mass) * cgrid_fft_laplace_expectation_value(grid);
  cgrid_inverse_fft_norm(grid);
  return tmp;
}

/*
 * @FUNC{grid_wf_propagate_kinetic_fft, "Propagate kinetic portion of wavefunction (FFT)"}
 * @DESC{"Auxiliary routine to propagate kinetic energy using FFT. 
          The wave function must be in reciprocal space"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex time, "Time step"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_kinetic_fft(wf *gwf, REAL complex time) {

  INT i, j, k, ij, ijnz, nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz, nxy = nx * ny, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL kx, ky, kz, lx, ly, lz, step = gwf->grid->step;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0;
  REAL complex *value = gwf->grid->value, time_mass = -I * time * HBAR / (gwf->mass * 2.0);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_fft(gwf, time_mass)) return;
#endif
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
#pragma omp parallel for firstprivate(lx,ly,lz,nx,ny,nz,nx2,ny2,nz2,nxy,step,value,time_mass,kx0,ky0,kz0) private(i,j,ij,ijnz,k,kx,ky,kz) default(none) schedule(runtime)
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
      value[ijnz + k] *= CEXP(time_mass * (kx * kx + ky * ky + kz * kz));
    }
  } 
}

/*
 * @FUNC{grid_wf_propagate_kinetic_cfft, "Propagate kinetic portion of wavefunction (CFFT)"}
 * @DESC{"Auxiliary routine to propagate kinetic energy using FFT with high wavenumber sink (cutoff specified by gwf$->$kmax).
          The wave function must be in reciprocal space.
 *        This can alleviate problems related to FFT dealiasing and numerical noise that is amplified by the operator splitting scheme"}
 * @ARG1{wf *gwf, "Wavefunction to be propagated"}
 * @ARG2{REAL complex time, "Time step"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void grid_wf_propagate_kinetic_cfft(wf *gwf, REAL complex time) {

  INT i, j, k, ij, ijnz, nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz, nxy = nx * ny, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL kx, ky, kz, lx, ly, lz, step = gwf->grid->step;
  REAL kx0 = gwf->grid->kx0, ky0 = gwf->grid->ky0, kz0 = gwf->grid->kz0, tot, kmax = gwf->kmax;
  REAL complex *value = gwf->grid->value, time_mass = -I * time * HBAR / (gwf->mass * 2.0);

#ifdef USE_CUDA
  if(cuda_status() && !grid_cuda_wf_propagate_kinetic_cfft(gwf, time_mass, kmax)) return;
#endif
  
  lx = 2.0 * M_PI / (step * (REAL) nx);
  ly = 2.0 * M_PI / (step * (REAL) ny);
  lz = 2.0 * M_PI / (step * (REAL) nz);
#pragma omp parallel for firstprivate(lx,ly,lz,nx,ny,nz,nx2,ny2,nz2,nxy,step,value,time_mass,kx0,ky0,kz0,kmax) private(i,j,ij,ijnz,k,kx,ky,kz,tot) default(none) schedule(runtime)
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

      tot = kx * kx + ky * ky + kz * kz;
      if(tot < kmax)
        /* psi(k,t+dt) = psi(k,t) exp( - i (hbar^2 * k^2 / 2m) dt / hbar ) */
        value[ijnz + k] *= CEXP(time_mass * tot);
      else
        value[ijnz + k] = 0.0;
    }
  } 
}

/*
 * @FUNC{grid_wf_square_of_potential_gradient, "Square of potential gradient of wavefunction"}
 * @DESC{"Calculate square of potential gradient: $|\nabla V|^2 = |(dV/dx)|^2 + |(dV/dy)|^2 + |(dV/dz)|^2$"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{cgrid *sq_grad_pot, "Output grid"}
 * @ARG3{cgrid *potential, "Potental input grid"}
 * @RVAL{void, "No return value"}
 *
 * TODO: Write a specific routine for this so that cworkspaces can be eliminate completely.
 *
 */

EXPORT void grid_wf_square_of_potential_gradient(wf *gwf, cgrid *sq_grad_pot, cgrid *potential) {

  cgrid *cworkspace, *cworkspace2, *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");
  cworkspace = gwf->cworkspace;
  if(!gwf->cworkspace2) gwf->cworkspace2 = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace 2");
  cworkspace2 = gwf->cworkspace2;

  cgrid_copy(sq_grad_pot, potential);
  if(grid_analyze_method) {
    cgrid_fft(sq_grad_pot);
    cgrid_fft_gradient_x(sq_grad_pot, cworkspace);
    cgrid_fft_gradient_y(sq_grad_pot, cworkspace2);
    cgrid_fft_gradient_z(sq_grad_pot, sq_grad_pot);  
    cgrid_inverse_fft_norm(sq_grad_pot);
    cgrid_inverse_fft_norm(cworkspace);
    cgrid_inverse_fft_norm(cworkspace2);
  } else {
    cgrid_fd_gradient_x(sq_grad_pot, cworkspace);
    cgrid_fd_gradient_y(sq_grad_pot, cworkspace2);
    cgrid_fd_gradient_z(sq_grad_pot, sq_grad_pot);
  }
  
  cgrid_conjugate_product(sq_grad_pot, sq_grad_pot, sq_grad_pot);
  cgrid_conjugate_product(cworkspace, cworkspace, cworkspace);
  cgrid_conjugate_product(cworkspace2, cworkspace2, cworkspace2);
  
  cgrid_sum(cworkspace, cworkspace, cworkspace2);
  cgrid_sum(sq_grad_pot, sq_grad_pot, cworkspace);
}
