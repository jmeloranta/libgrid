/*
 * CUDA device code (REAL complex; cgrid).
 *
 * TODO: Should we use to BC routines rather than hard code the BCs below? (efficiency?)
 *
 */

#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "defs.h"
#include "linalg-cuda.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * grid_wf_absorb cuda equivalent.
 *
 * i         = Index i (x) (INT; input).
 * j         = Index j (y) (INT; input).
 * k         = Index k (z) (INT; input).
 * amp       = Ampltude for absorption (REAL; input).
 * lx        = Lower limit index for i (x) (INT; input).
 * hx        = Upper limit index for i (x) (INT; input).
 * ly        = Lower limit index for j (y) (INT; input).
 * hy        = Upper limit index for j (y) (INT; input).
 * lz        = Lower limit index for k (z) (INT; input).
 * hz        = Upper limit index for k (z) (INT; input).
 * time_step = Time step (REAL complex; input).
 * 
 * Returns the (complex) time step to be applied.
 *
 */

__device__ CUCOMPLEX grid_cuda_wf_absorb(INT i, INT j, INT k, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, CUCOMPLEX time_step) {

  CUCOMPLEX t;

  if(i >= lx && i <= hx && j >= ly && j <= hy && k >= lz && k <= hz) return time_step;

  t.x = 1.0; t.y = 0.0;

  if(i < lx) t.y -= ((CUREAL) (lx - i)) / (3.0 * (CUREAL) lx);
  else if(i > hx) t.y -= ((CUREAL) (i - hx)) / (3.0 * (CUREAL) lx);

  if(j < ly) t.y -= ((CUREAL) (ly - j)) / (3.0 * (CUREAL) ly);
  else if(j > hy) t.y -= ((CUREAL) (j - hy)) / (3.0 * (CUREAL) ly);

  if(k < lz) t.y -= ((CUREAL) (lz - k)) / (3.0 * (CUREAL) lz);
  else if(k > hz) t.y -= ((CUREAL) (k - hz)) / (3.0 * (CUREAL) lz);

  t.y *= amp;
  return t * time_step;
}

/*
 * Propagate wf using CN along X.
 *
 * if amp == 0.0, use constant time step given by tstep.
 * else call grid_wf_absorb().
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_x_gpu(INT nx, INT ny, INT nz, INT nyz, INT ny2, CUCOMPLEX *psi, char bc, CUCOMPLEX *pot, CUCOMPLEX *wrk, CUCOMPLEX c, CUCOMPLEX cb, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL y0, CUCOMPLEX tstep, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUREAL y;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(j >= ny || k >= nz) return;

  tid = j * nz + k;
  d = &wrk[nx * (3 * tid + 0)];
  b = &wrk[nx * (3 * tid + 1)];
  pwrk = &wrk[nx * (3 * tid + 2)];
  y = ((REAL) (j - ny2)) * step - y0;    

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(i = 1; i < nx - 1; i++) {
    if(amp != 0.0) tim = grid_cuda_wf_absorb(i, j, k, amp, lx, hx, ly, hy, lz, hz, tstep);
    else tim = tstep;
    cp = c / tim;
    ind = i * nyz + tid;
    /* Left-hand side (+) */
    /* (C + dx^2 laplace + CB V + C2 grad + C3 y grad) -- only C and laplace have diag elements */
    d[i] = cp - 2.0;  // -2 from Laplacian, C = cp = c / dt
    if(pot) d[i] = d[i] + cb * pot[ind];
    /* Right-hand side (-) */
    /* (C - dy^2 laplace - CB V - C2 grad - C3 y grad)psi */
    b[i] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind - nyz]) 
             - c2 * (psi[ind + nyz] - psi[ind - nyz]) - c3 * y * (psi[ind + nyz] - psi[ind - nyz]);
    if(pot) b[i] = b[i] - cb * pot[ind] * psi[ind];
  }

  // Boundary conditions
  ind = j * nz + k; // i = 0 - left boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(0, j, k, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  /* Right-hand side (-) */
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind - nyz] = 0
      b[0] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind]) - c2 * psi[ind + nyz] - c3 * y * psi[ind + nyz];
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind - nyz] = psi[ind + nyz]
      b[0] = cp * psi[ind] - (2.0 * psi[ind + nyz] - 2.0 * psi[ind]);
    break;
    case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind - nyz] = psi[ind + (nx-1)*nyz]
      b[0] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind + (nx-1)*nyz])
             - c2 * (psi[ind + nyz] - psi[ind + (nx-1) * nyz]) - c3 * y * (psi[ind + nyz] - psi[ind + (nx-1) * nyz]);
    break;
  }
  d[0] = cp - 2.0;  // LHS: -2 diag elem from Laplacian (x C)
  if(pot) {
    b[0] = b[0] - cb * pot[ind] * psi[ind];   // RHS(-)
    d[0] = d[0] + cb * pot[ind];              // LHS(+)
  }

  ind = (nx - 1) * nyz + j * nz + k;  // i = nx - 1, right boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(nx-1, j, k, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  /* Right-hand side (-) */
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind + nyz] = 0
      b[nx-1] = cp * psi[ind] - (psi[ind - nyz] - 2.0 * psi[ind]) + c2 * psi[ind - nyz] + c3 * y * psi[ind - nyz];  // c2,c3: - from difference
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind + nyz] = psi[ind - nyz]
      b[nx-1] = cp * psi[ind] - (2.0 * psi[ind - nyz] - 2.0 * psi[ind]);
    break;
    case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind + nyz] = psi[ind - (nx-1)*nyz]
      b[nx-1] = cp * psi[ind] - (psi[ind - nyz] - 2.0 * psi[ind] + psi[ind - (nx-1) * nyz])
             - c2 * (psi[ind - (nx-1) * nyz] - psi[ind - nyz]) - c3 * y * (psi[ind - (nx-1) * nyz] - psi[ind - nyz]);
    break;
  }      
  d[nx-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt
  if(pot) {
    b[nx-1] = b[nx-1] - cb * pot[ind] * psi[ind]; // RHS(-)
    d[nx-1] = d[nx-1] + cb * pot[ind];            // LHS(+)
  }

  if(bc == WF_PERIODIC_BOUNDARY)
    grid_cuda_solve_tridiagonal_system_cyclic2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz, pwrk);
  else
    grid_cuda_solve_tridiagonal_system2(nx, d, b, &psi[j * nz + k], c2 + c3 * y, nyz);
}

/*
 * Propagate wavefunction using Crank-Nicolson.
 *
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 * tstep      = time step length (REAL; input).
 * wf         = Source/destination grid for operation (CUCOMPLEX *; input).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * pot        = external potential (CUCOMPLEX *; input). May be NULL.
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kx0        = base momentum along x (REAL; input).
 * omega      = rotation freq (REAL; input).
 * y0         = y0 grid spatial offset (REAL; input).
 * wrk        = Workspace (CUCOMPLEX *; input).
 * amp        = Absorbing amplitude (REAL; input). If equal to 0.0, constant time step is used.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_xW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, CUCOMPLEX *gwf, char bc, CUCOMPLEX *pot, REAL mass, REAL step, REAL kx0, REAL omega, REAL y0, CUCOMPLEX *wrk, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, cb, c2, c3;
  INT nyz = ny * nz, ny2 = ny / 2;

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t) <=> A x = b
   * (C + dx^2 laplace + CB V + C2 grad + C3 y grad) psi(t+dt) 
   *         = (C - dx^2 laplace - CB V - C2 grad - C3 y grad) psi(t)
   * where C = 4 i m dx^2 / (hbar dt), CB = -2m dx^2 / hbar^2, C2 = -i dx kx, C3 = m \omega i dx / hbar
   *
   */
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  cb = CUMAKE(-2.0 * mass * step * step / (HBAR * HBAR), 0.0);  // diag element coefficient for the potential
  c2 = CUMAKE(0.0, -step * kx0); // coeff for moving background
  c3 = CUMAKE(0.0, mass * omega * step / HBAR); // coeff for rotating liquid around Z

  grid_cuda_wf_propagate_kinetic_cn_x_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, ny2, gwf, bc, pot, wrk, c, cb, c2, c3, step, y0, tstep, amp, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}

/*
 * Propagate wf using CN along Y.
 *
 * if time == 0, use constant time step given by tstep.
 * else call 
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_y_gpu(INT nx, INT ny, INT nz, INT nyz, INT nx2, CUCOMPLEX *psi, char bc, CUCOMPLEX *pot, CUCOMPLEX *wrk, CUCOMPLEX c, CUCOMPLEX cb, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL x0, CUCOMPLEX tstep, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUREAL x;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || k >= nz) return;

  x = ((REAL) (i - nx2)) * step - x0;    
  tid = i * nz + k;
  d = &wrk[ny * (3 * tid + 0)];
  b = &wrk[ny * (3 * tid + 1)];
  pwrk = &wrk[ny * (3 * tid + 2)];

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(j = 1; j < ny - 1; j++) {
    if(amp != 0.0) tim = grid_cuda_wf_absorb(i, j, k, amp, lx, hx, ly, hy, lz, hz, tstep);
    else tim = tstep;
    cp = c / tim;
    ind = i * nyz + j * nz + k;
    /* Left-hand side (+) */
    /* (C + dy^2 laplace + CB V + C2 grad + C3 x grad) */
    d[j] = cp - 2.0; // -2 from Laplacian, cp = c / dt, LHS(+)
    if(pot) d[j] = d[j] + cb * pot[ind];
    /* Right-hand side (-) */
    /* (C - dx^2 laplace - CB V - C2 grad - C3 x grad) */
    b[j] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind - nz]) 
           - c2 * (psi[ind + nz] - psi[ind - nz]) - c3 * x * (psi[ind + nz] - psi[ind - nz]);
    if(pot) b[j] = b[j] - cb * pot[ind] * psi[ind];
  }
  
  // Boundary conditions
 
  ind = i * nyz + k; // j = 0 - left boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(i, 0, k, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  /* Right-hand side (-) */
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind - nz] = 0
      b[0] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind]) - c2 * psi[ind + nz] - c3 * x * psi[ind + nz];
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind - nz] = psi[ind + nz]
      b[0] = cp * psi[ind] - (2.0 * psi[ind + nz] - 2.0 * psi[ind]);
    break;
   case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind - nz] = psi[ind + (ny-1)*nz]
      b[0] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind + (ny-1)*nz])
             - c2 * (psi[ind + nz] - psi[ind + (ny-1) * nz]) - c3 * x * (psi[ind + nz] - psi[ind + (ny-1) * nz]);
   break;
  }
  d[0] = cp - 2.0;  // -2 from Laplacian, cp = c / dt
  if(pot) {
    b[0] = b[0] - cb * pot[ind] * psi[ind];  // RHS(-)
    d[0] = d[0] + cb * pot[ind];             // LHS(+)
  }

  ind = i * nyz + (ny-1) * nz + k;  // j = ny - 1 - right boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(i, ny-1, k, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  /* Right-hand side (-) */
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind + nz] = 0
      b[ny-1] = cp * psi[ind] - (psi[ind - nz] - 2.0 * psi[ind]) + c2 * psi[ind - nz] + c3 * x * psi[ind - nz];
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind + nz] = psi[ind - nz]
      b[ny-1] = cp * psi[ind] - (2.0 * psi[ind - nz] - 2.0 * psi[ind]);
    break;
    case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind + nz] = psi[ind - (ny-1)*nz]
      b[ny-1] = cp * psi[ind] - (psi[ind - nz] - 2.0 * psi[ind] + psi[ind - (ny-1)*nz])
             - c2 * (psi[ind - (ny-1) * nz] - psi[ind - nz]) - c3 * x * (psi[ind - (ny-1) * nz] - psi[ind - nz]);
    break;
  }
  d[ny-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt
  if(pot) {
    b[ny-1] = b[ny-1] - cb * pot[ind] * psi[ind]; // RHS(-)
    d[ny-1] = d[ny-1] + cb * pot[ind];            // LHS(+)
  }

  if(bc == WF_PERIODIC_BOUNDARY)
    grid_cuda_solve_tridiagonal_system_cyclic2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz, pwrk);
  else
    grid_cuda_solve_tridiagonal_system2(ny, d, b, &psi[i * nyz + k], c2 + c3 * x, nz);
}

/*
 * Propagate wavefunction using Crank-Nicolson (Y).
 *
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 * tstep      = time step length (REAL; input).
 * wf         = Source/destination grid for operation (CUCOMPLEX *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * pot        = external potential (CUCOMPLEX *; input). May be NULL.
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * ky0        = base momentum along y (REAL; input).
 * omega      = rotation freq (REAL; input).
 * x0         = x0 grid spatial offset (REAL; input).
 * wrk        = Workspace (CUCOMPLEX *; scratch space).
 * amp        = Absorbing amplitude (REAL; input). If equal to 0.0, constant time step is used.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_yW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, CUCOMPLEX *gwf, char bc, CUCOMPLEX *pot, REAL mass, REAL step, REAL ky0, REAL omega, REAL x0, CUCOMPLEX *wrk, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, cb, c2, c3;
  INT nyz = ny * nz, nx2 = nx / 2;

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t) <=> A x = b
   * (C + dy^2 laplace + CB V + C2 grad + C3 x grad) psi(t+dt) 
   *         = (C - dx^2 laplace - CB V - C2 grad - C3 x grad) psi(t)
   * where C = 4 i m dy^2 / (hbar dt), CB = -2m dy^2 / hbar^2, C2 = -i dy ky, C3 = -m \omega i dy / hbar
   *
   */
 
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  cb = CUMAKE(-2.0 * mass * step * step / (HBAR * HBAR), 0.0);  // diag element coefficient for the potential
  c2 = CUMAKE(0.0, -step * ky0); // coeff for moving background
  c3 = CUMAKE(0.0, -mass * omega * step / HBAR); // coeff for rotating liquid around Z

  grid_cuda_wf_propagate_kinetic_cn_y_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nx2, gwf, bc, pot, wrk, c, cb, c2, c3, step, x0, tstep, amp, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}

/*
 * Propagate wf using CN along Z.
 *
 * if time == 0, use constant time step given by tstep.
 * else call 
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_z_gpu(INT nx, INT ny, INT nz, INT nyz, INT nxy, CUCOMPLEX *psi, char bc, CUCOMPLEX *pot, CUCOMPLEX *wrk, CUCOMPLEX c, CUCOMPLEX cb, CUCOMPLEX c2, CUREAL step, CUCOMPLEX tstep, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k, j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || j >= ny) return;

  tid = i * ny + j;
  d = &wrk[nz * (3 * tid + 0)];
  b = &wrk[nz * (3 * tid + 1)];
  pwrk = &wrk[nz * (3 * tid + 2)];

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(k = 1; k < nz - 1; k++) {
    if(amp != 0.0) tim = grid_cuda_wf_absorb(i, j, k, amp, lx, hx, ly, hy, lz, hz, tstep);
    else tim = tstep;
    cp = c / tim;
    ind = i * nyz + j * nz + k;
    /* Left-hand side (+) */
    /* (C + dz^2 laplace + CB V + C2 grad) */
    d[k] = cp - 2.0; // Diagonal: -2 from Laplacian, cp = c / dt
    if(pot) d[k] = d[k] + cb * pot[ind]; // add possible potential to diagonal
    /* Right-hand side (-) */
    /* (C - dz^2 laplace - CB V - C2 grad) */
    b[k] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind - 1]) 
           - c2 * (psi[ind + 1] - psi[ind - 1]); // NOTE: No rotation term as the rotation is about the z-axis
    if(pot) b[k] = b[k] - cb * pot[ind] * psi[ind];
  }

  // Boundary conditions
 
  ind = i * nyz + j * nz; // k = 0 - left boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(i, j, 0, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  /* Right-hand side (-) */
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind - 1] = 0
      b[0] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind]) - c2 * psi[ind + 1];
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind - 1] = psi[ind + 1]
      b[0] = cp * psi[ind] - (2.0 * psi[ind + 1] - 2.0 * psi[ind]);
    break;
   case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind - 1] = psi[ind + (nz-1)]
      b[0] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind + (nz-1)])
             - c2 * (psi[ind + 1] - psi[ind + (nz-1)]);
    break;
  }
  d[0] = cp - 2.0;  // -2 from Laplacian, cp = c / dt
  if(pot) {
    b[0] = b[0] - cb * pot[ind] * psi[ind];  // RHS(-)
    d[0] = d[0] + cb * pot[ind];             // LHS(+)
  }

  ind = i * nyz + j * nz + (nz - 1);  // k = nz-1 - right boundary
  if(amp != 0.0) tim = grid_cuda_wf_absorb(i, j, nz-1, amp, lx, hx, ly, hy, lz, hz, tstep);
  else tim = tstep;
  cp = c / tim;
  switch(bc) {
    case WF_DIRICHLET_BOUNDARY:
      // Dirichlet: psi[ind + 1] = 0
      b[nz-1] = cp * psi[ind] - (psi[ind - 1] - 2.0 * psi[ind]) + c2 * psi[ind - 1];
    break;
    case WF_NEUMANN_BOUNDARY:
      // Neumann: psi[ind + 1] = psi[ind - 1]
      b[nz-1] = cp * psi[ind] - (2.0 * psi[ind - 1] - 2.0 * psi[ind]);
    break;
    case WF_PERIODIC_BOUNDARY:
      // Periodic: psi[ind + 1] = psi[ind - (nz-1)]
      b[nz-1] = cp * psi[ind] - (psi[ind - 1] - 2.0 * psi[ind] + psi[ind - (nz-1)])
             - c2 * (psi[ind - (nz-1)] - psi[ind - 1]);
    break;
  }      
  d[nz-1] = cp - 2.0;  // -2 from Laplacian, cp = c / dt
  if(pot) {
    b[nz-1] = b[nz-1] - cb * pot[ind] * psi[ind]; // RHS(-)
    d[nz-1] = d[nz-1] + cb * pot[ind];            // LHS(+)
  }

  if(bc == WF_PERIODIC_BOUNDARY)
    grid_cuda_solve_tridiagonal_system_cyclic2(nz, d, b, &psi[i * nyz + j * nz], c2, 1, pwrk);
  else
    grid_cuda_solve_tridiagonal_system2(nz, d, b, &psi[i * nyz + j * nz], c2, 1);
}

/*
 * Propagate wavefunction using Crank-Nicolson (Z).
 *
 * nx         = # of points along x (INT; input).
 * ny         = # of points along y (INT; input).
 * nz         = # of points along z (INT; input).
 * tstep      = time step length (REAL; input).
 * wf         = Source/destination grid for operation (CUCOMPLEX *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * pot        = external potential (CUCOMPLEX *; input). May be NULL.
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kz0        = base momentum along z (REAL; input).
 * wrk        = Workspace (CUCOMPLEX *; scratch space).
 * amp        = Absorbing amplitude (REAL; input).  If equal to 0.0, constant time step is used.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_zW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, CUCOMPLEX *gwf, char bc, CUCOMPLEX *pot, REAL mass, REAL step, REAL kz0, CUCOMPLEX *wrk, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, cb, c2;
  INT nyz = ny * nz, nxy = nx * ny;

  /*
   * (1 + .5 i (T + V - v0 pz) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - v0 pz) dt / hbar) psi(t) <=> A x = b
   * (C + dz^2 laplace + CB V + C2 grad) psi(t+dt) 
   *         = (C - dz^2 laplace - CB V - C2 grad) psi(t)
   * where C = 4 i m dz^2 / (hbar dt), CB = -2m dz^2 / hbar^2, C2 = -i dz kz.
   *
   */
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  cb = CUMAKE(-2.0 * mass * step * step / (HBAR * HBAR), 0.0);  // diag element coefficient for the potential
  c2 = CUMAKE(0.0, -step * kz0); // coeff for moving background

  grid_cuda_wf_propagate_kinetic_cn_z_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nxy, gwf, bc, pot, wrk, c, cb, c2, step, tstep, amp, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}

/*
 * Potential energy propagation in real space (possibly with absorbing boundaries).
 *
 */

__global__ void grid_cuda_wf_propagate_potential_gpu(CUCOMPLEX *b, CUCOMPLEX *pot, CUCOMPLEX time_step, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {  /* Exectutes at GPU */

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, i = blockIdx.z * blockDim.z + threadIdx.z, idx;
  CUCOMPLEX c;

  if(i >= nx || j >= ny || k >= nz) return;

  idx = (i * ny + j) * nz + k;
  if(amp == 0.0) 
    c = CUMAKE(0.0, -1.0 / HBAR) * time_step;
  else
    c = CUMAKE(0.0, -1.0 / HBAR) * grid_cuda_wf_absorb(i, j, k, amp, lx, hx, ly, hy, lz, hz, time_step);
  b[idx] = b[idx] * CUCEXP(c * pot[idx]);
}

/*
 * Propagate potential energy in real space with absorbing boundaries.
 *
 * wf       = Source/destination grid for operation (REAL complex *; input/output).
 * pot      = Potential grid (CUCOMPLEX *; input).
 * time_step= Time step length (CUCOMPLEX; input).
 * amp      = Max amplitude for imag. part (CUREAL; input).
 * lx       = Lower bound for absorbing bc (INT; input).
 * hx       = Upper bound for absorbing bc (INT; input).
 * ly       = Lower bound for absorbing bc (INT; input).
 * hy       = Upper bound for absorbing bc (INT; input).
 * lz       = Lower bound for absorbing bc (INT; input).
 * hz       = Upper bound for absorbing bc (INT; input).
 * nx       = # of points along x (INT).
 * ny       = # of points along y (INT).
 * nz       = # of points along z (INT).
 *
 * Only periodic boundaries!
 *
 */

extern "C" void grid_cuda_wf_propagate_potentialW(CUCOMPLEX *grid, CUCOMPLEX *pot, CUCOMPLEX time_step, CUREAL amp, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz, INT nx, INT ny, INT nz) {

  dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (ny + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK,
              (nx + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);

  grid_cuda_wf_propagate_potential_gpu<<<blocks,threads>>>(grid, pot, time_step, amp, lx, hx, ly, hy, lz, hz, nx, ny, nz);
  cuda_error_check();
}
