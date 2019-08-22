/*
 * CUDA device code (REAL complex; cgrid).
 *
 * TODO: Should we use to BC routines rather than hard code the BCs below? (efficiency?)
 *
 */

#include <stdio.h>
#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/device_launch_parameters.h>
#include <cufft.h>
#include "cuda.h"
#include "cuda-math.h"
#include "defs.h"
#include "linalg-cuda.h"
#include "grid_wf-cuda.h"

extern void *grid_gpu_mem_addr;
extern "C" void cuda_error_check();

/*
 * Propagate wf using CN along X.
 *
 * if lz == 0, use constant time step given by tstep.
 * else call grid_wf_absorb().
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_x_gpu(INT nx, INT ny, INT nz, INT nyz, INT ny2, CUCOMPLEX *psi, char bc, CUCOMPLEX *wrk, CUCOMPLEX *wrk2, CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL y0, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUREAL y, tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(j >= ny || k >= nz) return;

  tid = j * nz + k;
  d = &wrk[nx * tid];
  b = &wrk2[nx * tid];
  pwrk = &wrk3[nx * tid];
  y = ((REAL) (j - ny2)) * step - y0;    

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(i = 1; i < nx - 1; i++) {
    if(lz) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    } else tim = tstep;
    cp = c / tim;
    ind = i * nyz + tid;
    /* Left-hand side (+) */
    /* (C + dx^2 laplace + C2 grad + C3 y grad) -- only C and laplace have diag elements */
    d[i] = cp - 2.0;  // -2 from Laplacian, C = cp = c / dt
    /* Right-hand side (-) */
    /* (C - dy^2 laplace - C2 grad - C3 y grad)psi */
    b[i] = cp * psi[ind] - (psi[ind + nyz] - 2.0 * psi[ind] + psi[ind - nyz]) 
             - c2 * (psi[ind + nyz] - psi[ind - nyz]) - c3 * y * (psi[ind + nyz] - psi[ind - nyz]);
  }

  // Boundary conditions
  ind = j * nz + k; // i = 0 - left boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(0, j, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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

  ind = (nx - 1) * nyz + j * nz + k;  // i = nx - 1, right boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(nx-1, j, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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
 * wf         = Source/destination grid for operation (cudaXtDesc *; input).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kx0        = base momentum along x (REAL; input).
 * omega      = rotation freq (REAL; input).
 * y0         = y0 grid spatial offset (REAL; input).
 * wrk        = Workspace (cudaXtDesc *; input).
 * wrk2       = Workspace (cudaXtDesc *; input).
 * wrk3       = Workspace (cudaXtDesc *; input).
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_xW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, cudaXtDesc *gwf, char bc, REAL mass, REAL step, REAL kx0, REAL omega, REAL y0, cudaXtDesc *wrk, cudaXtDesc *wrk2, cudaXtDesc *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, c2, c3;
  INT nyz = ny * nz, ny2 = ny / 2;

  if(gwf->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }
  cudaSetDevice(gwf->GPUs[0]);

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 px) dt / hbar) psi(t) <=> A x = b
   * (C + dx^2 laplace + C2 grad + C3 y grad) psi(t+dt) 
   *         = (C - dx^2 laplace - C2 grad - C3 y grad) psi(t)
   * where C = 4 i m dx^2 / (hbar dt), C2 = -i dx kx, C3 = m \omega i dx / hbar
   *
   */
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  c2 = CUMAKE(0.0, -step * kx0); // coeff for moving background
  c3 = CUMAKE(0.0, mass * omega * step / HBAR); // coeff for rotating liquid around Z

  grid_cuda_wf_propagate_kinetic_cn_x_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, ny2, (CUCOMPLEX *) gwf->data[0], bc, (CUCOMPLEX *) wrk->data[0], (CUCOMPLEX *) wrk2->data[0], (CUCOMPLEX *) wrk3->data[0], 
       c, c2, c3, step, y0, tstep, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}

/*
 * Propagate wf using CN along Y.
 *
 * if lz == 0, use constant time step given by tstep.
 * else call 
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_y_gpu(INT nx, INT ny, INT nz, INT nyz, INT nx2, CUCOMPLEX *psi, char bc, CUCOMPLEX *wrk, CUCOMPLEX *wrk2, CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL x0, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUREAL x, tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || k >= nz) return;

  x = ((REAL) (i - nx2)) * step - x0;    
  tid = i * nz + k;
  d = &wrk[ny * tid];
  b = &wrk2[ny * tid];
  pwrk = &wrk3[ny * tid];

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(j = 1; j < ny - 1; j++) {
    if(lz) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    } else tim = tstep;
    cp = c / tim;
    ind = i * nyz + j * nz + k;
    /* Left-hand side (+) */
    /* (C + dy^2 laplace + C2 grad + C3 x grad) */
    d[j] = cp - 2.0; // -2 from Laplacian, cp = c / dt, LHS(+)
    /* Right-hand side (-) */
    /* (C - dx^2 laplace - C2 grad - C3 x grad) */
    b[j] = cp * psi[ind] - (psi[ind + nz] - 2.0 * psi[ind] + psi[ind - nz]) 
           - c2 * (psi[ind + nz] - psi[ind - nz]) - c3 * x * (psi[ind + nz] - psi[ind - nz]);
  }
  
  // Boundary conditions
 
  ind = i * nyz + k; // j = 0 - left boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(i, 0, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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

  ind = i * nyz + (ny-1) * nz + k;  // j = ny - 1 - right boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(i, ny-1, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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
 * wf         = Source/destination grid for operation (cudaXtDesc *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * ky0        = base momentum along y (REAL; input).
 * omega      = rotation freq (REAL; input).
 * x0         = x0 grid spatial offset (REAL; input).
 * wrk        = Workspace (cudaXtDesc *; scratch space).
 * wrk2       = Workspace (cudaXtDesc *; scratch space).
 * wrk3       = Workspace (cudaXtDesc *; scratch space).
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_yW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, cudaXtDesc *gwf, char bc, REAL mass, REAL step, REAL ky0, REAL omega, REAL x0, cudaXtDesc *wrk, cudaXtDesc *wrk2, cudaXtDesc *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, c2, c3;
  INT nyz = ny * nz, nx2 = nx / 2;

  if(gwf->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }
  cudaSetDevice(gwf->GPUs[0]);

  /*
   * (1 + .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - \omega Lz - v0 py) dt / hbar) psi(t) <=> A x = b
   * (C + dy^2 laplace + C2 grad + C3 x grad) psi(t+dt) 
   *         = (C - dx^2 laplace - C2 grad - C3 x grad) psi(t)
   * where C = 4 i m dy^2 / (hbar dt), C2 = -i dy ky, C3 = -m \omega i dy / hbar
   *
   */
 
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  c2 = CUMAKE(0.0, -step * ky0); // coeff for moving background
  c3 = CUMAKE(0.0, -mass * omega * step / HBAR); // coeff for rotating liquid around Z

  grid_cuda_wf_propagate_kinetic_cn_y_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nx2, (CUCOMPLEX *) gwf->data[0], bc, (CUCOMPLEX *) wrk->data[0], (CUCOMPLEX *) wrk2->data[0], (CUCOMPLEX *) wrk3->data[0],
         c, c2, c3, step, x0, tstep, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}

/*
 * Propagate wf using CN along Z.
 *
 * if lz == 0, use constant time step given by tstep.
 * else call 
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_z_gpu(INT nx, INT ny, INT nz, INT nyz, INT nxy, CUCOMPLEX *psi, char bc, CUCOMPLEX *wrk, CUCOMPLEX *wrk2, CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUREAL step, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k, j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind;
  CUREAL tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || j >= ny) return;

  tid = i * ny + j;
  d = &wrk[nz * tid];
  b = &wrk2[nz * tid];
  pwrk = &wrk3[nz * tid];

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(k = 1; k < nz - 1; k++) {
    if(lz) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    } else tim = tstep;
    cp = c / tim;
    ind = i * nyz + j * nz + k;
    /* Left-hand side (+) */
    /* (C + dz^2 laplace + C2 grad) */
    d[k] = cp - 2.0; // Diagonal: -2 from Laplacian, cp = c / dt
    /* Right-hand side (-) */
    /* (C - dz^2 laplace - C2 grad) */
    b[k] = cp * psi[ind] - (psi[ind + 1] - 2.0 * psi[ind] + psi[ind - 1]) 
           - c2 * (psi[ind + 1] - psi[ind - 1]); // NOTE: No rotation term as the rotation is about the z-axis
  }

  // Boundary conditions
 
  ind = i * nyz + j * nz; // k = 0 - left boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(i, j, 0, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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

  ind = i * nyz + j * nz + (nz - 1);  // k = nz-1 - right boundary
  if(lz) {
    tmp = grid_cuda_wf_absorb(i, j, nz-1, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  } else tim = tstep;
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
 * wf         = Source/destination grid for operation (cudaXtDesc *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kz0        = base momentum along z (REAL; input).
 * wrk        = Workspace (cudaXtDesc *; scratch space).
 * wrk2       = Workspace (cudaXtDesc *; scratch space).
 * wrk3       = Workspace (cudaXtDesc *; scratch space).
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_zW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, cudaXtDesc *gwf, char bc, REAL mass, REAL step, REAL kz0, cudaXtDesc *wrk, cudaXtDesc *wrk2, cudaXtDesc *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  CUCOMPLEX c, c2;
  INT nyz = ny * nz, nxy = nx * ny;

  if(gwf->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }
  cudaSetDevice(gwf->GPUs[0]);

  /*
   * (1 + .5 i (T + V - v0 pz) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - v0 pz) dt / hbar) psi(t) <=> A x = b
   * (C + dz^2 laplace + C2 grad) psi(t+dt) 
   *         = (C - dz^2 laplace - C2 grad) psi(t)
   * where C = 4 i m dz^2 / (hbar dt), CB = -2m dz^2 / hbar^2, C2 = -i dz kz.
   *
   */
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  c2 = CUMAKE(0.0, -step * kz0); // coeff for moving background

  grid_cuda_wf_propagate_kinetic_cn_z_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nxy, (CUCOMPLEX *) gwf->data[0], bc, (CUCOMPLEX *) wrk->data[0], (CUCOMPLEX *) wrk2->data[0], (CUCOMPLEX *) wrk3->data[0],
         c, c2, step, tstep, lx, hx, ly, hy, lz, hz);
  cuda_error_check();
}
