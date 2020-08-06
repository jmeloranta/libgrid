/*
 * CUDA device code (REAL complex; cgrid).
 *
 * TODO: There are too many if's and switches below (performance). Branching can be expensive.
 *       Perhaps use single source with preprocessor directives to generate code for each case
 *       separately.
 *       Can we use more local memory?
 *
 */

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <device_launch_parameters.h>
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
 * nx    = Grid dimension along X (INT; input).
 * ny    = Grid dimension along Y (INT; input).
 * nz    = Grid dimension along Z (INT; input).
 * nyz   = Precomputed ny * nz (INT; input).
 * ny2   = Precomputed ny / 2 (INT; input).
 * psi   = Wavefunction (CUCOMPLEX *; input/output).
 * bc    = Boundary condition: WF_DIRICHLET_BOUNDARY, WF_NEUMANN_BOUNDARY, or WF_PERIODIC_BOUNDARY (char; input).
 * wrk   = Workspace (CUCOMPLEX *; input).
 * wrk2  = Workspace (CUCOMPLEX *; input).
 * wrk3  = Workspace (CUCOMPLEX *; input). Not needed if BC = Neumann.
 * c     = Precomputed: i * 4 * mass * step^2 / HBAR (REAL; input).
 * c2    = Precomputed: -i * step * kx0 (REAL; input);
 * c3    = Precomputed: i * mass * omega * step / HBAR (REAL; input).
 * step  = Spatial step length (REAL; input).
 * y0    = Spatial offset along y (REAL; input).
 * tstep = Time step length (REAL; input).
 * lx    = Lower X index for absorbing boundary (INT; input).
 * hx    = Upper X index for absorbing boundary (INT; input).
 * ly    = Lower Y index for absorbing boundary (INT; input).
 * hy    = Upper Y index for absorbing boundary (INT; input).
 * lz    = Lower Z index for absorbing boundary (INT; input).
 * hz    = Upper Z index for absorbing boundary (INT; input).
 * 
 * if lx == 0, use constant time step given by tstep.
 * else call grid_wf_absorb().
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_x_gpu(INT nx, INT ny, INT nz, INT nyz, INT ny2, __restrict__ CUCOMPLEX *psi, char bc, __restrict__ CUCOMPLEX *wrk, __restrict__ CUCOMPLEX *wrk2, __restrict__ CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL y0, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT i, k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, tid, ind, ntid;
  CUREAL y, tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(j >= ny || k >= nz) return;

  tid = j * nz + k;
  ntid = nx * tid;
  d = &wrk[ntid];
  b = &wrk2[ntid];
  if(wrk3) pwrk = &wrk3[ntid];
  y = ((REAL) (j - ny2)) * step - y0;    
  tim = tstep;

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(i = 1; i < nx - 1; i++) {
    if(lx) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    }
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
  if(lx) {
    tmp = grid_cuda_wf_absorb(0, j, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  }
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
  if(lx) {
    tmp = grid_cuda_wf_absorb(nx-1, j, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  }
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
 * wf         = Source/destination grid for operation (gpu_mem_block *; input).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kx0        = base momentum along x (REAL; input).
 * omega      = rotation freq (REAL; input).
 * y0         = y0 grid spatial offset (REAL; input).
 * wrk        = Workspace (gpu_mem_block *; input).
 * wrk2       = Workspace (gpu_mem_block *; input).
 * wrk3       = Workspace (gpu_mem_block *; input). Not needed if BC = Neumann.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_xW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, gpu_mem_block *gwf, char bc, REAL mass, REAL step, REAL kx0, REAL omega, REAL y0, gpu_mem_block *wrk, gpu_mem_block *wrk2, gpu_mem_block *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *WRK = wrk->gpu_info->descriptor, *WRK2 = wrk2->gpu_info->descriptor, *WRK3 = wrk3?wrk3->gpu_info->descriptor:NULL;
  CUCOMPLEX c, c2, c3;
  INT nyz = ny * nz, ny2 = ny / 2;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_propagate_kinetic_cn_x wrong subformat.\n");
    abort();
  }

  wrk->gpu_info->subFormat = wrk2->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  if(wrk3) wrk3->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);

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

  grid_cuda_wf_propagate_kinetic_cn_x_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, ny2, (CUCOMPLEX *) GWF->data[0], bc, (CUCOMPLEX *) WRK->data[0], (CUCOMPLEX *) WRK2->data[0], 
    (CUCOMPLEX *) (WRK3?WRK3->data[0]:NULL), c, c2, c3, step, y0, tstep, lx, hx, ly, hy, lz, hz);

  cuda_error_check();
}

/*
 * Propagate wf using CN along Y.
 *
 * nx    = Grid dimension along X (INT; input).
 * ny    = Grid dimension along Y (INT; input).
 * nz    = Grid dimension along Z (INT; input).
 * nyz   = Precomputed ny * nz (INT; input).
 * nx2   = Precomputed nx / 2 (INT; input).
 * psi   = Wavefunction (CUCOMPLEX *; input/output).
 * bc    = Boundary condition: WF_DIRICHLET_BOUNDARY, WF_NEUMANN_BOUNDARY, or WF_PERIODIC_BOUNDARY (char; input).
 * wrk   = Workspace (CUCOMPLEX *; input).
 * wrk2  = Workspace (CUCOMPLEX *; input).
 * wrk3  = Workspace (CUCOMPLEX *; input). Not needed if BC = Neumann.
 * c     = Precomputed: i * 4 * mass * step^2 / HBAR (REAL; input).
 * c2    = Precomputed: -i * step * ky0 (REAL; input);
 * c3    = Precomputed: -i * mass * omega * step / HBAR (REAL; input).
 * step  = Spatial step length (REAL; input).
 * y0    = Spatial offset along y (REAL; input).
 * tstep = Time step length (REAL; input).
 * lx    = Lower X index for absorbing boundary (INT; input).
 * hx    = Upper X index for absorbing boundary (INT; input).
 * ly    = Lower Y index for absorbing boundary (INT; input).
 * hy    = Upper Y index for absorbing boundary (INT; input).
 * lz    = Lower Z index for absorbing boundary (INT; input).
 * hz    = Upper Z index for absorbing boundary (INT; input).
 *
 * if ly == 0, use constant time step given by tstep.
 * else call grid_wf_absorb().
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_y_gpu(INT nx, INT ny, INT nz, INT nyz, INT nx2, __restrict__ CUCOMPLEX *psi, char bc, __restrict__ CUCOMPLEX *wrk, __restrict__ CUCOMPLEX *wrk2, __restrict__ CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUCOMPLEX c3, CUREAL step, CUREAL x0, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k = blockIdx.x * blockDim.x + threadIdx.x, j, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind, ntid;
  CUREAL x, tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || k >= nz) return;

  x = ((REAL) (i - nx2)) * step - x0;    
  tid = i * nz + k;
  ntid = ny * tid;
  d = &wrk[ntid];
  b = &wrk2[ntid];
  if(wrk3) pwrk = &wrk3[ntid];
  tim = tstep;

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(j = 1; j < ny - 1; j++) {
    if(ly) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    }
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
  if(ly) {
    tmp = grid_cuda_wf_absorb(i, 0, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  }
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
  if(ly) {
    tmp = grid_cuda_wf_absorb(i, ny-1, k, lx, hx, ly, hy, lz, hz);
    tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
  }
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
 * wf         = Source/destination grid for operation (gpu_mem_block *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * ky0        = base momentum along y (REAL; input).
 * omega      = rotation freq (REAL; input).
 * x0         = x0 grid spatial offset (REAL; input).
 * wrk        = Workspace (gpu_mem_block *; scratch space).
 * wrk2       = Workspace (gpu_mem_block *; scratch space).
 * wrk3       = Workspace (gpu_mem_block *; scratch space). Not needed if BC = Neumann.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_yW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, gpu_mem_block *gwf, char bc, REAL mass, REAL step, REAL ky0, REAL omega, REAL x0, gpu_mem_block *wrk, gpu_mem_block *wrk2, gpu_mem_block *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((nz + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (nx + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *WRK = wrk->gpu_info->descriptor, *WRK2 = wrk2->gpu_info->descriptor, *WRK3 = wrk3?wrk3->gpu_info->descriptor:NULL;
  CUCOMPLEX c, c2, c3;
  INT nyz = ny * nz, nx2 = nx / 2;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_propagate_kinetic_cn_y wrong subformat.\n");
    abort();
  }

  wrk->gpu_info->subFormat = wrk2->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  if(wrk3) wrk3->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);

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

  grid_cuda_wf_propagate_kinetic_cn_y_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nx2, (CUCOMPLEX *) GWF->data[0], bc, (CUCOMPLEX *) WRK->data[0], (CUCOMPLEX *) WRK2->data[0],
    (CUCOMPLEX *) (WRK3?WRK3->data[0]:NULL), c, c2, c3, step, x0, tstep, lx, hx, ly, hy, lz, hz);

  cuda_error_check();
}

/*
 * Propagate wf using CN along Z.
 *
 * nx    = Grid dimension along X (INT; input).
 * ny    = Grid dimension along Y (INT; input).
 * nz    = Grid dimension along Z (INT; input).
 * nyz   = Precomputed ny * nz (INT; input).
 * nxy   = Precomputed nx * ny (INT; input).
 * psi   = Wavefunction (CUCOMPLEX *; input/output).
 * bc    = Boundary condition: WF_DIRICHLET_BOUNDARY, WF_NEUMANN_BOUNDARY, or WF_PERIODIC_BOUNDARY (char; input).
 * wrk   = Workspace (CUCOMPLEX *; input).
 * wrk2  = Workspace (CUCOMPLEX *; input).
 * wrk3  = Workspace (CUCOMPLEX *; input). Not needed if BC = Neumann.
 * c     = Precomputed: i * 4 * mass * step^2 / HBAR (REAL; input).
 * c2    = Precomputed: -i * step * kz0 (REAL; input);
 * step  = Spatial step length (REAL; input).
 * tstep = Time step length (REAL; input).
 * lx    = Lower X index for absorbing boundary (INT; input).
 * hx    = Upper X index for absorbing boundary (INT; input).
 * ly    = Lower Y index for absorbing boundary (INT; input).
 * hy    = Upper Y index for absorbing boundary (INT; input).
 * lz    = Lower Z index for absorbing boundary (INT; input).
 * hz    = Upper Z index for absorbing boundary (INT; input).
 *
 * if lz == 0, use constant time step given by tstep.
 * else call grid_wf_absorb().
 *
 */

__global__ void grid_cuda_wf_propagate_kinetic_cn_z_gpu(INT nx, INT ny, INT nz, INT nyz, INT nxy, __restrict__ CUCOMPLEX *psi, char bc, __restrict__ CUCOMPLEX *wrk, __restrict__ CUCOMPLEX *wrk2, __restrict__ CUCOMPLEX *wrk3, CUCOMPLEX c, CUCOMPLEX c2, CUREAL step, CUCOMPLEX tstep, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  INT k, j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y * blockDim.y + threadIdx.y, tid, ind, ntid;
  CUREAL tmp;
  CUCOMPLEX *d, *b, *pwrk, cp, tim;

  if(i >= nx || j >= ny) return;

  tid = i * ny + j;
  ntid = nz * tid;
  d = &wrk[ntid];
  b = &wrk2[ntid];
  if(wrk3) pwrk = &wrk3[ntid];
  tim = tstep;

  /* create left-hand diagonal element (d) and right-hand vector (b) */
  for(k = 1; k < nz - 1; k++) {
    if(lz) {
      tmp = grid_cuda_wf_absorb(i, j, k, lx, hx, ly, hy, lz, hz);
      tim = CUMAKE((1.0 - tmp) * tstep.x, -tstep.x * tmp);
    }
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
  }
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
  }
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
 * wf         = Source/destination grid for operation (gpu_mem_block *; input/output).
 * bc         = boundary condition (gwf->boundary) (char; input).
 * mass       = gwf mass (REAL; input).
 * step       = spatial step (REAL; input).
 * kz0        = base momentum along z (REAL; input).
 * wrk        = Workspace (gpu_mem_block *; scratch space).
 * wrk2       = Workspace (gpu_mem_block *; scratch space).
 * wrk3       = Workspace (gpu_mem_block *; scratch space). Not needed if BC = Neumann.
 * lx         = Absorbing low boundary index x (INT; input).
 * hx         = Absorbing high boundary index x (INT; input).
 * ly         = Absorbing low boundary index y (INT; input).
 * hy         = Absorbing high boundary index y (INT; input).
 * lz         = Absorbing low boundary index z (INT; input).
 * hz         = Absorbing high boundary index z (INT; input).
 * 
 */

extern "C" void grid_cuda_wf_propagate_kinetic_cn_zW(INT nx, INT ny, INT nz, CUCOMPLEX tstep, gpu_mem_block *gwf, char bc, REAL mass, REAL step, REAL kz0, gpu_mem_block *wrk, gpu_mem_block *wrk2, gpu_mem_block *wrk3, INT lx, INT hx, INT ly, INT hy, INT lz, INT hz) {

  dim3 threads(CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK, CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK);
  dim3 blocks((ny + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK), 
              (nx + CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK - 1) / (CUDA_CN_THRADJ*CUDA_THREADS_PER_BLOCK));
  cudaXtDesc *GWF = gwf->gpu_info->descriptor, *WRK = wrk->gpu_info->descriptor, *WRK2 = wrk2->gpu_info->descriptor, *WRK3 = wrk3?wrk3->gpu_info->descriptor:NULL;
  CUCOMPLEX c, c2;
  INT nyz = ny * nz, nxy = nx * ny;

  if(gwf->gpu_info->subFormat != CUFFT_XT_FORMAT_INPLACE) {
    fprintf(stderr, "libgrid(cuda): wf_propagate_kinetic_cn_z wrong subformat.\n");
    abort();
  }

  wrk->gpu_info->subFormat = wrk2->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;
  if(wrk3) wrk3->gpu_info->subFormat = CUFFT_XT_FORMAT_INPLACE;

  if(GWF->nGPUs > 1) {
    fprintf(stderr, "libgrid(cuda): Non-local grid operations disabled for multi-GPU calculations.\n");                                                               
    abort();
  }

  cudaSetDevice(GWF->GPUs[0]);

  /*
   * (1 + .5 i (T + V - v0 pz) dt / hbar) psi(t+dt) = (1 - .5 i (T + V - v0 pz) dt / hbar) psi(t) <=> A x = b
   * (C + dz^2 laplace + C2 grad) psi(t+dt) 
   *         = (C - dz^2 laplace - C2 grad) psi(t)
   * where C = 4 i m dz^2 / (hbar dt), CB = -2m dz^2 / hbar^2, C2 = -i dz kz.
   *
   */
  c = CUMAKE(0.0, 4.0 * mass * step * step / HBAR);
  c2 = CUMAKE(0.0, -step * kz0); // coeff for moving background

  grid_cuda_wf_propagate_kinetic_cn_z_gpu<<<blocks,threads>>>(nx, ny, nz, nyz, nxy, (CUCOMPLEX *) GWF->data[0], bc, (CUCOMPLEX *) WRK->data[0], (CUCOMPLEX *) WRK2->data[0], 
    (CUCOMPLEX *) (WRK3?WRK3->data[0]:NULL), c, c2, step, tstep, lx, hx, ly, hy, lz, hz);

  cuda_error_check();
}
