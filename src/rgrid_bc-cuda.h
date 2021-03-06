/*
 * BC routines / CUDA.
 *
 * bc = Boundary condition as follows:
 * 0 = Dirichlet
 * 1 = Neumann
 * 2 = Periodic
 *
 */

__device__ inline REAL rgrid_cuda_bc_x_plus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(i == nx-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[((nx - 2) * ny + j) * nzz + k];  // i = nx-2
      case 2: /* Periodic */
        return a[j * nzz + k];  // i = 0
    }
  }
  return a[((i+1) * ny + j) * nzz + k];
}

__device__ inline REAL rgrid_cuda_bc_x_minus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(i == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[(ny + j) * nzz + k];  // i = 1
      case 2: /* Periodic */
        return a[((nx-1) * ny + j) * nzz + k];  // i = nx-1
    }
  }
  return a[((i-1) * ny + j) * nzz + k];
}

__device__ inline REAL rgrid_cuda_bc_y_plus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(j == ny-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[(i * ny + (ny-2)) * nzz + k];  // j = ny-2
      case 2: /* Periodic */
        return a[i * ny * nzz + k];  // j = 0
    }
  }
  return a[(i * ny + (j+1)) * nzz + k];
}

__device__ inline REAL rgrid_cuda_bc_y_minus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(j == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[(i * ny + 1) * nzz + k];  // j = 1
      case 2: /* Periodic */
        return a[(i * ny + (ny-1)) * nzz + k];  // j = ny-1
    }
  } 
  return a[(i * ny + (j-1)) * nzz + k];
}

__device__ inline REAL rgrid_cuda_bc_z_plus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(k == nz-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[(i * ny + j) * nzz + (nz-2)];  // k = nz-2
      case 2: /* Periodic */
        return a[(i * ny + j) * nzz];  // k = 0
    }
  }
  return a[(i * ny + j) * nzz + (k+1)];
}

__device__ inline REAL rgrid_cuda_bc_z_minus(REAL *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz, INT nzz) {

  if(k == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return 0.0;
      case 1: /* Neumann */
        return a[(i * ny + j) * nzz + 1];  // k = 1
      case 2: /* Periodic */
        return a[(i * ny + j) * nzz + (nz-1)];  // k = nz-1
    }
  }
  return a[(i * ny + j) * nzz + (k-1)];
}
