/*
 * Boundary condition routines for CUDA. Not accessed in multi-GPU case (FD disabled).
 *
 * Last reviewed: 26 Sep 2019.
 *
 * TODO: For Dirichlet BC one should be able to change to other values than zero.
 *
 */

/*
 * Access point at (i+1, j, k).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input). Grid value at i+1 will be returned.
 * j   = Index j for grid access (INT; input).
 * k   = Index k for grid access (INT; input).
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i+1, j, k) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_x_plus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(i == nx-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0, 0.0);   // return zero
      case 1: /* Neumann */
        return a[((nx - 2) * ny + j) * nz + k];  // return value at i = nx-2
      case 2: /* Periodic */
        return a[j * nz + k];  // return value at i = 0
    }
  }
  return a[((i+1) * ny + j) * nz + k];
}

/*
 * Access point at (i-1, j, k).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input). Grid value at i-1 will be returned.
 * j   = Index j for grid access (INT; input).
 * k   = Index k for grid access (INT; input).
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i-1, j, k) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_x_minus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(i == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0,0.0);
      case 1: /* Neumann */
        return a[(ny + j) * nz + k];  // i = 1
      case 2: /* Periodic */
        return a[((nx-1) * ny + j) * nz + k];  // i = nx-1
    }
  }
  return a[((i-1) * ny + j) * nz + k];
}

/*
 * Access point at (i, j+1, k).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input).
 * j   = Index j for grid access (INT; input). Grid value at j+1 will be returned.
 * k   = Index k for grid access (INT; input).
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i, j+1, k) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_y_plus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(j == ny-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0,0.0);
      case 1: /* Neumann */
        return a[(i * ny + (ny-2)) * nz + k];  // j = ny-2
      case 2: /* Periodic */
        return a[i * ny * nz + k];  // j = 0
    }
  }
  return a[(i * ny + (j+1)) * nz + k];
}

/*
 * Access point at (i, j-1, k).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input).
 * j   = Index j for grid access (INT; input). Grid value at j-1 will be returned.
 * k   = Index k for grid access (INT; input).
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i, j-1, k) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_y_minus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(j == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0,0.0);
      case 1: /* Neumann */
        return a[(i * ny + 1) * nz + k];  // j = 1
      case 2: /* Periodic */
        return a[(i * ny + (ny-1)) * nz + k];  // j = ny-1
    }
  } 
  return a[(i * ny + (j-1)) * nz + k];
}

/*
 * Access point at (i, j, k+1).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input).
 * j   = Index j for grid access (INT; input).
 * k   = Index k for grid access (INT; input). Grid value at k+1 will be returned.
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i, j, k+1) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_z_plus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(k == nz-1) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0, 0.0);
      case 1: /* Neumann */
        return a[(i * ny + j) * nz + (nz-2)];  // k = nz-2
      case 2: /* Periodic */
        return a[(i * ny + j) * nz];  // k = 0
    }
  }
  return a[(i * ny + j) * nz + (k+1)];
}

/*
 * Access point at (i, j, k-1).
 *
 * a   = Grid to be accessed (CUCOMPLEX *; input).
 * bc  = Boundary condition as follows: 0 = Dirichlet, 1 = Neumann, 2 = Periodic.
 * i   = Index i for grid access (INT; input).
 * j   = Index j for grid access (INT; input).
 * k   = Index k for grid access (INT; input). Grid value at k-1 will be returned.
 * nx  = Number of points in grid along x (INT; input).
 * ny  = Number of points in grid along y (INT; input).
 * nz  = Number of points in grid along z (INT; input).
 *
 * This is device code. Returns the grid point value at (i, j, k-1) according to the
 * chosen boundary condition.
 *
 */

__device__ inline CUCOMPLEX cgrid_cuda_bc_z_minus(CUCOMPLEX *a, char bc, INT i, INT j, INT k, INT nx, INT ny, INT nz) {

  if(k == 0) {
    switch(bc) {
      case 0: /* Dirichlet */
        return CUMAKE(0.0, 0.0);
      case 1: /* Neumann */
        return a[(i * ny + j) * nz + 1];  // k = 1
      case 2: /* Periodic */
        return a[(i * ny + j) * nz + (nz-1)];  // k = nz-1
    }
  }
  return a[(i * ny + j) * nz + (k-1)];
}
