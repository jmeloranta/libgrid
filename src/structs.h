/** 
 *  Data structures for libgrid
 *
 */ 

#ifndef __GRID_STRUCTS__
#define __GRID_STRUCTS__

/*
 * @brief Wall clock and CPU time structures.
 *
 */

typedef struct grid_timer_struct {
  struct timeval zero_time;  /* Wall clock starting time */
  clock_t zero_clock;        /* CPU starting time */
} grid_timer;

/**
 * Complex grid structure
 *
 */

typedef struct cgrid_struct {
  REAL complex *value;  /* Array holding the complex values */
  char id[32];          /* String describing the grid */
  size_t grid_len;      /* Data length in bytes allocated for grid->value */
  INT nx, ny, nz;       /* Grid dimensions along x, y, z directions */
  REAL step;            /* Grid spatial step length */
  REAL x0, y0, z0;      /* Real space origin for the grid */
  REAL kx0, ky0, kz0;   /* Reciprocal space origin for the grid */
  REAL omega;           /* Rotation frequency about the z axis */
  REAL complex (*value_outside)(struct cgrid_struct *grid, INT i, INT j, INT k); /* Pointer to function returning the grid value at boundary. */
  void *outside_params_ptr; /* Optional parameters for the boundary value function */
  REAL complex default_outside_params; /* Default parameter for the boundary value function */
#if defined(SINGLE_PREC)
  fftwf_plan plan, iplan;  /* Single precision FFTW plans */
#elif defined(DOUBLE_PREC)
  fftw_plan plan, iplan;   /* Double precision FFTW plans */
#elif defined(QUAD_PREC)
  fftwl_plan plan, iplan;  /* Quad precision FFTW plans */
#endif
#ifdef USE_CUDA
  cufftHandle cufft_handle;/* CUFFT plan handle */
  char host_lock;          /* 0 = may move to GPU memory; 1 = locked in host memory */
#endif
  REAL fft_norm;           /* FFT normalization constant */
  REAL fft_norm2;          /* FFT normalization constant including the spatial step length */
  char flag;               /* claim/release interface flag */
} cgrid;

/*
 * Real grid structure
 *
 */

typedef struct rgrid_struct {
  REAL *value;                /* Array holding the real grid values */
  char id[32];                /* String identifying the grid */
  size_t grid_len;            /* Amount of memory in bytes allocated for the grid */
  INT nx, ny, nz, nz2;        /* Grid dimensions along x, y, z. nz2 = 2 * (nz / 2 + 1) */
  REAL step;                  /* Spatial step length */
  REAL x0, y0, z0;            /* Origin in real space */
  REAL kx0, ky0, kz0;         /* Origin in reciprocal space */
  REAL (*value_outside)(struct rgrid_struct *grid, INT i, INT j, INT k);  /* Pointer to function returing values at the grid boundary */
  void *outside_params_ptr;   /* Parameter pointer for boundary function data */
  REAL default_outside_params;/* Default parameter for boundary function */
#if defined(SINGLE_PREC)
  fftwf_plan plan, iplan;     /* Single precision FFTW plans */
#elif defined(DOUBLE_PREC)
  fftw_plan plan, iplan;      /* Double precision FFTW plans */
#elif defined(QUAD_PREC)
  fftwl_plan plan, iplan;     /* Quad precision FFTW plans */
#endif
#ifdef USE_CUDA
  cufftHandle cufft_handle_r2c; /* CUFFT R2C plan handle */
  cufftHandle cufft_handle_c2r; /* CUFFT C2R plan handle */
  char host_lock;               /* host memory lock flag: 0 = may move to GPU memory; 1 = locked in host memory */
#endif
  REAL fft_norm;                /* FFT normalization constant */
  REAL fft_norm2;               /* FFT normalization constant including the step length */
  char flag;                    /* claim/release interface flag */
} rgrid;

/*
 * Wave function data structure
 *
 */

typedef struct wf_struct {
  cgrid *grid;                  /* Grid holding the wave function data */
  REAL mass;                    /* Particle mass */
  REAL norm;                    /* Wave function norm */
  REAL cfft_width;              /* Window width for CFFT */
  char boundary;                /* Boundary condition */
  char propagator;              /* Time propagator */
  cgrid *cworkspace;            /* Workspace (NULL if not in use) */
  cgrid *cworkspace2;           /* Workspace (NULL if not in use) */
  cgrid *cworkspace3;           /* Workspace (NULL if not in use) */
  REAL (*ts_func)(INT, INT, INT, INT, INT, INT, INT, INT, INT);  /* Time step function (allows spatially dependent time) */
  INT lx, hx, ly, hy, lz, hz;   /* Abs boundary region indices */
} wf;

/*
 * Grid rotation structure
 *
 */

typedef struct rotation_struct {
  rgrid *rgrid;                /* Real grid to be rotated */
  cgrid *cgrid;                /* Complex grid to be rotated */
  REAL sinth;                  /* sin(theta) */
  REAL costh;                  /* cos(theta) */
} grid_rotation;

/*
 * Precomputed function structure (REAL).
 *
 */

typedef struct rfunction {
  char id[32]; /* function ID string */ 
  REAL begin;  /* starting value */
  INT nsteps;  /* number of values */
  REAL step;   /* step */
  REAL *value; /* values */
  size_t length; /* Data length */
} rfunction;

/*
 * Precomputed function structure (REAL complex).
 *
 */

typedef struct cfunction {
  char id[32]; /* function ID string */ 
  REAL begin;  /* starting value */
  INT nsteps;  /* number of values */
  REAL step;   /* step */
  REAL *value; /* values */
  size_t length; /* Data length */
} cfunction;

#endif /* __GRID_STRUCTS__ */
