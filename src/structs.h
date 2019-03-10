/*
 * Grid structures.
 *
 */

#ifndef __GRID_STRUCTS__
#define __GRID_STRUCTS__

typedef struct grid_timer_struct { /* wall clock timing structure */
  struct timeval zero_time;
  clock_t zero_clock;
} grid_timer;

struct grid_abs {
  REAL amp;
  INT data[6];
};

typedef struct cgrid_struct { /* complex grid data type */
  REAL complex *value;
  char id[32];
  size_t grid_len;
  INT nx, ny, nz;
  REAL step;
  REAL x0, y0, z0;
  REAL kx0, ky0, kz0;
  REAL omega;         /* rotation frequency about the z axis */
  REAL complex (*value_outside)(struct cgrid_struct *grid, INT i, INT j, INT k);
  void *outside_params_ptr;
  REAL complex default_outside_params;
#if defined(SINGLE_PREC)
  fftwf_plan plan, iplan, implan, iimplan;
#elif defined(DOUBLE_PREC)
  fftw_plan plan, iplan, implan, iimplan;
#elif defined(QUAD_PREC)
  fftwl_plan plan, iplan, implan, iimplan;
#endif
#ifdef USE_CUDA
  cufftHandle cufft_handle;
#endif
  REAL fft_norm;
  REAL fft_norm2;
  char flag;
} cgrid;

typedef struct rgrid_struct { /* real grid data type */
  REAL *value;
  char id[32];
  size_t grid_len;
  INT nx, ny, nz, nz2;
  REAL step;
  REAL x0, y0, z0;
  REAL kx0, ky0, kz0;
  REAL (*value_outside)(struct rgrid_struct *grid, INT i, INT j, INT k);
  void *outside_params_ptr;
  REAL default_outside_params;
#if defined(SINGLE_PREC)
  fftwf_plan plan, iplan;
#elif defined(DOUBLE_PREC)
  fftw_plan plan, iplan;
#elif defined(QUAD_PREC)
  fftwl_plan plan, iplan;
#endif
#ifdef USE_CUDA
  cufftHandle cufft_handle_r2c;
  cufftHandle cufft_handle_c2r;
#endif
  REAL fft_norm;
  REAL fft_norm2;
  char flag;
} rgrid;

typedef struct wf_struct { /* wavefunction */
  cgrid *grid;
  REAL mass;
  REAL norm;
  char boundary;
  char propagator;
  cgrid *cworkspace;
  cgrid *cworkspace2;
  cgrid *cworkspace3;
  REAL complex (*ts_func)(INT, INT, INT, void *);
  struct grid_abs abs_data;
} wf;

typedef struct rotation_struct {  /* structure for rotating grids */
  rgrid *rgrid;
  cgrid *cgrid;
  REAL sinth;
  REAL costh;
} grid_rotation;

#endif /* __GRID_STRUCTS__ */
