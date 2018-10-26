/*
 * Benchmark FFTW 3D r2c/c2r and c2c/c2c  + their in-place & out-of-place variants.
 *
 */

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include <sys/time.h>

/* 3D grid size to test */
#define NX 512
#define NY 256
#define NZ 256

/* Number of threads to use */
#define THREADS 8

int main(int argc, char **argv) {

  REAL *inp;
  REAL complex *out, *inp2, *out2;
  long s = 2 * NX * NY * (NZ / 2 + 1), i;
#ifdef SINGLE_PREC
  fftwf_plan f, b;
#else
  fftw_plan f, b;
#endif
  struct timeval st, now;

#ifdef SINGLE_PREC
  fprintf(stderr, "Single precision floats.\n");
  inp = (float *) fftwf_malloc(sizeof(float) * (size_t) s);
  out = (float complex *) inp; /* in-place */
  out2 = (float complex *) fftwf_malloc(sizeof(float) * (size_t) s); /* out-of-place */
  fftwf_plan_with_nthreads(THREADS);
#else
  fprintf(stderr, "Double precision floats.\n");
  inp = (double *) fftw_malloc(sizeof(double) * (size_t) s);
  out = (double complex *) inp; /* in-place */
  out2 = (double complex *) fftw_malloc(sizeof(double) * (size_t) s); /* out-of-place */
  fftw_plan_with_nthreads(THREADS);
#endif

  /* generate random data */
  srand48(123);
  for(i = 0; i < NX * NY * NZ; i++)
    inp[i] = (REAL) drand48();
    
  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  f = fftwf_plan_dft_r2c_3d(NX, NY, NZ, inp, out, FFTW_MEASURE);
  b = fftwf_plan_dft_c2r_3d(NX, NY, NZ, out, inp, FFTW_MEASURE);
#else
  f = fftw_plan_dft_r2c_3d(NX, NY, NZ, inp, out, FFTW_MEASURE);
  b = fftw_plan_dft_c2r_3d(NX, NY, NZ, out, inp, FFTW_MEASURE);
#endif
  fprintf(stderr, "done.\n");

  gettimeofday(&st, 0);
#ifdef SINGLE_PREC
  fftwf_execute(f);
  fftwf_execute(b);
#else
  fftw_execute(f);
  fftw_execute(b);
#endif
  gettimeofday(&now, 0);
  printf("In-place r2c/c2r: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));

#ifdef SINGLE_PREC
  fftwf_destroy_plan(f);
  fftwf_destroy_plan(b);
#else
  fftw_destroy_plan(f);
  fftw_destroy_plan(b);
#endif
  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  f = fftwf_plan_dft_r2c_3d(NX, NY, NZ, inp, out2, FFTW_MEASURE);
  b = fftwf_plan_dft_c2r_3d(NX, NY, NZ, out2, inp, FFTW_MEASURE);
#else
  f = fftw_plan_dft_r2c_3d(NX, NY, NZ, inp, out2, FFTW_MEASURE);
  b = fftw_plan_dft_c2r_3d(NX, NY, NZ, out2, inp, FFTW_MEASURE);
#endif
  fprintf(stderr, "done.\n");

  gettimeofday(&st, 0);
#ifdef SINGLE_PREC
  fftwf_execute(f);
  fftwf_execute(b);
#else
  fftw_execute(f);
  fftw_execute(b);
#endif
  gettimeofday(&now, 0);
  printf("Out-of-place r2c/c2r: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));

#ifdef SINGLE_PREC
  fftwf_free(inp);
  fftwf_free(out2);
  fftwf_destroy_plan(f);
  fftwf_destroy_plan(b);
  inp2 = (REAL complex *) fftwf_malloc(sizeof(REAL complex) * NX * NY * NZ);
  out2 = (REAL complex *) fftwf_malloc(sizeof(REAL complex) * NX * NY * NZ); /* out-of-place */
#else
  fftw_free(inp);
  fftw_free(out2);
  fftw_destroy_plan(f);
  fftw_destroy_plan(b);
  inp2 = (REAL complex *) fftw_malloc(sizeof(REAL complex) * NX * NY * NZ);
  out2 = (REAL complex *) fftw_malloc(sizeof(REAL complex) * NX * NY * NZ); /* out-of-place */
#endif

  srand48(123);
  for(i = 0; i < NX * NY * NZ; i++)
    inp2[i] = (REAL complex) (drand48() + I * drand48());

  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  f = fftwf_plan_dft_3d(NX, NY, NZ, inp2, inp2, FFTW_FORWARD, FFTW_MEASURE);
  b = fftwf_plan_dft_3d(NX, NY, NZ, inp2, inp2, FFTW_BACKWARD, FFTW_MEASURE);
#else
  f = fftw_plan_dft_3d(NX, NY, NZ, inp2, inp2, FFTW_FORWARD, FFTW_MEASURE);
  b = fftw_plan_dft_3d(NX, NY, NZ, inp2, inp2, FFTW_BACKWARD, FFTW_MEASURE);
#endif
  fprintf(stderr, "done.\n");

  gettimeofday(&st, 0);
#ifdef SINGLE_PREC
  fftwf_execute(f);
  fftwf_execute(b);
#else
  fftw_execute(f);
  fftw_execute(b);
#endif
  gettimeofday(&now, 0);
  printf("In-place c2c/c2c: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));
#ifdef SINGLE_PREC
  fftwf_destroy_plan(f);
  fftwf_destroy_plan(b);
#else
  fftw_destroy_plan(f);
  fftw_destroy_plan(b);
#endif

  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  f = fftwf_plan_dft_3d(NX, NY, NZ, inp2, out2, FFTW_FORWARD, FFTW_MEASURE);
  b = fftwf_plan_dft_3d(NX, NY, NZ, out2, inp2, FFTW_BACKWARD, FFTW_MEASURE);
#else
  f = fftw_plan_dft_3d(NX, NY, NZ, inp2, out2, FFTW_FORWARD, FFTW_MEASURE);
  b = fftw_plan_dft_3d(NX, NY, NZ, out2, inp2, FFTW_BACKWARD, FFTW_MEASURE);
#endif
  fprintf(stderr, "done.\n");

  gettimeofday(&st, 0);
#ifdef SINGLE_PREC
  fftwf_execute(f);
  fftwf_execute(b);
#else
  fftw_execute(f);
  fftw_execute(b);
#endif
  gettimeofday(&now, 0);
  printf("Out-of-place c2c/c2c: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));
#ifdef SINGLE_PREC
  fftwf_destroy_plan(f);
  fftwf_destroy_plan(b);
#else
  fftw_destroy_plan(f);
  fftw_destroy_plan(b);
#endif

  exit(0);
}
