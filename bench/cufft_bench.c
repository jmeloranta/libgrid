/*
 * Benchmark cuFFT 3D r2c/c2r and c2c/c2c transformations (single or double precision).
 *
 * TODO: Include in-place / out-of-place.
 *
 */

#ifdef USE_CUDA

#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <cuda/cuda_runtime_api.h>
#include <cuda/cuda.h>
#include <cuda/cufft.h>
#include <sys/time.h>

#define NX 512
#define NY 256
#define NZ 256

void memtogpu(void *gpu, void *inp, size_t size) {

  if(cudaMemcpy(gpu, inp, size, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Error in memcpy to GPU.\n");
    exit(1);
  }
}

void gputomem(void *inp, void *gpu, size_t size) {

  if(cudaMemcpy(inp, gpu, sizeof(REAL) * NX * NY * NZ, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Error in memcpy from GPU.\n");
    exit(1);
  }
}

int main(int argc, char **argv) {

  void *gpu_in, *gpu_out;
  REAL *inp;
  REAL complex *inp2;
  size_t s = 2 * NX * NY * (NZ / 2 + 1), i;
  cufftHandle f, b;
  struct timeval st, now;

#ifdef SINGLE_PREC
  fprintf(stderr, "Single precision floats.\n");
#else
  fprintf(stderr, "Double precision floats.\n");
#endif
  inp = (REAL *) malloc(sizeof(REAL) * s);
  inp2 = (REAL complex *) malloc(sizeof(REAL complex) * s);

  /* generate random data */
  srand48(123);
  for(i = 0; i < NX * NY * NZ; i++)
    inp[i] = (REAL) drand48();
    
  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  if(cufftPlan3d(&f, NX, NY, NZ, CUFFT_R2C) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
  if(cufftPlan3d(&b, NX, NY, NZ, CUFFT_C2R) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
#else
  if(cufftPlan3d(&f, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
  if(cufftPlan3d(&b, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
#endif
  /* FFTW compatibility mode needed? */
  fprintf(stderr, "Done.\n");

  /* Allocate GPU memory */
  if(cudaMalloc((void **) &gpu_in, sizeof(REAL) * NX * NY * NZ) != cudaSuccess) {
    fprintf(stderr, "Error in malloc.\n");
    exit(1);
  }
  if(cudaMalloc((void **) &gpu_out, sizeof(REAL) * NX * NY * 2 * (NZ/2 + 1)) != cudaSuccess) {
    fprintf(stderr, "Error in malloc.\n");
    exit(1);
  }

  gettimeofday(&st, 0);

  memtogpu(gpu_in, inp, NX * NY * NZ * sizeof(REAL));

  /* in place */
#ifdef SINGLE_PREC
  if(cufftExecR2C(f, (cufftReal *) gpu_in, (cufftComplex *) gpu_out) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (forward).\n");
    exit(1);
  }
#else
  if(cufftExecD2Z(f, (cufftDoubleReal *) gpu_in, (cufftDoubleComplex *) gpu_out) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (forward).\n");
    exit(1);
  }
#endif

  gputomem(inp, gpu_out, sizeof(REAL complex) * NX * NY * (NZ / 2 + 1));

  memtogpu(gpu_out, inp, sizeof(REAL complex) * NX * NY * (NZ / 2 + 1));

#ifdef SINGLE_PREC
  if(cufftExecC2R(b, (cufftComplex *) gpu_out, (cufftReal *) gpu_in) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (inverse).\n");
    exit(1);
  }
#else
  if(cufftExecZ2D(b, (cufftDoubleComplex *) gpu_out, (cufftDoubleReal *) gpu_in) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (inverse).\n");
    exit(1);
  }
#endif

  gputomem(inp, gpu_in, sizeof(REAL) * NX * NY * NZ);
  
  gettimeofday(&now, 0);

  printf("GPU r2c/c2r: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));

  cudaFree(gpu_in);
  cudaFree(gpu_out);
  cufftDestroy(f);
  cufftDestroy(b);

  /****************************/

  /* generate random data */
  srand48(123);
  for(i = 0; i < NX * NY * NZ; i++)
    inp2[i] = (REAL complex) (drand48() + I * drand48());
    
  fprintf(stderr, "Planning.... ");
#ifdef SINGLE_PREC
  if(cufftPlan3d(&f, NX, NY, NZ, CUFFT_C2C) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
#else
  if(cufftPlan3d(&f, NX, NY, NZ, CUFFT_Z2Z) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in plan.\n");
    exit(1);
  }
#endif
  /* FFTW compatibility mode needed? */
  fprintf(stderr, "Done.\n");

  /* Allocate GPU memory */
  if(cudaMalloc((void **) &gpu_in, sizeof(REAL complex) * NX * NY * NZ) != cudaSuccess) {
    fprintf(stderr, "Error in malloc.\n");
    exit(1);
  }
  if(cudaMalloc((void **) &gpu_out, sizeof(REAL complex) * NX * NY * NZ) != cudaSuccess) {
    fprintf(stderr, "Error in malloc.\n");
    exit(1);
  }

  gettimeofday(&st, 0);

  memtogpu(gpu_in, inp2, sizeof(REAL complex) * NX * NY * NZ);

  /* in place */
#ifdef SINGLE_PREC
  if(cufftExecC2C(f, (cufftComplex *) gpu_in, (cufftComplex *) gpu_out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (forward).\n");
    exit(1);
  }
#else
  if(cufftExecZ2Z(f, (cufftDoubleComplex *) gpu_in, (cufftDoubleComplex *) gpu_out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (forward).\n");
    exit(1);
  }
#endif

  gputomem(inp2, gpu_out, sizeof(REAL complex) * NX * NY * NZ);

  memtogpu(gpu_out, inp2, sizeof(REAL complex) * NX * NY * NZ);

  /* in place */
#ifdef SINGLE_PREC
  if(cufftExecC2C(f, (cufftComplex *) gpu_out, (cufftComplex *) gpu_in, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (inverse).\n");
    exit(1);
  }
#else
  if(cufftExecZ2Z(f, (cufftDoubleComplex *) gpu_out, (cufftDoubleComplex *) gpu_in, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    fprintf(stderr, "Error in FFT (inverse).\n");
    exit(1);
  }
#endif

  gputomem(inp2, gpu_in, sizeof(REAL complex) * NX * NY * NZ);
  
  gettimeofday(&now, 0);
  printf("GPU c2c/c2c: Wall time = %le s.\n", 1.0 * (REAL) (now.tv_sec - st.tv_sec) + 1e-6 * (REAL) (now.tv_usec - st.tv_usec));

  cudaFree(gpu_in);
  cudaFree(gpu_out);
  cufftDestroy(f);
  cufftDestroy(b);

  exit(0);
}

#endif
