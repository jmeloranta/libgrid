/*
 * Main file to be included from applications using libgrid.
 * So, each program should have #include <grid/grid.h> at the beginning.
 *
 */

#ifndef __GRID__
#define __GRID__
#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <math.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <complex.h>

#include <time.h>
#include <sys/time.h>

#include <fftw3.h>

#ifdef USE_CUDA
#include <cuComplex.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda.h"
#include <curand.h>

#ifdef SINGLE_PREC
#define CUREAL float
#define CUCOMPLEX cufftComplex
#else
#define CUREAL double
#define CUCOMPLEX cufftDoubleComplex
#endif
#endif

#include "structs.h"
#include "proto.h"
#include "defs.h"

/* get rid of the identifier used for extracting prototypes */
#define EXPORT

/* Formatting strings */

#if defined(SINGLE_PREC)
#define FMT_R "%e"
#elif defined(DOUBLE_PREC)
#define FMT_R "%le"
#elif defined(QUAD_PREC)
#define FMT_R "%Le"
#endif

#ifdef SHORT_INT
#define FMT_I "%d"
#else
#define FMT_I "%ld"
#endif

#endif /* __GRID__ */
