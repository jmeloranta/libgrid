/*
 * Header file for basic complex arthmetic operations for CUDA.
 *
 */

#ifdef SINGLE_PREC
#define CUREAL float
#define CUCOMPLEX cufftComplex
#else
#define CUREAL double
#define CUCOMPLEX cufftDoubleComplex
#endif

#ifdef SINGLE_PREC
#define CUCMUL cuCmulf
#define CUCDIV cuCdivf
#define CUCADD cuCaddf
#define CUCSUB cuCsubf
#define CUCONJ cuConjf
#define CUMAKE make_cuFloatComplex
#define CUCREAL cuCrealf
#define CUCIMAG cuCimagf
#else
#define CUCMUL cuCmul
#define CUCDIV cuCdiv
#define CUCADD cuCadd
#define CUCSUB cuCsub
#define CUCONJ cuConj
#define CUMAKE make_cuDoubleComplex
#define CUCREAL cuCreal
#define CUCIMAG cuCimag
#endif

/* Overload operators */

__device__ inline CUCOMPLEX operator*(CUCOMPLEX a, CUCOMPLEX b) { return CUCMUL(a,b); }
__device__ inline CUCOMPLEX operator*(CUCOMPLEX a, REAL b) { return CUMAKE(a.x * b, a.y * b); }
__device__ inline CUCOMPLEX operator*(REAL a, CUCOMPLEX b) { return CUMAKE(b.x * a, b.y * a); }

__device__ inline CUCOMPLEX operator+(CUCOMPLEX a, CUCOMPLEX b) { return CUCADD(a,b); }
__device__ inline CUCOMPLEX operator+(REAL a, CUCOMPLEX b) { return CUMAKE(a + b.x, b.y); }
__device__ inline CUCOMPLEX operator+(CUCOMPLEX a, REAL b) { return CUMAKE(a.x + b, a.y); }
__device__ inline CUCOMPLEX operator+(CUCOMPLEX a) { return a; }

__device__ inline CUCOMPLEX operator-(CUCOMPLEX a, CUCOMPLEX b) { return CUCSUB(a, b); }
__device__ inline CUCOMPLEX operator-(CUCOMPLEX a, REAL b) { return CUMAKE(a.x - b, a.y); }
__device__ inline CUCOMPLEX operator-(REAL a, CUCOMPLEX b) { return CUMAKE(a - b.x, -b.y); }
__device__ inline CUCOMPLEX operator-(CUCOMPLEX a) { return CUMAKE(-a.x, -a.y); }

__device__ inline CUCOMPLEX operator/(CUCOMPLEX a, CUCOMPLEX b) { return CUCDIV(a,b); }
__device__ inline CUCOMPLEX operator/(CUCOMPLEX a, REAL b) { return CUMAKE(a.x / b, a.y / b); }
__device__ inline CUCOMPLEX operator/(REAL a, CUCOMPLEX b) { return CUMAKE(a * b.x / (b.x * b.x + b.y * b.y), -a * b.y / (b.x * b.x + b.y * b.y)); }

__device__ inline CUREAL CUCABS(CUCOMPLEX val) {

  CUREAL a = val.x, b = val.y, aa = FABS(a), bb = FABS(b);

  if(aa >= bb) return aa * SQRT(1.0 + b * b / (a * a));
  else return bb * SQRT(1.0 + a * a / (b * b));
//  return SQRT(val.x * val.x + val.y * val.y); (fast but not stable)
}

__device__ inline CUREAL CUCARG(CUCOMPLEX val) {

  return ATAN2(val.y, val.x);
}

__device__ inline CUCOMPLEX CUCLOG(CUCOMPLEX val) {

  CUCOMPLEX rv;

// clog(z) = log(cabs(z)) + I * carg(z)
  rv.x = LOG(CUCABS(val));
  rv.y = CUCARG(val);
  return rv; 
}

__device__ inline CUCOMPLEX CUCEXP(CUCOMPLEX val) {

  CUREAL tmp = EXP(val.x);

  val.x = tmp * COS(val.y);
  val.y = tmp * SIN(val.y);
  return val;
}

__device__ inline CUCOMPLEX CUCPOW(CUCOMPLEX val, CUREAL exp) {

  CUCOMPLEX tmp = CUCLOG(val);

  tmp.x *= exp;
  tmp.y *= exp;
  return CUCEXP(tmp);
}

__device__ inline CUCOMPLEX CUCMULR(CUCOMPLEX cval, CUREAL rval) {

  cval.x *= rval;
  cval.y *= rval;
  return cval;
}

__device__ inline CUCOMPLEX CUCMULI(CUCOMPLEX cval, CUREAL ival) {

  CUREAL tmp;

  tmp = -cval.y * ival;
  cval.y = cval.x * ival;
  cval.x = tmp;
  return cval;
}

__device__ inline CUCOMPLEX CUCADDR(CUCOMPLEX cval, CUREAL rval) {

  cval.x += rval;
  return cval;
}

__device__ inline CUCOMPLEX CUCADDI(CUCOMPLEX cval, CUREAL ival) {

  cval.y += ival;
  return cval;
}

__device__ inline CUCOMPLEX CUCSUBR(CUCOMPLEX cval, CUREAL rval) {

  cval.x -= rval;
  return cval;
}

__device__ inline CUCOMPLEX CUCSUBI(CUCOMPLEX cval, CUREAL ival) {

  cval.y -= ival;
  return cval;
}

__device__ inline CUREAL CUCSQNORM(CUCOMPLEX val) {

  return val.x * val.x + val.y * val.y;
}
