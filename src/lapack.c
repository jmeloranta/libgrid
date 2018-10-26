/*
 * Wrappers for LAPACK (optional).
 *
 */

#include "grid.h"
#include "private.h"

#ifdef USE_LAPACK

void zhegv_(int *, char *, char *, int *, double complex *, int *, double complex *, int *, double *, double complex *, int *, double *, int *);
void zheev_(char *, char *, int *, double complex *, int *, double *, double complex *, int *, double *, int *);
void chegv_(int *, char *, char *, int *, float complex *, int *, float complex *, int *, float *, float complex *, int *, float *, int *);
void cheev_(char *, char *, int *, float complex *, int *, float *, float complex *, int *, float *, int *);

/* NOTE: These are not parallel versions !!! -- parallelization is most often achieved at a higher level */
/* Are these even thread safe? */

/*
 * Solve generalized hermitian matrix eigenvalue problem. See the manual page for c/zhegv (LAPACK).
 *
 * eigenvalue  = Vector containing the eigenvalues (REAL *; output).
 * hamiltonian = Matrix for which the eigen states are solved for and eigenvectors on exit (REAL complex *; input/output).
 * overlap     = Overlap matrix (REAL *; input - overwritten on exit).
 * states      = Number of eigen states sought (INT).
 *
 * Returns zhegv exit status (info).
 *
 */

EXPORT INT grid_generalized_hermitian_eigenvalue_problem(REAL *eigenvalue, REAL complex *hamiltonian, REAL complex *overlap, INT states) {

  int itype, n, lda, ldb, lwork, info;
  char jobz, uplo;
  REAL complex *a, *b, *work;
  REAL *w, *rwork;
  
  itype = 1;
  jobz = 'V';
  uplo = 'U';
  n = (int) states;
  lda = n;
  ldb = n;
  a = hamiltonian;
  b = overlap;
  w = eigenvalue;
  lwork = 2 * n;
  work = (REAL complex *) malloc(((size_t) lwork) * sizeof(REAL complex));
  rwork = (REAL *) malloc(((size_t) (3 * n - 2)) * sizeof(REAL));
  
  if (!work || !rwork) return -999;
  
#if defined(SINGLE_PREC)
  chegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
#elif defined(DOUBLE_PREC)
  zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
#elif defined(QUAD_PREC)
#warning "No quad precision for Xhegv - using dummy routine."
#endif  

  return info;
}

/*
 * Solve hermitian matrix eigenvalue problem. See the manual page for c/zheev (LAPACK).
 *
 * eigenvalue  = Vector containing the eigenvalues (REAL *; output).
 * hamiltonian = Matrix for which the eigen states are solved for and eigenvectors on exit (REAL complex *; input/output).
 * states      = Number of eigen states sought (INT).
 *
 * Returns zhegv exit status (info).
 *
 */

EXPORT INT grid_hermitian_eigenvalue_problem(REAL *eigenvalue, REAL complex *hamiltonian, INT states) {

  int n, lda, lwork, info;
  char jobz, uplo;
  REAL complex *a, *work;
  REAL *w, *rwork;
  
  jobz = 'V';
  uplo = 'U';
  n = (int) states;
  lda = n;
  a = hamiltonian;
  w = eigenvalue;
  lwork = 2 * n;
  work = (REAL complex *) malloc(((size_t) lwork) * sizeof(REAL complex));
  rwork = (REAL *) malloc(((size_t) (3 * n - 2)) * sizeof(REAL));
  
  if (!work || !rwork) return -999;
  
#if defined(SINGLE_PREC)
  cheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
#elif defined(DOUBLE_PREC)
  zheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
#elif defined(QUAD_PREC)
#warning "No quad precision for Xheev - using dummy routine."
#endif
  
  return info;
}

#endif
