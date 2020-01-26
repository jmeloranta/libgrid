/*
 * Some global parameters.
 *
 */

#include "au.h"

#define GRID_EPS 1.0E-16  // Small epsilon

#define GRID_ABS_BC_FFT (3.0E4 / GRID_AUTOK) // Absorbing boundary base value for FFT

#define CGRID_DIRICHLET_BOUNDARY cgrid_value_outside_dirichlet
#define CGRID_NEUMANN_BOUNDARY cgrid_value_outside_neumann
#define CGRID_PERIODIC_BOUNDARY cgrid_value_outside_periodic

#define RGRID_DIRICHLET_BOUNDARY rgrid_value_outside_dirichlet
#define RGRID_NEUMANN_BOUNDARY rgrid_value_outside_neumann
#define RGRID_PERIODIC_BOUNDARY rgrid_value_outside_periodic

/* Wavefunction boundary conditions */
#define WF_DIRICHLET_BOUNDARY     1
#define WF_NEUMANN_BOUNDARY       2
#define WF_PERIODIC_BOUNDARY      3

/* Wavefunction propagators */
#define WF_2ND_ORDER_FFT	1
#define WF_4TH_ORDER_FFT	2
#define WF_2ND_ORDER_CFFT	3
#define WF_4TH_ORDER_CFFT	4
#define WF_2ND_ORDER_CN		5
#define WF_4TH_ORDER_CN		6
