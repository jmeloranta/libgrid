/*
 * Some global parameters.
 *
 */

#include "au.h"

#define GRID_EPS 1.0E-16  // Small epsilon
#define GRID_EPS2 1E-5    // Large epsilon (for dividing by density)

#define GRID_ABS_BC_CN (10.0 / GRID_AUTOFS)  // Absorbing boundary base value for CN
#define GRID_ABS_BC_FFT (3.0E4 / GRID_AUTOK) // Absorbing boundary base value for FFT

#define CGRID_DIRICHLET_BOUNDARY cgrid_value_outside_constantdirichlet
#define CGRID_NEUMANN_BOUNDARY cgrid_value_outside_neumann
#define CGRID_PERIODIC_BOUNDARY cgrid_value_outside

#define RGRID_DIRICHLET_BOUNDARY rgrid_value_outside_constantdirichlet
#define RGRID_NEUMANN_BOUNDARY rgrid_value_outside_neumann
#define RGRID_PERIODIC_BOUNDARY rgrid_value_outside

/* Special boundaries for vortex solutions in superfluid helium */
#define CGRID_VORTEX_X_BOUNDARY cgrid_value_outside_vortex_x
#define CGRID_VORTEX_Y_BOUNDARY cgrid_value_outside_vortex_y
#define CGRID_VORTEX_Z_BOUNDARY cgrid_value_outside_vortex_z
/* Warning: The above DO NOT give the usual periodic boundaries */

#define WF_DIRICHLET_BOUNDARY     1
#define WF_NEUMANN_BOUNDARY       2
#define WF_PERIODIC_BOUNDARY      3
#define WF_VORTEX_X_BOUNDARY      4
#define WF_VORTEX_Y_BOUNDARY      5
#define WF_VORTEX_Z_BOUNDARY      6

#define WF_2ND_ORDER_FFT	1
#define WF_4TH_ORDER_FFT	2
#define WF_2ND_ORDER_CFFT	3
#define WF_4TH_ORDER_CFFT	4
#define WF_2ND_ORDER_CN		5
#define WF_4TH_ORDER_CN		6
