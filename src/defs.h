/*
 * Some global parameters.
 *
 */

#include "au.h"

#define GRID_EPS 1.0E-16  // Small epsilon
#define GRID_EPS2 1E-5    // Large epsilon (for dividing by density)

#define GRID_ABS_BC_FFT (3.0E4 / GRID_AUTOK) // Absorbing boundary base value for FFT

#define CGRID_DIRICHLET_BOUNDARY cgrid_value_outside_dirichlet
#define CGRID_NEUMANN_BOUNDARY cgrid_value_outside_neumann
#define CGRID_PERIODIC_BOUNDARY cgrid_value_outside_periodic
/* sin/cos transform compatible boundaries */
#define CGRID_FFT_EEE_BOUNDARY cgrid_value_outside_fft_eee
#define CGRID_FFT_OEE_BOUNDARY cgrid_value_outside_fft_oee
#define CGRID_FFT_EOE_BOUNDARY cgrid_value_outside_fft_eoe
#define CGRID_FFT_EEO_BOUNDARY cgrid_value_outside_fft_eeo
#define CGRID_FFT_OOE_BOUNDARY cgrid_value_outside_fft_ooe
#define CGRID_FFT_EOO_BOUNDARY cgrid_value_outside_fft_eoo
#define CGRID_FFT_OEO_BOUNDARY cgrid_value_outside_fft_oeo
#define CGRID_FFT_OOO_BOUNDARY cgrid_value_outside_fft_ooo

/* Special boundaries for vortex solutions (compatibility) */
#define CGRID_VORTEX_X_BOUNDARY cgrid_value_outside_fft_eoo
#define CGRID_VORTEX_Y_BOUNDARY cgrid_value_outside_fft_oeo
#define CGRID_VORTEX_Z_BOUNDARY cgrid_value_outside_fft_ooe

#define RGRID_DIRICHLET_BOUNDARY rgrid_value_outside_dirichlet
#define RGRID_NEUMANN_BOUNDARY rgrid_value_outside_neumann
#define RGRID_PERIODIC_BOUNDARY rgrid_value_outside_periodic

/* Wavefunction boundary conditions */
#define WF_DIRICHLET_BOUNDARY     1
#define WF_NEUMANN_BOUNDARY       2
#define WF_PERIODIC_BOUNDARY      3
#define WF_FFT_EEE_BOUNDARY       4
#define WF_FFT_OEE_BOUNDARY       5
#define WF_FFT_EOE_BOUNDARY       6
#define WF_FFT_EEO_BOUNDARY       7
#define WF_FFT_OOE_BOUNDARY       8
#define WF_FFT_EOO_BOUNDARY       9
#define WF_FFT_OEO_BOUNDARY      10
#define WF_FFT_OOO_BOUNDARY      11
/* For compatibility */
#define WF_VORTEX_X_BOUNDARY     9
#define WF_VORTEX_Y_BOUNDARY     10
#define WF_VORTEX_Z_BOUNDARY     8

/* Wavefunction propagators */
#define WF_2ND_ORDER_FFT	1
#define WF_4TH_ORDER_FFT	2
#define WF_2ND_ORDER_CFFT	3
#define WF_4TH_ORDER_CFFT	4
#define WF_2ND_ORDER_CN		5
#define WF_4TH_ORDER_CN		6
