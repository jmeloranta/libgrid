/*
 * General FFTW routines (wisdom related routines).
 *
 */

#include "grid.h"

/*
 * Return default wisdom file name.
 *
 */

static char *grid_fft_wisfile() {

  char hn[128];
  static char *buf = NULL;

  if(buf == NULL && !(buf = (char *) malloc(128))) {
    fprintf(stderr, "libgrid: memory allocation failure (grid_fft_wisfile).\n");
    abort();
  }
  gethostname(hn, sizeof(hn));
  sprintf(buf, "fftw-%s.wis", hn);  
  return buf;
}

/*
 * FFTW Wisdom interface - import wisdom.
 *
 * file = file name for reading wisdom data (input; char *).
 *        If NULL pointer, default file name will be used.
 *
 * No return value.
 *
 */

EXPORT void grid_fft_read_wisdom(char *file) {

  if(!file) file = grid_fft_wisfile();
  /* Attempt to use wisdom (FFTW) from previous runs */
#if defined(SINGLE_PREC)
  if(fftwf_import_wisdom_from_filename(file) == 1) {
#elif defined(DOUBLE_PREC)
  if(fftw_import_wisdom_from_filename(file) == 1) {
#elif defined(QUAD_PREC)
  if(fftwl_import_wisdom_from_filename(file) == 1) {
#endif
    fprintf(stderr, "libgrid: Using wisdom stored in %s.\n", file);
  } else fprintf(stderr, "libgrid: No existing wisdom file.\n");
}

/*
 * FFTW Wisdom interface - export wisdom.
 *
 * file = file name for saving wisdom data (input, char *).
 *        If NULL pointer, default file name will be used.
 *
 * No return value.
 *
 */

EXPORT void grid_fft_write_wisdom(char *file) {

  if(!file) file = grid_fft_wisfile();
#if defined(SINGLE_PREC)
  fftwf_export_wisdom_to_filename(file);
#elif defined(DOUBLE_PREC)
  fftw_export_wisdom_to_filename(file);
#elif defined(QUAD_PREC)
  fftwl_export_wisdom_to_filename(file);
#endif
  fprintf(stderr, "libgrid: Wisdom file %s written.\n", file);
}
