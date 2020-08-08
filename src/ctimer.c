/*
 * Time measurement routines.
 *
 */

#include "grid.h"

/*
 * @FUNC{grid_timer_start, "Start clock timer"}
 * @DESC{"Set timer start. To measure the wall clock time spent:\\
         grid_timer timer;\\
         grid_timer_start(&timer);\\
         ...\\
         seconds = grid_timer_wall_clock_time(&timer);\\
         To get CPU time, replace the above line with:\\
         seconds = grid_timer_cpu_time(&timer);\\
         For CUDA, only the wall clock time measurement makes sense"}
 * @ARG1{grid_timer *timer, "User supplied timer pointer"}
 * RVAL{void, "No return value"}
 *
 */

EXPORT void grid_timer_start(grid_timer *timer) {

  gettimeofday(&timer->zero_time, 0);
  timer->zero_clock = clock();
}

/*
 * @FUNC{grid_timer_wall_clock_time, "Measure wall clock timer"}
 * @DESC{"Get wall clock time elapsed since timer start (see grid_timer_start)"}
 * @ARG1{grid_timer *timer, "User supplied timer pointer"}
 * @RVAL{REAL, "Returns wall clock time in seconds"}
 *
 */

EXPORT REAL grid_timer_wall_clock_time(grid_timer *timer) {

  struct timeval now;

  gettimeofday(&now, 0);

  return 1.0 * ((REAL) (now.tv_sec - timer->zero_time.tv_sec))
    + 1e-6 * ((REAL) (now.tv_usec - timer->zero_time.tv_usec));
}

/*
 * @FUNC{grid_timer_cpu_time, "Measure CPU timer"}
 * @DESC{"Get CPU time elapsed since timer start (see grid_timer_start)"}
 * @ARG1{grid_timer *timer, "User supplied timer pointer"}
 * @RVAL{REAL, "Returns CPU time in seconds"}
 *
 */

EXPORT REAL grid_timer_cpu_time(grid_timer *timer) {

  clock_t now = clock();

  return ((REAL) (now - timer->zero_clock)) / ((REAL) CLOCKS_PER_SEC);
}
