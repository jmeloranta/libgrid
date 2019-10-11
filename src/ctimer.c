/*
 * CPU usage time routines - or, actually, WALL CLOCK TIME.
 * This is better used for judging the parallel performance of the code.
 *
 */

#include "grid.h"

/*
 * Set timer start.
 *
 * timer  = User supplied timer pointer (grid_timer *; input).
 *
 * No return value.
 *
 */

EXPORT void grid_timer_start(grid_timer *timer) {

  gettimeofday(&timer->zero_time, 0);
  timer->zero_clock = clock();
}

/*
 * Get wall clock time elapsed since timer start.
 * 
 * timer  = User supplied timer pointer (grid_timer *; input).
 *
 * Returns wall clock time in seconds.
 *
 */

EXPORT REAL grid_timer_wall_clock_time(grid_timer *timer) {

  struct timeval now;

  gettimeofday(&now, 0);

  return 1.0 * ((REAL) (now.tv_sec - timer->zero_time.tv_sec))
    + 1e-6 * ((REAL) (now.tv_usec - timer->zero_time.tv_usec));
}

/*
 * Get CPU time elapsed since timer start.
 * 
 * timer  = User supplied timer pointer (grid_timer *; input).
 *
 * Returns CPU time in seconds.
 *
 */

EXPORT REAL grid_timer_cpu_time(grid_timer *timer) {

  clock_t now = clock();

  return ((REAL) (now - timer->zero_clock)) / ((REAL) CLOCKS_PER_SEC);
}

