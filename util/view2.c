/*
 * Show 1-D animation using xmgrace.
 *
 * Usage: view file1 file2...
 *
 * Movies can be made with ffmpeg:
 * ffmpeg -f image2 -framerate 12 -i "movie-%05d.jpg" -s 800x600 movie.avi
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <grace_np.h>
#include <grid/grid.h>

void my_error_function(const char *msg) {
  fprintf(stderr, "library message: \"%s\"\n", msg);
}

int main(int argc, char **argv) {

  INT i, j, npts, save;
  FILE *fp;
  REAL x, x_prev, y, begin, end, step = -1.0, ymax = -1.0, ymin = 0.0;

  if(argv[1][0] == '-' && argv[1][1] == 's') {
    argv = &argv[1];
    save = 1;
    argc--;
  } else save = 0;

  if(argc < 2) {
    fprintf(stderr, "Usage: view file1 file2...\n");
    exit(1);
  }

  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Can't open file %s.\n", argv[1]);
    exit(1);
  }
  /* First pass to figure out grid step length, begin and end. */
  for (i = 0; ; i++) {
    if(fscanf(fp, " " FMT_R " " FMT_R, &x, &y) != 2) break;
    if(y > ymax) ymax = 1.5 * y;
    if(!i) begin = x;
    if(i && step == -1.0) step = x - x_prev;
    if(feof(fp)) break;
    x_prev = x;
  }
  end = x_prev;
  step = FABS(step);
  fclose(fp);

  npts = 1 + (int) (0.5 + ((end - begin) / step));

  printf("begin = " FMT_R ", end = " FMT_R ", step = "FMT_R ", ymax = " FMT_R ", ymin = " FMT_R ".\n", begin, end, step, ymax, ymin);

  GraceRegisterErrorFunction(my_error_function);
  /* Start Grace with a buffer size of 2048 and open the pipe */
  if (GraceOpenVA("/usr/bin/xmgrace", 2048, "-free", "-nosafe", "-nosigcatch","-geometry", "930x730", NULL) == -1) {
    fprintf(stderr, "Can't start Grace. \n");
    exit(1);
  }

  /* Send some initialization commands to Grace */
  GracePrintf("world xmin " FMT_R, begin);
  GracePrintf("world xmax " FMT_R, end);
  GracePrintf("world ymin " FMT_R, ymin);
  GracePrintf("world ymax " FMT_R, ymax);

  GracePrintf("xaxis tick major " FMT_R, (end - begin) / 10.0);
  GracePrintf("xaxis tick minor " FMT_R, (end - begin) / 20.0);
  GracePrintf("yaxis tick major " FMT_R, ymax / 10.0);
  GracePrintf("yaxis tick minor " FMT_R, ymax / 20.0);

  printf("# of files = %d\n", argc);
  for (i = 1; i < argc; i++) {

    if(!(fp = fopen(argv[i], "r"))) {
      fprintf(stderr, "Can't open file %s.\n", argv[i]);
      exit(1);
    }

    if(i > 1)
      GracePrintf("kill g0.s0");
    for (j = 0; j < npts; j++) {
      char buf[128];
      if(fscanf(fp, " " FMT_R " " FMT_R, &x, &y) != 2) {
	fprintf(stderr, "File format error.\n");
	exit(1);
      }
      sprintf(buf, "File = " FMT_I, i);
      GracePrintf("title \"%s\"", buf);
      GracePrintf("g0.s0 point " FMT_R "," FMT_R, x, y);
    }
    GracePrintf("redraw");
    if(save) {
      GracePrintf("HARDCOPY DEVICE \"JPEG\"");
      GracePrintf("PRINT TO \"movie-%05d.jpg\"", i);
      GracePrintf("PRINT");
      
    }
    if(i == 1) sleep(5);
    else usleep(300000);
    fclose(fp);
  }
  return 0;
}
