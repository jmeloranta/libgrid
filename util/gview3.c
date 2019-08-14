/*
 * Show 2D (x,y) plot animation.
 *
 * Usage: view file1 file2...
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

  INT i, nsteps;
  FILE *fp;
  REAL x, xmin = 1E99, xmax = -1E99, y, ymin = 1E99, ymax = -1E99;
  char buf[512];

  if(argc != 2) {
    fprintf(stderr, "Usage: view3 file\n");
    exit(1);
  }

  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Can't open file %s.\n", argv[1]);
    exit(1);
  }

  for (i = 0; ; i++) {
    if(feof(fp)) break;
    if(fscanf(fp, " " FMT_R " " FMT_R, &x, &y) != 2) continue;
    if(y > ymax) ymax = y;
    if(y < ymin) ymin = y;
    if(x > xmax) xmax = x;
    if(x < xmin) xmin = x;
  }
  fclose(fp);
  printf("Region [" FMT_R "X" FMT_R "] x [" FMT_R "X" FMT_R "].\n", xmin, xmax, ymin, ymax);

  GraceRegisterErrorFunction(my_error_function);
  /* Start Grace with a buffer size of 2048 and open the pipe */
  if (GraceOpenVA("/usr/bin/xmgrace", 2048, "-free", "-nosigcatch","-geometry", "930x730", NULL) == -1) {
    fprintf(stderr, "Can't start Grace. \n");
    exit(1);
  }

  /* Send some initialization commands to Grace */
  GracePrintf("world xmin " FMT_R, xmin);
  GracePrintf("world xmax " FMT_R, xmax);
  GracePrintf("world ymin " FMT_R, ymin);
  GracePrintf("world ymax " FMT_R, ymax);

  GracePrintf("xaxis tick major " FMT_R, (xmax - xmin) / 10.0);
  GracePrintf("xaxis tick minor " FMT_R, (xmax - xmin) / 20.0);
  GracePrintf("yaxis tick major " FMT_R, (ymax - ymin) / 10.0);
  GracePrintf("yaxis tick minor " FMT_R, (ymax - ymin) / 20.0);  
  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Can't open file %s.\n", argv[1]);
    exit(1);
  }
  nsteps = 0;
  for(i = 0; !feof(fp); i++) {
    fgets(buf, sizeof(buf), fp);    
    if(buf[0] == '\n') {
      GracePrintf("title \"%d\"", nsteps);
      GracePrintf("s0 linestyle 0");
      GracePrintf("s0 symbol 1");
      GracePrintf("s0 symbol size 0.3");
      GracePrintf("s0 symbol linewidth 2");
      GracePrintf("redraw");
      if(nsteps == 0) sleep(5);
      else usleep(300000);
      GracePrintf("kill g0.s0 SAVEALL");     
      nsteps++;
      continue;
    }
    if(sscanf(buf, FMT_R " " FMT_R "\n", &x, &y) != 2) break;
    GracePrintf("g0.s0 point " FMT_R "," FMT_R, x, y);
  }
  fclose(fp);
  return 0;
}
