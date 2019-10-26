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
  FILE *fp, *fp2;
  REAL x, xmin = 1E99, xmax = -1E99, y, ymin = 1E99, ymax = -1E99;
  char buf[512], buf2[512], s1, s2;

  if(argc != 3) {
    fprintf(stderr, "Usage: view3 file1 file2\n");
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
  if(!(fp2 = fopen(argv[2], "r"))) {
    fprintf(stderr, "Can't open file %s.\n", argv[2]);
    exit(1);
  }
  nsteps = 0;
  s1 = s2 = 0;
  for(i = 0; !feof(fp); i++) {
    if(!s1) fgets(buf, sizeof(buf), fp);    
    if(!s2) fgets(buf2, sizeof(buf2), fp2);    
    if(buf[0] == '\n') s1 = 1;
    if(buf2[0] == '\n') s2 = 1;
    if(s1 == 1 && s2 == 1) {
      GracePrintf("title \"%d\"", nsteps);
      GracePrintf("s0 linestyle 0");
      GracePrintf("s0 symbol 1");
      GracePrintf("s0 symbol color 1");
      GracePrintf("s0 symbol size 0.3");
      GracePrintf("s0 symbol linewidth 2");
      GracePrintf("s1 linestyle 0");
      GracePrintf("s1 symbol color 2");
      GracePrintf("s1 symbol 1");
      GracePrintf("s1 symbol size 0.3");
      GracePrintf("s1 symbol linewidth 2");
      GracePrintf("redraw");
      if(nsteps == 0) sleep(5);
      else usleep(300000);
      GracePrintf("kill g0.s0 SAVEALL");     
      GracePrintf("kill g0.s1 SAVEALL");     
      nsteps++;
      s1 = s2 = 0;
      continue;
    }
    if(!s1) {
      if(sscanf(buf, FMT_R " " FMT_R "\n", &x, &y) != 2) break;
      GracePrintf("g0.s0 point " FMT_R "," FMT_R, x, y);
    }
    if(!s2) {
      if(sscanf(buf2, FMT_R " " FMT_R "\n", &x, &y) != 2) break;
      GracePrintf("g0.s1 point " FMT_R "," FMT_R, x, y);
    }
  }
  fclose(fp);
  fclose(fp2);
  return 0;
}
