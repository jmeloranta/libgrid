/*
 * Scan through source files and generate documentation.
 *
 * Usage: gendoc file1.c file2.c ...
 *
 * Example:
 *
 * @FUNC{rgrid_multiply, "Multiply grid by a constant"}
 * @DESC{"This function multiplies a grid by a given constant"}
 * @ARG1{rgrid *grid, "Source/Destination grid"}
 * @ARG2{REAL value, "Value for multiplication"}
 * @RVAL{void, "No return value."}
 *
 * will generate LaTeX code for the subsection describing the function.
 *
 * Note: This is not very tolerant for typos...
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *quoted(char *txt) {

  char buf[2048];
  int i, j;
  
  for (i = j = 0; i < strlen(txt); i++) {
    switch (txt[i]) {
      case '_':
        buf[j] = '\\';
        buf[j+1] = '_';
        j += 2;
        break;
      case '&':
        buf[j] = '\\';
        buf[j+1] = '&';
        j += 2;
        break;
      default:
        buf[j] = txt[i];
        j++;
    }
  }
  buf[j] = '\0';
  strcpy(txt, buf);
  return txt;
}

void scan_docs(char *file) {

  FILE *fp;
  char buf[1024], func_name[128], func_doc[1024], func_desc[2048];
  char arg1[128], arg2[128], arg3[128], arg4[128], arg5[128], arg6[128], arg7[128], rval[128];
  char arg1_doc[1024], arg2_doc[1024], arg3_doc[1024], arg4_doc[1024], arg5_doc[1024], arg6_doc[1024], rval_doc[1024];
  char arg7_doc[1024];

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "Cannot open file %s.\n", file);
    exit(1);
  }

  while(!feof(fp)) {
    fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@FUNC{%128[^,], \"%1024[^\"]\"}", func_name, func_doc) != 2) continue;
    // Use fscanf to get multiline description
    if(fscanf(fp, "%*[a-z A-Z*/]@DESC{\"%1024[^\"]\"}\n", func_desc) != 1) {
      fprintf(stderr, "Error reading description for %s.\n", func_name);
      continue;
    }
    fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG1{%128[^,], \"%1024[^\"]\"}", arg1, arg1_doc) != 2) {arg1[0] = 0; arg1_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG2{%128[^,], \"%1024[^\"]\"}", arg2, arg2_doc) != 2) {arg2[0] = 0; arg2_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG3{%128[^,], \"%1024[^\"]\"}", arg3, arg3_doc) != 2) {arg3[0] = 0; arg3_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG4{%128[^,], \"%1024[^\"]\"}", arg4, arg4_doc) != 2) {arg4[0] = 0; arg4_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG5{%128[^,], \"%1024[^\"]\"}", arg5, arg5_doc) != 2) {arg5[0] = 0; arg5_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG6{%128[^,], \"%1024[^\"]\"}", arg6, arg6_doc) != 2) {arg6[0] = 0; arg6_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@ARG7{%128[^,], \"%1024[^\"]\"}", arg7, arg7_doc) != 2) {arg7[0] = 0; arg7_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    if(sscanf(buf, "%*[a-z A-Z*/]@RVAL{%128[^,], \"%1024[^\"]\"}", rval, rval_doc) != 2) {rval[0] = 0; rval_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    printf("\\subsection{Function %s() -- %s}\n", quoted(func_name), quoted(func_doc));
    printf("%s.\\\\\n", quoted(func_desc));
    printf("\\begin{longtable}{p{.43\\textwidth} p{.55\\textwidth}}\n");
    printf("Argument & Description\\\\\n");
    printf("\\cline{1-2}\n");
    if(arg1[0]) printf("%s & %s.\\\\\n", quoted(arg1), quoted(arg1_doc));
    if(arg2[0]) printf("%s & %s.\\\\\n", quoted(arg2), quoted(arg2_doc));
    if(arg3[0]) printf("%s & %s.\\\\\n", quoted(arg3), quoted(arg3_doc));
    if(arg4[0]) printf("%s & %s.\\\\\n", quoted(arg4), quoted(arg4_doc));
    if(arg5[0]) printf("%s & %s.\\\\\n", quoted(arg5), quoted(arg5_doc));
    if(arg6[0]) printf("%s & %s.\\\\\n", quoted(arg6), quoted(arg6_doc));
    if(arg7[0]) printf("%s & %s.\\\\\n", quoted(arg7), quoted(arg7_doc));
    printf("\\end{longtable}\n");
    if(rval[0]) printf("Return value: %s (%s).\n", quoted(rval_doc), quoted(rval));
    printf("\n");
  }

  fclose(fp);
}

int main(int argc, char **argv) {

  int i;

  for(i = 1; i < argc; i++) {
    fprintf(stderr, "Processing file: %s\n", argv[i]);
    scan_docs(argv[i]);
  }

  return 0;
}

