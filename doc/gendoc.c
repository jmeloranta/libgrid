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

#define ARGMAX 21

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
  char buf[1024], fmt[1024], func_name[128], func_doc[1024], func_desc[2048];
  char args[ARGMAX][128], args_doc[ARGMAX][1024], rval[128], rval_doc[2048];
  int i, j;

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
    for(i = 0; i < ARGMAX; i++) {
      strcpy(fmt, "%*[a-z A-Z*/]@ARG");
      sprintf(fmt + strlen(fmt), "%d", i+1);
      strcat(fmt, "{%128[^,], \"%1024[^\"]\"}");
      if(sscanf(buf, fmt, args[i], args_doc[i]) != 2) {args[i][0] = 0; args_doc[i][0] = 0; break;}
      else fgets(buf, sizeof(buf), fp);
    }
    if(sscanf(buf, "%*[a-z A-Z*/]@RVAL{%128[^,], \"%1024[^\"]\"}", rval, rval_doc) != 2) {rval[0] = 0; rval_doc[0] = 0;}
    else fgets(buf, sizeof(buf), fp);
    printf("\\subsection{Function %s() -- %s}\n", quoted(func_name), quoted(func_doc));
    printf("%s.\\\\\n", quoted(func_desc));
    if(args[0][0]) {
      printf("\\begin{longtable}{p{.43\\textwidth} p{.55\\textwidth}}\n");
      printf("Argument & Description\\\\\n");
      printf("\\cline{1-2}\n");
      for (j = 0; j < i; j++)
        if(args[j][0]) printf("%s & %s.\\\\\n", quoted(args[j]), quoted(args_doc[j]));
      printf("\\end{longtable}\n");
    }
    if(rval[0]) printf("\\noindent\nReturn value: %s (%s).\n", quoted(rval_doc), quoted(rval));
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

