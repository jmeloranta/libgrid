#!/bin/bash
INC=`fgrep CUDAINC\  ../make.conf | cut -d= -f2`
LIB=`fgrep CUDALIB\  ../make.conf | cut -d= -f2`
cat << E_O_F > sm-det-tmp.c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

int main(int argc, char *argv) {

  int ndev, i;
  struct cudaDeviceProp prop;

  if(cudaGetDeviceCount(&ndev) != cudaSuccess) {
    printf("sm_30\n");
    exit(0);
  }
  i = 0;

  cudaGetDeviceProperties(&prop, i);
  printf("sm_%1d%1d\n", prop.major, prop.minor);
  exit(0);
}
E_O_F
cc -I$INC -o sm-det-tmp sm-det-tmp.c -L$LIB -lcuda -lcudart -lstdc++
./sm-det-tmp
rm sm-det-tmp.c sm-det-tmp
exit 0
