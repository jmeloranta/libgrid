#!/bin/bash
if [ -f /usr/bin/cuda-gcc ]; then
# Fedora with negativo17 repo
  echo /usr/bin/cuda-gcc
else
# Debian Buster with testing repo
  echo /usr/bin/gcc-8
fi
exit 0

