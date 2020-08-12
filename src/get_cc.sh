#!/bin/bash
if [ -f /usr/bin/cuda-gcc ]; then
# Fedora with negativo17 repo or Debian
  echo /usr/bin/cuda-gcc
else
# Guess that the current gcc is OK...
  echo /usr/bin/gcc
fi
exit 0

