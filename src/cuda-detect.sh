#!/bin/bash
if `/usr/bin/which -q nvcc > /dev/null 2>&1`; then 
  echo "yes"
else
  echo "no"
fi
