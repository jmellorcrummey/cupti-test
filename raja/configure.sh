#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-73810.0
##
## All rights reserved.
## 
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

TEST_DIR=rajaperf/cupti-test
CUDA=/usr/local/cuda-10.0
CUB=cub-1.8.0
ROOT=`pwd`

rm -rf $TEST_DIR > /dev/null
mkdir $TEST_DIR && cd $TEST_DIR


cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCUB_PATH="-I$ROOT/$CUB" \
   -C $ROOT/raja-config \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA} \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  ..
