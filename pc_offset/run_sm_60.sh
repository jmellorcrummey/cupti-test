#!/bin/bash

echo "Configure cuda-10.0"
cd ../cupti-preload
make clean
make P=/usr/local/cuda-10.0/extras/CUPTI C=/usr/local/cuda-10.0
cd -
make clean
make ARCH=sm_60 CUDA_DIR=/usr/local/cuda-10.0
CUPTI_SAMPLING_PERIOD=4 ../cupti-preload/enablesampling ./main &> sm_60_cuda_10.0.log
nvdisasm vecAdd.cubin &> sm_60_cuda_10.0.sass

echo "Configure cuda-9.2"
cd ../cupti-preload
make clean
make P=/usr/local/cuda-9.2/extras/CUPTI C=/usr/local/cuda-9.2
cd -
make clean
make ARCH=sm_60 CUDA_DIR=/usr/local/cuda-9.2
CUPTI_SAMPLING_PERIOD=4 ../cupti-preload/enablesampling ./main &> sm_60_cuda_9.2.log
nvdisasm vecAdd.cubin &> sm_60_cuda_9.2.sass
