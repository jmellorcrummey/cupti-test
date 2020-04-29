# Report memory corruption when massive threads are launching kernels to different streams

## Observation

We wrote a simple benchmark using CUDA Driver API to test CUPTI's tracing overhead. The benchmark itself is finished without problem, whereas when a simple subscriber is used (`../cupti-preload/subscriber/subscribe.cpp`), we noticed a double-free memory corruption. If `cuCtxDestory` on line 147 (`main.cu`) is commented, the problem disappears. *nvprof* does not report memory corruption with this benchmark.

## Example usage

    ./run.sh

## Environment

CUDA: V10.2.89

NVIDIA DRIVER: 440.33.01

## Error Message

    *** Error in `./main': double free or corruption (fasttop): 0x00000000328f8090 ***
    ======= Backtrace: =========
    /lib64/libc.so.6(cfree+0x49c)[0x7fffab099bec]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x140630)[0x7fffaab30630]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x137fcc)[0x7fffaab27fcc]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x13e2b4)[0x7fffaab2e2b4]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x138818)[0x7fffaab28818]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x3a2a5c)[0x7fffaad92a5c]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x13a998)[0x7fffaab2a998]
    /usr/local/cuda-10.2/extras/CUPTI/lib64/libcupti.so.10.2(+0x2edaec)[0x7fffaacddaec]
    /lib64/libpthread.so.0(+0x8cd4)[0x7fffab3b8cd4]
    /lib64/libc.so.6(clone+0xe4)[0x7fffab127e94]
    ======= Memory map: ========
    10000000-10010000 r-xp 00000000 fd:00 267320                             /home/kz21/Codes/hpctoolkit-gpu-samples/cupti_test/context/main
    10010000-10020000 r--p 00000000 fd:00 267320                             /home/kz21/Codes/hpctoolkit-gpu-samples/cupti_test/context/main
    10020000-10030000 rw-p 00010000 fd:00 267320                             /home/kz21/Codes/hpctoolkit-gpu-samples/cupti_test/context/main
    322c0000-3a990000 rw-p 00000000 00:00 0                                  [heap]
