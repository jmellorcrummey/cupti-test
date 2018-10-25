## Report cupti (cuda-9.2 and cuda-10.0) attributes samples back to an incorrect pc

### Observations

It has been found that start from cuda-9.2, some pc-sampling samples may be attributed to the last `BRA` instruction that jumps to itself. And because `EXIT` is placed before the last `BRA`, there's no reason to attribute any sample back to it.

### Usage

**sm_60 GPU**

    git clone --recursive https://github.com/Jokeren/hpctoolkit-gpu-samples
    cd hpctoolkit-gpu-samples/cupti_test/pc_offset
    ./run_sm_60.sh
  
check if the offset of the last BRAs in *sm_60_cuda_9.2.sass* and *sm_60_cuda_10.0.sass* can be found in *sm_60_cuda_9.2.log* and *sm_60_cuda_10.0.log* accordingly.

**sm_70 GPU**
    
    git clone --recursive https://github.com/Jokeren/hpctoolkit-gpu-samples
    cd hpctoolkit-gpu-samples/cupti_test/pc_offset
    ./run_sm_70.sh
  
check if the offset of the last BRAs in *sm_70_cuda_9.2.sass* and *sm_70_cuda_10.0.sass* can be found in *sm_70_cuda_9.2.log* and *sm_70_cuda_10.0.log* accordingly.

**Check**

If the last **BRA** is:

    /*0150*/                   BRA `(.L_1);
        
Please search `0x150` in the log file.
