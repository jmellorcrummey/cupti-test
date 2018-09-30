This directory contains a test that illustrates the overhead of CUPTI PC Sampling when measuring a kernel from the RAJA perf suite.

The directory contains a Makefile for 
- downloading the raja perf suite, configuring it, and building it
- building a shared library that enables CUPTI PC sampling in a constructor and turning it off in a destructor
- running the MULADDSUB kernel from the RAJA perf suite without CUPTI
- running the MULADDSUB kernel from the RAJA perf suite with CUPTI PC sampling enabled with a low sampling frequency

On an x86_64 system equipped with a V100, without CUPTI the MULADDSUB kernel runs in ~4 secondsa. With CUPTI PC Sampling enabled, the kernel takes ~10 minutes to complete.


Performing the tesst:

	make prepare 			# download, configure, and build the software:
	make test 			# run the RAJA MULADDSUB kernel without CUPTI in a few seconds and then with PC sampling for ~10 minutes
