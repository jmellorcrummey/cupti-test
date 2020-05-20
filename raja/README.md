This directory contains a test that illustrates the overhead of CUPTI PC Sampling when measuring a kernel from the RAJA perf suite.

The directory contains a Makefile for 
- downloading the raja perf suite, configuring it, and building it
- building a shared library that enables CUPTI PC sampling in a constructor and turns it off in a destructor
- running the MULADDSUB kernel from the RAJA perf suite without CUPTI
- running the MULADDSUB kernel from the RAJA perf suite with CUPTI PC sampling enabled with a low sampling frequency

On an x86_64 system equipped with a V100, without CUPTI the MULADDSUB kernel runs in ~3.5 seconds. 
With CUPTI PC Sampling enabled, the kernel takes ~10 minutes to complete.

A reference typescript of our test results along with information about the x86_64 CPU on 
which it was performed is included in the file rice-test-results.txt. 


Performing the test:

	make prepare 			# download, configure, and build the software:
	make test 			# run the RAJA MULADDSUB kernel without CUPTI and then with CUPTI PC sampling
