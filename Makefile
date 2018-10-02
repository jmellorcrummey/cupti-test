CUBTGZ=v1.8.0.tar.gz
all:
	@echo make targets: prepare, test, distclean

prepare: fetch configure build

fetch: rajaperf cub

rajaperf:
	git clone https://github.com/llnl/rajaperf
	cd rajaperf; git submodule init && git submodule update

cub:
	wget https://github.com/NVlabs/cub/archive/$(CUBTGZ)
	tar xf v1.8.0.tar.gz

configure: 
	./configure.sh

build:
	cd cupti-preload; make
	cd rajaperf/cupti-test; make -j

test:
	@echo '******* time raja perf suite kernel MULADDSUB without CUPTI ********'
	time rajaperf/cupti-test/bin/raja-perf.exe -k MULADDSUB
	@echo '******* time raja perf suite kernel MULADDSUB with CUPTI PC Sampling ********'
	CUPTI_SAMPLING_PERIOD=1 time cupti-preload/enablesampling rajaperf/cupti-test/bin/raja-perf.exe -k MULADDSUB
	CUPTI_SAMPLING_PERIOD=2 time cupti-preload/enablesampling rajaperf/cupti-test/bin/raja-perf.exe -k MULADDSUB
	CUPTI_SAMPLING_PERIOD=3 time cupti-preload/enablesampling rajaperf/cupti-test/bin/raja-perf.exe -k MULADDSUB
	CUPTI_SAMPLING_PERIOD=4 time cupti-preload/enablesampling rajaperf/cupti-test/bin/raja-perf.exe -k MULADDSUB

distclean:
	cd cupti-preload; make clean
	/bin/rm -rf rajaperf RAJA* cub-* $(CUBTGZ)* 
	
