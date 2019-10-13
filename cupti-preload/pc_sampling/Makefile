#
# Copyright 2014-2015 NVIDIA Corporation. All rights reserved
#
C?=/usr/local/cuda-9.2
P?=$(C)/extras/CUPTI

CL=$(C)/lib64
PL=$(P)/lib64
INCLUDES=-I$(C)/include -I$(P)/include
CLIBS= -L$(CL) -Wl,-rpath=$(CL) -L$(PL) -Wl,-rpath=$(PL)

CC=g++

OPT=-fPIC -g -O3 -std=c++11

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        LIBS= -Xlinker -framework -Xlinker cuda $(CLIBS)  -lcupti
    else
        LIBS= -lcuda $(CLIBS) -lcudart -lcupti
    endif
    OBJ = o
endif

pc_sampling.so: pc_sampling.o
	$(CC) $(OPT) -shared -o $@ pc_sampling.$(OBJ) $(LIBS)

pc_sampling.o: pc_sampling.cpp
	$(CC) $(OPT) -c $(INCLUDES) $<

clean:
	rm -f pc_sampling.so pc_sampling.o

