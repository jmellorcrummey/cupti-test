#
# Copyright 2014-2015 NVIDIA Corporation. All rights reserved
#
C=/usr/local/cuda-9.2
C=/usr/local/cuda-10.0
P=$(C)/extras/CUPTI
CUPTI=`pwd`/../cupti-linux-cuda10-1.0.6-24971014
P=$(CUPTI)

CL=$(C)/lib64
PL=$(P)/lib64
PL=$(P)/lib/x64
INCLUDES=-I$(C)/include -I$(P)/include
CLIBS= -L$(CL) -Wl,-rpath=$(CL) -L$(PL) -Wl,-rpath=$(PL)

CC=g++

OPT=-fPIC -g -O3

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

