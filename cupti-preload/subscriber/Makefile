C?=/usr/local/cuda-10.1
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

subscriber.so: subscriber.o
	$(CC) $(OPT) -shared -o $@ subscriber.$(OBJ) $(LIBS)

subscriber.o: subscriber.cpp
	$(CC) $(OPT) -c $(INCLUDES) $<

clean:
	rm -f subscriber.so subscriber.o

