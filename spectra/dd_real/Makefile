

EIGEN_ROOT = /cygdrive/e/Temp/Eigen_3.3.4/eigen-eigen-5a0156e40feb
SPECTRA_ROOT= /cygdrive/e/Temp/spectra-master

RM := rm -rf
CXX = g++

#CXXFLAGS = -O3 -std=c++11 -fopenmp 

#-mavx2 -mfma 选项是必须的，不然fma函数没作用！
#加上fma指令！
CXXFLAGS += -O3 -mavx2 -mfma -fopenmp -std=c++11

LDFLAGS = -fopenmp
LDLIBS = 

CPPFLAGS = -I$(EIGEN_ROOT) -I$(SPECTRA_ROOT)/include

sources := $(wildcard *.cpp)
objects = $(subst .cpp,.o,$(sources))


.PHONY: all
all:	$(objects)
	$(CXX)  $(LDFLAGS) $(objects)


.PHONY: clean
clean:
	$(RM) $(objects)
