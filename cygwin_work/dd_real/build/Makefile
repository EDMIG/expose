

EIGEN_ROOT = ~/e_Temp/Eigen_3.3.4/eigen-eigen-5a0156e40feb
SPECTRA_ROOT= ~/e_Temp/spectra-master

RM := rm -rf
CXX = g++

#CXXFLAGS = -O3 -std=c++11 -fopenmp

#-mavx2 -mfma ѡ���Ǳ����ģ���Ȼfma����û���ã�
#����fmaָ�
CXXFLAGS += -O3 -mavx2 -mfma -fopenmp -std=c++14

LDFLAGS = -fopenmp
LDLIBS =

CPPFLAGS = -I $(EIGEN_ROOT) -I $(SPECTRA_ROOT)/include -I ../qd

#sources := $(wildcard *.cpp)
sources := bits.cpp dd_real.cpp qd_real.cpp qd_const.cpp dd_const.cpp util.cpp
objects = $(subst .cpp,.o,$(sources))

vpath %.cpp ..
vpath %.cpp ../qd
vpath %.h ..
vpath %.h ../qd

.PHONY: all
all:	test_qd

test_qd: test_qd.o
test_qd: objects += test_qd.o
test_qd:$(objects)
	$(CXX)  $(LDFLAGS) $(objects)  -o $@

test_spectra:CXX=mpicxx
test_spectra:test_spectra.o
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

test_Eigen:test_Eigen.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

test_mpi:CXX=mpicxx
test_mpi:test_mpi.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

.PHONY: clean
clean:
	$(RM) *.o
