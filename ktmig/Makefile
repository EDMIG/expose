CXX = g++

BOOST_ROOT = /cygdrive/e/Temp/boost_1_67_0.cygwin
SPDLOG_ROOT = /cygdrive/e/Temp/spdlog-1.x/include
JSON_ROOT = /cygdrive/e/Temp/json-develop/single_include
INCLUDES= -I$(BOOST_ROOT)  -I$(SPDLOG_ROOT) -I$(JSON_ROOT)

CPPFLAGS = $(INCLUDES)

CXXFLAGS	=  -D_WIN32_WINNT=0x600 -march=native -Wall -Wextra -pedantic -std=gnu++0x -pthread -Wl,--no-as-needed
CXX_RELEASE_FLAGS = -O3
CXX_DEBUG_FLAGS= -g

LDFLAGS = -static -L$(BOOST_ROOT)/lib
LDFLAGS += -lboost_filesystem -lboost_system

all:	xmig

xmig: xmig.cpp xmig/constant.cpp
	$(CXX) xmig.cpp xmig/constant.cpp  -o xmig $(CPPFLAGS) $(CXXFLAGS) $(CXX_RELEASE_FLAGS) $(LDFLAGS)

clean:
	rm -f *.o xmig.exe
