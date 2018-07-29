#pragma once
#include <unistd.h> //size_t off_t
#include <cstdint>


//常量
namespace xmig
{
  static const size_t KB=1024UL;
  static const size_t MB=1024UL*1024UL;
  static const size_t GB=1024UL*1024UL*1024UL;
  static const int RESAMP=4;
  static const int NPADDING=256;
  static const int NTRACE_PER_MPI_SEND=10000;
  static const int MAX_NTRACE_PER_LOOP=700000;
}

//总体控制参数
namespace XMIG
{
  extern int NTRACE_PER_LOOP;
  extern int NTRACE_PER_MIGT;
  extern int NUM_BUFT;
  extern float INV_KPKP[128];
}
