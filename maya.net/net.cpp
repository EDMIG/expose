#include <vector>
#include <cstdio>
#include <iostream>

using namespace std;

const static int NNET_INPUT_NUM=178;
const static int NNET_NUM=31;

template<typename T, int N>
struct ProcessSetting
{
  T xmax[N];
  T xmin[N];
  T ymax;
  T ymin;
};

template<typename T, int N>
void mapminmax_apply(const T x[N], T y[N], const ProcessSetting<T,N> &para)
{
  size_t n=N;
  T xmax,xmin,ymin,ymax;
  ymax=para.ymax;
  ymin=para.ymin;
  for(size_t i=0; i<n; i++)
  {
    xmax=para.xmax[i];
    xmin=para.xmin[i];
    y[i]=(ymax-ymin)*(x[i]-xmin)/(xmax-xmin)+ymin;
  }
  return;
}

template<typename T, int N>
void mapminmax_reverse(const T y[N], T x[N], const ProcessSetting<T,N> &para)
{
  size_t n=N;
  T xmax,xmin,ymin,ymax;
  ymax=para.ymax;
  ymin=para.ymin;

  for(size_t i=0; i<n; i++)
  {
    xmax=para.xmax[i];
    xmin=para.xmin[i];
    x[i]=(y[i]-ymin)/(ymax-ymin)*(xmax-xmin)+xmin;
  }
  return;
}

template<typename T, int N=NNET_INPUT_NUM>
class NNet2
{
private:
  ProcessSetting<T,N> input;
  ProcessSetting<T,N> output;

  T IW[2*N];
  T LW[2];


  T b1[2];
  T b2[1];

public:
  T in[N];
public:
  void Read(FILE *fp)
  {
    fread(IW,sizeof(T),2*N,fp);
    fread(LW,sizeof(T),2,fp);
    fread(b1,sizeof(T),2,fp);
    fread(b2,sizeof(T),1,fp);

    fread(input.xmax,sizeof(T),N,fp);
    fread(input.xmin,sizeof(T),N,fp);
    fread(&input.ymax,sizeof(T),1,fp);
    fread(&input.ymin,sizeof(T),1,fp);

    fread(output.xmax,sizeof(T),1,fp);
    fread(output.xmin,sizeof(T),1,fp);
    fread(&output.ymax,sizeof(T),1,fp);
    fread(&output.ymin,sizeof(T),1,fp);

    return;
  }
public:
  
};
