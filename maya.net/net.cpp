#include <vector>
#include <cstdio>
#include <iostream>
#include <cassert>

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
  T net(const T *x)
  {
    T hid[2],out,y;
    mapminmax_apply<T,N>(x,in,input);
    layer1(hid);
    layer2(hid,out);
    mapminmax_reverse<T,N>(&out,&y,output);
    return y;
  }
private:
  void layer1(T hid[2])
  {
    T y1,y2;
    y1=b1[0];
    y2=b1[1];
    size_t n=N;
    for(size_t i=0; i<n; i++)
    {
      y1+=in[i]*IW[2*i];
      y2+=in[i]*IW[2*i+1];
    }
    hid[0]=tanh(y1);
    hid[1]=tanh(y2);
  }
  void layer2(const T hid[2], T &out)
  {
    T y=hid[0]*LW[0]+hid[1]*LW[1]+b2[0];
    //purelin
    return y;
  }
};

void test_nnet()
{
  const char *xfile="x.bin";
  const char *net1_file="net1.bin";
  const char *net2_file="net2.bin";
  const char *net3_file="net3.bin";

  const char *input_file="input.bin";

  double x[NNET_INPUT_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet1[NNET_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet2[NNET_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet3[NNET_NUM];

  FILE *fp=fopen(xfile,"rb");
  assert(fp);
  fread(x,sizeof(x[0]),NNET_INPUT_NUM,fp);
  fclose(fp);

  FILE *fp1=fopen(net1_file,"rb");
  assert(fp1);

  FILE *fp2=fopen(net2_file,"rb");
  assert(fp2);

  FILE *fp3=fopen(net3_file,"rb");
  assert(fp3);

  for(int i=0; i<31; i++)
  {
    nnet1[i].Read(fp1);
    nnet2[i].Read(fp2);
    nnet3[i].Read(fp3);

    double out1=nnet1[i].net(x);
    double out2=nnet2[i].net(x);
    double out3=nnet3[i].net(x);

    cout<<"i="<<i<<", "<<out1<<", "<<out2<<", "<<out3<<endl;

  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);

  return ;
}

#include "HeapMedian.h"

double inputMatrix[528][NNET_INPUT_NUM];

void test_nnet_1()
{
  const char *net1_file="net1.bin";
  const char *net2_file="net2.bin";
  const char *net3_file="net3.bin";

  const char *input_file="input.bin";

  double x[NNET_INPUT_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet1[NNET_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet2[NNET_NUM];
  NNet2<double,NNET_INPUT_NUM> nnet3[NNET_NUM];

  FILE *fp=fopen(input_file,"rb");
  assert(fp);
  fread(&inputMatrix[0][0],sizeof(inputMatrix[0][0]),528*NNET_INPUT_NUM,fp);
  fclose(fp);

  FILE *fp1=fopen(net1_file,"rb");
  assert(fp1);

  FILE *fp2=fopen(net2_file,"rb");
  assert(fp2);

  FILE *fp3=fopen(net3_file,"rb");
  assert(fp3);

  for(int i=0; i<31; i++)
  {
    nnet1[i].Read(fp1);
    nnet2[i].Read(fp2);
    nnet3[i].Read(fp3);
  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);

  double maxparam=2;
  double alpha=0.5;

  int k=15;

  Mediator<double,15> mediators[NNET_NUM];

  for(int i=0; i<NNET_NUM; i++)
  {
    for(int j=0; j<k; j++)
    {
      mediators[i].insert(0);
    }
  }

  double ws[NNET_NUM]={0};

  for(int i=0; i<528; i++)
  {
    double *x=&inputMatrix[i][0];
    for(int j=0; j<NNET_NUM; j++)
    {
      double out1=nnet1[i].net(x);
      double out2=nnet2[i].net(x);
      double out3=nnet3[i].net(x);

      double w=(out1+out2+out3)/3;

      mediators[j].insert(w);
      w=mediators[j].getMedian();

      double dc1=abs(w-ws[j])>maxparam;

      w=alpha*(dc1*(w-ws[j])+ws[j])+(1-alpha)*ws[j];

      if(j==25) w-=15;
      ws[j]=w;

    }
  }
  return ;
}

#include "tri.h"
void test_triEd()
{
  double feature[178];
  //distance计算，只用到了X坐标
  //dDistance是脸宽
  double dDistance=abs(PointsXY[0][0]-PointsXY[16][0]);
  for(int i=0; i<178; i++)
  {
    int index1=triEd[i][0]-1;
    int index2=triEd[i][1]-1;
    double dist=abs(PointsXY[index1][0]-PointsXY[index2][0]);
    feature[i]=dist/dDistance;
    printf("f[%d]=%g\n",i+1,feature[i]);
  }
}

double test_median()
{
  static double data[]={

  };
  int k=15;
  int n=100;
  Mediator<double,15> mediator;
  int i;
  for(i=0; i<k; i++)
  {
    mediator.insert(data[i]);
  }

  double y;
  y=mediator.getMedian();
  printf("%g\n",y);
  for(;i<n; i++)
  {
    mediator.insert(data[i]);
    y=mediator.getMedian();
    printf("%g\n",y);
  }

  return y;
}

static const string BSName[]={
    "Blink_Left",//1
    "Blink_Right",//2
};

static const string BShape="Facial_Blends_nc11_2";

#include <sstream>

string createMayaCmd(double ws[50],int Time)
{
  ostringstream cmd;
  cmd<<"cmds.select(‘Facial_Blends_nc11_2’,r=True);\n";
  cmd<<"cmds.currentTime("<<Time<<");\n";

  string root=R"(cmds.setAttr(")"+BShape+"."；

  for(int i=0; i<50; i++)
  {
    cmd<<root<<BSName[i]<<"\,"<<ws[i]<<");\n";
  }

  cmd<<"cmds.setKeyframe();\n";

  return cmd.str();
}

#include<WinSock2.h>
#pragma comment(lib,"ws2_32.lib")

void test_maya_commandports()
{
  static int port=7002;
  WORD sockVersion=MAKEWORD(2,2);
  WSADATA wsaData;
  if(WSAStartup(sockVersion,&wsaData)!=0)
  {
    assert(0);
  }

  SOCKET sclient=socket(AF_INET,SOCK_STREAM, IPPROTO_TCP);

  assert(sclient!=INVALID_SOCKET);

  sockaddr_in serAddr;
  serAddr.sin_family=AF_INET;
  serAddr.sin_port=htons(port);
  serAddr.sin_addr.S_un.S_addr=inet_addr("127.0.0.1");
  if(connect(sclient,(sockaddr*)&serAddr, sizeof(serAddr))==SOCKET_ERROR)
  {
    assert(0);
  }
  double ws[50]={0};
  double dw=0.2;

  static const char *ws_file="e:/1109/liu.bin";

  FILE *fp=fopen(ws_file,"rb");assert(fp);

  int Time=0;
  while(1)
  {
    size_t nread=fread(ws,sizeof(ws[0]),50,fp);
    if(nread!=50) break;
    for(int i=0; i<50; i++)
    {
      ws[i]/=100;
      if(ws[i]<0) ws[i]=0;
      if(ws[i]>1) ws[i]=1;
    }

    string data=createMayaCmd(ws,Time); Time++;

    const char *sendData=data.c_str();

    send(sclient,sendData,strlen(sendData),0);

    Sleep(40);
  }

  fclose(fp);
  closesocket(sclient);

}

struct FeaAux
{
  int n;
  double vmean;
  double xmean[NNET_INPUT_NUM];
  double var[NNET_INPUT_NUM];
  void addx(double x[NNET_INPUT_NUM])
  {
    double a=double(n)/double(n+1);
    double a1=1-a;
    a=sqrt(a);
    a1=sqrt(a1);
    vmean=0;
    for(int i=0; i<NNET_INPUT_NUM; i++)
    {
      double t=x[i]-xmean[i];
      var[i]=hypot(a*var[i],a1*t);
      vmean+=var[i];
    }
    vmean/=NNET_INPUT_NUM;
    n++;
  }
  FeaAux()
  {
    n=0;
    vmean=0;
    for(int i=0; i<NNET_INPUT_NUM; i++)
    {
      var[i]=0;
    }
  }

  void setXmean(double xmean_[NNET_INPUT_NUM])
  {
    for(int i=0; i<NNET_INPUT_NUM; i++)
    {
      xmean[i]=xmean_[i];
    }
  }

  static void test()
  {
    const char *xs_file="alldist.bin";
    const char *xmean_file="xmean.bin";
    FILE *fp;
    fp=fopen(xmean_file,"rb");
    assert(fp);
    double xmean[NNET_INPUT_NUM];
    fread(xmean,sizeof(xmean[0]),NNET_INPUT_NUM,fp);
    fclose(fp);

    fp=fopen(xs_file,"rb");assert(fp);
    double x[NNET_INPUT_NUM];

    FeaAux aux;
    aux.setXmean(xmean);
    while(1)
    {
      if(fread(x,sizeof(double),NNET_INPUT_NUM,fp)==0)
      {
        break;
      }

      aux.addx(x);
      printf("n=%d,vmean=%g\n",aux.n,aux.vmean);
    }

  }

};

double *fea_compute(const double XY[68][2],double fea[NNET_INPUT_NUM])
{
  double dDistance=abs(XY[0][0]-XY[16][0]);

  for(int i=0; i<178; i++)
  {
    int index1=triEd[i][0]-1;
    int index2=triEd[i][1]-1;
    double dist=abs(XY[index1][0]-XY[index2][0]);
    fea[i]=dist/dDistance;
  }

  return fea;
}

//根据xmean & vmean 归一化 x
double *feature_norm(double vmean, const double fea_mean[NNET_INPUT_NUM],
        double x[NNET_INPUT_NUM])
        {
          const double **mean=fea_mean;
          for(int i=0; i<NNET_INPUT_NUM; i++)
          {
            x[i]=(x[i]-mean[i])/vmean;
          }
        }


void test_tracker_data()
{

  double feature[178];
  double PointsXY[68][2];

  FeaAux aux;

  int frames=0;
  while(1)
  {
    frames++;
    if(frames==5)
    {
      aux.setXmean(feature);break;
    }
  }

  //此处初始化可能有BUG
  double ws[NNET_NUM]={0}；
  double ws50[50]={0};
  static const int maps[]=
  {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    0,0,0,0,0,0,
    23,
    0,0,
    26,27,28,29,30,31,32,33,
    0,0,
    36,37,
    0,0,
    40,41,42,43,
    0,0,0,0,0,0,0,
  }

  fseek(fp,0,0);
  frames=0;
  while(1)
  {
    aux.addx(feature);
    double vmean=aux.vmean;
    double *xmean=aux.xmean;
    double *x=feature;

    double *w=ws;
    for(int i=0; i<50; i++)
    {
        ws50[i]=0;
        if(maps[i]>0)
        {
          assert(maps[i]==i+1);
          ws50[i]=*w++;
          ws50[i]/=100;
          if(ws50[i]<0) ws50[i]=0;
          if(ws50[i]>1) ws50[i]=1;
        }
    }
    assert(w==ws+31);
  }


}

int _tmain(int argc, _TCHAR* argv[])
{
  test_maya_commandports();
}
