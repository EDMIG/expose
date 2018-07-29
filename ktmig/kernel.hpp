#pragma once

namespace KTMIG
{

template<typename _T>
_T pow2(_T x)
{
    return x*x;
}

template<typename _T>
_T pow3(_T x)
{
    return x*x*x;
}

template<typename _T>
void interp1(_T x, int idx, const _T *ya, _T &y)
{
    //assert(x>=0 && x<=1);
    if(x==_T(0))
    {
        y=ya[idx];
    }
    else if(x==_T(1))
    {
        y=ya[idx+1];
    }
    else
    {
        _T x1=1-x;
        y=x1*ya[idx]+x*ya[idx+1];
    }

    return;
}

template<typename _T>
void interp12(_T x, int idx, const _T *ya1,const _T *ya2, _T &y1,_T &y2)
{
    //assert(x>=0 && x<=1);
    if(x==_T(0))
    {
        y1=ya1[idx];
        y2=ya2[idx];
    }
    else if(x==_T(1))
    {
        y1=ya1[idx+1];
        y2=ya2[idx+1];
    }
    else
    {
        _T x1=1-x;
        y1=x1*ya1[idx]+x*ya1[idx+1];
        y2=x1*ya2[idx]+x*ya2[idx+1];
    }

    return;
}


template<typename _T>
void interp123(_T x, int idx, const _T *ya1,const _T *ya2, const _T *ya3, _T &y1,_T &y2,_T&y3)
{
    //assert(x>=0 && x<=1);
    if(x==_T(0))
    {
        y1=ya1[idx];
        y2=ya2[idx];
        y3=ya3[idx];
    }
    else if(x==_T(1))
    {
        y1=ya1[idx+1];
        y2=ya2[idx+1];
        y3=ya3[idx+1];
    }
    else
    {
        _T x1=1-x;
        y1=x1*ya1[idx]+x*ya1[idx+1];
        y2=x1*ya2[idx]+x*ya2[idx+1];
        y3=x1*ya3[idx]+x*ya3[idx+1];
    }

    return;
}

template<typename _T>
_T calc_tt(_T t02,_T x2, _T x4, _T x6, _T c2, _T c4, _T c6)
{
    _T tt;
    tt=(t02+c2*x2+c4*x4);
    if(tt<_T(0.8))
        tt=sqrt(t02+c2*x2);
    else
        tt=(tt+c6*x6)/sqrt(tt);

    return tt;
}

template<typename _T>
_T calc_weight(_T t0, _T vs, _T ts, _T tg)
{
    _T w;
    w=1;
    w=t0*vs*(1/ts+1/tg)*2;

    return w;
}

template<typename _T, int N=8>
void update_image(_T t1, _T w1, _T t2, _T w2, const _T *din, _T *image)
{
    static int MASK=12;
    static _T SCALE=4096.0;

    _T dt,dw;
    dt=(t2-t1)/N;
    dw=(w2-w2)/N;

    int kt,kdt;
    //转化为整数运算，避免浮点数插值
    //效率未必能提高多少！
    kt=(t1+_T(0.5))*SCALE;
    kdt=(dt*SCALE);

    _T w;
    w=w1;

    for(int i=0; i<N; i++)
    {
        int it=kt>>MASK;
        //注意下标，it可能需要减一
        //不需要测试边界条件，调用者负责边界测试
        image[i]+=w*din[it];

        kt+=kdt;
        w+=dw;
    }
}

//反假频处理
template<typename _T, int N=8>
void update_image_kp(_T t1, _T w1, _T t2, _T w2,
                     int kp, _T awt, const _T *din, _T *image)
{

    static int MASK=12;
    static _T SCALE=4096.0;

    //assert(awt=1.0/(kp*kp))

    _T dt, dw,w;
    dt=(t2-t1)/N;
    dw=(w2-w1)/N;
    w=w1;

    w*=awt;
    dw*=awt;

    int kt, kdt;

    kt=(t1+_T(0.5))*SCALE;
    kdt=dt*SCALE:

    for(int i=0; i<N; i++)
    {
        int it=kt>>MASK;
        //没有处理边界越界情况
        image[i]+=w*(2*din[it]-din[it-kp]-din[it+kp]);
        kt+=kdt;
        w+=dw;
    }
}
}


namespace KTMIG {

void migt_core(_T *image， int ntout, int ntrace)
{
    //TMINC对应表数据:apx,apy,vel*等和输出间隔比
    //表数据一个间隔，相当于image输出TMINC个输出点
    //对应进行插值计算的步长
    //为了节省空间和计算
    //TMINC
    static const int TMINC=32;

    //输入道xy坐标
    _T xin,yin;
    xin=(sx+gx)/2;
    yin=(sy+gy)/2;

    //最大孔径
    _T apx_max,apy_max;
    apx_max=apx[ntab-1];
    apy_max=apy[ntab-1];

    for(int itrace=itr_first; itrace<ntrace; itrace+=itr_step)
    {
        //输出道xy坐标
        _T xout, yout;
        xout=cpds[itrace].cpdx;
        yout=cdps[itrace].cdpy;

        //输入和输出距离
        _T x2,y2;
        x2=pow2(xout-xin);
        y2=pow2(yout-yin);

        //最大孔径之外
        if(x2*apx_max+y2*apy_max>0.98) continue;

        //输出道位置速度
        _T vs0;

        //空间假频因子
        _T dp;
        dp=sqrt(pow2((xin-xout)*sub_alias_dx)+pow2((yin-yout)*crs_alias_dy));
        dp*=0.7;
        //dp*=0.4;

        //空间距离信息
        _T s2,s4,s6;
        _T g2,g4,g6;

        s2=pow2(xout-sx)+pow2(yout-sy);
        s4=pow2(s2*_T(1.0e-6));
        s6=pow3(s2*_T(1.0e-6));

        g2=pow2(xout-gx)+pow2(yout-gy);
        g4=pow2(g2*_T(1.0e-6));
        g6=pow3(g2*_T(1.0e-6));

        //搜索第一个孔径内
        int L,L1;
        for(L=LLive/TMINC; L<NTAB; L++)
        {
            if(x2*apx[L]+y2*apy[L]<1) break;
        }

        L1=L+1;


    }
}


}
