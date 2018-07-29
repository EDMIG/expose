#pragma once
namespace xmig
{
  struct XHead
  {
    float m_sx,m_sy,m_gx,m_gy;
    //cx=(sx+gx)/2
    //cy=(sy+gy)/2
    float m_cx,m_cy;

    int m_sidx,m_gidx;
    float sx()const{return m_sx;}
    float sy()const{return m_sy;}
    float gx()const{return m_gx;}
    float gy()const{return m_gy;}
    float cx()const{return m_cx;}
    float cy()const{return m_cy;}

    void sx(float t){m_sx=t;}
    void sy(float t){m_sy=t;}
    void gx(float t){m_gx=t;}
    void gy(float t){m_gy=t;}
    void cx(float t){m_cx=t;}
    void cy(float t){m_cy=t;}
  };
}
