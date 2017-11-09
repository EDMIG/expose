
#pragma once

#include "Median.h"

void heap_median(MedianFilter f, const Vector &x, Vector &y);

template<typename Item, int nItems_>
class Mediator
{
public:
  Mediator()
  {
    int nItems=nItems_;
    data=new Item[nItems];
    pos=new int[nItems];
    allocatedHeap=new int[nItems];
    heap=allocatedHeap+(nItems/2);
    N=nItems;
    minCt=maxCt=idx=0;
    while(nItems--)
    {
      pos[nItems]=((nItems+1)/2)*((nItems&1)?-1:1);
      heap[pos[nItems]]=nItems;
    }
  }
  ~Mediator()
  {
    delete[]data;
    delete[]pos;
    delete[]allocatedHeap;
  }

  void insert(Item v)
  {
    int p=pos[idx];
    Item old=data[idx];
    data[idx]=v;
    idx=(idx+1)%N;

    if(p>0)
    {
      if(minCt<(N-1)/2)
      {
        minCt++;
      }
      else if(v>old)
      {
        minSortDown(p);return;
      }

      if(minSortUp(p) && mmCmpExch(0,-1))
      {
        maxSortDown(-1);
      }

    }
    else if(p<0)
    {
      if(maxCt<N/2)
      {
        maxCt++;
      }
      else if(v<old)
      {
        maxSortDown(p); return;
      }

      if(maxSortUp(p) && minCt && mmCmpExch(1,0))
      {
        minSortDown(1);
      }

    }
    else
    {
      if(maxCt && maxSortUp(-1))
      {
        maxSortDown(-1);
      }
      if(minCt && minSortUp(1))
      {
        minSortDown(1);
      }
    }
  }
  Item getMedian()
  {
    Item v=data[heap[0]];
    if(minCt<maxCt)
    {
      v=(v+data[heap[-1]])/2;
    }
    return v;
  }
private:
  int mmexchange(int i, int j)
  {
    int t=head[i];
    heap[i]=heap[j];
    heap[j]=t;
    pos[heap[i]]=i;
    pos[head[j]]=j;
    return 1;
  }

  void minSortDown(int i)
  {
    for(i*=2;i<=minCt;i*=2)
    {
      if(i<minCt &&mmless(i+1,i))
      {
        i++;
      }
      if(!mmCmpExch(i,i/2))
      {
        break;
      }
    }
  }

  void maxSortDown(int i)
  {
    for(i*=2;i>=-maxCt;i*=2)
    {
      if(i>-maxCt && mmless(i,i-1))
      {
        --i;
      }
      if(!mmCmpExch(i/2,i))
      {
        break;
      }
    }
  }

  inline int mmless(int i, int j)
  {
    return data[heap[i]]<data[heap[j]];
  }

  inline int mmCmpExch(int i, int j)
  {
    return mmless(i,j) && mmexchange(i,j);
  }

  inline int minSortUp(int i)
  {
    while(i>0 && mmCmpExch(i,i/2))
    {
      i/=2;
    }
    return i==0;
  }

  inline int maxSortUp(int i)
  {
    while(i<0 &&mmCmpExch(i/2,i))
    {
      i/=2;
    }

    return i==0;
  }

private:
  Item *data;
  int *pos;
  int *heap;
  int *allocatedHeap;
  int N;
  int idx;
  int minCt;
  int maxCt;

};
