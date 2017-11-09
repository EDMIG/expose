#pragma once

#include "Element.h"
#include <cassert>
#include <stdexcept>

struct MedianFilter{
  unsigned k;
  unsigned blocks;
  unsigned n;
  unsigned half;
  unsigned result;

  MedianFilter(unsigned half_, unsigned blocks_):
    k{2*half_+1},
    blocks{blocks_},
    n{k*blocks_},
    half{half_},
    result{k*(blocks_-1)+1}
  {
    if(half==0){
      throw std::invalid_argument("half-window size must be at least 1");
    }
    
  }
};
