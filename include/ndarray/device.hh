#ifndef __NDARRAY_DEVICE_HH
#define __NDARRAY_DEVICE_HH

#include <ndarray/config.hh>

namespace ftk {

enum {
  NDARRAY_DEVICE_HOST,
  NDARRAY_DEVICE_CUDA,
  NDARRAY_DEVICE_HIP
};

}

#endif
