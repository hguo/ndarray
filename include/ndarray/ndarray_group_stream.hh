#ifndef _NDARRAY_NDARRAY_GROUP_STREAM_HH
#define _NDARRAY_NDARRAY_GROUP_STREAM_HH

#include <ndarray/ndarray_group.hh>

namespace ndarray {

struct ndarray_group_stream {
  std::shared_ptr<ndarray_group> advance_timestep();
  std::shared_ptr<ndarray_group> get_static_group();

public:
  std::shared_ptr<ndarray_group> static_group;
};

///////////

}

#endif
