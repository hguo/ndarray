#ifndef _NDARRAY_STREAM_HDF5_HH
#define _NDARRAY_STREAM_HDF5_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML && NDARRAY_HAVE_HDF5

#include <ndarray/ndarray_stream.hh>

namespace ftk {

/**
 * @brief HDF5 substream for time-varying data
 *
 * Reads variables from HDF5 files with support for:
 * - Variable name aliasing (possible_names)
 * - Optional variables
 * - Multi-file timesteps
 */
struct substream_h5 : public substream {
  substream_h5(stream& s) : substream(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);

  bool has_unlimited_time_dimension = false;
};

///////////
// Implementation
///////////

inline void substream_h5::initialize(YAML::Node y)
{
  if (!is_static)
    this->total_timesteps = this->filenames.size();
}

inline void substream_h5::read(int i, std::shared_ptr<ndarray_group> g)
{
  auto fid = H5Fopen( this->filenames[i].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if (fid >= 0) {
    for (const auto &var : variables) {

      // probe variable name
      hid_t did = H5I_INVALID_HID;
      for (const auto varname : var.possible_names) {
        did = H5Dopen2(fid, varname.c_str(), H5P_DEFAULT);
        if (did != H5I_INVALID_HID)
          break;
      }

      if (did == H5I_INVALID_HID) {
        if (var.is_optional)
          continue;
        else {
          nd::fatal("cannot read variable " + var.name);
          return;
        }
      } else {
        // create a new array
        auto native_type = H5Dget_type(did);
        auto p = ndarray_base::new_by_h5_dtype( native_type );

        // actual read
        p->read_h5_did(did);
        g->set(var.name, p);

        H5Dclose(did);
      }
    }
    H5Fclose(fid);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML && NDARRAY_HAVE_HDF5

#endif // _NDARRAY_STREAM_HDF5_HH
