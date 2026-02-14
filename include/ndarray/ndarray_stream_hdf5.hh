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
 * - Multiple timesteps per file (using h5_name patterns like "data_t%d")
 */
struct substream_h5 : public substream {
  substream_h5(stream& s) : substream(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);

  bool has_unlimited_time_dimension = false;
  int timesteps_per_file = 1;  // Number of timesteps (datasets) per file
};

///////////
// Implementation
///////////

inline void substream_h5::initialize(YAML::Node y)
{
  // Read timesteps_per_file if specified
  if (y["timesteps_per_file"]) {
    timesteps_per_file = y["timesteps_per_file"].as<int>();
  }

  if (!is_static) {
    // Total timesteps = number of files × timesteps per file
    this->total_timesteps = this->filenames.size() * timesteps_per_file;
  }
}

inline void substream_h5::read(int i, std::shared_ptr<ndarray_group> g)
{
  // Calculate which file and which timestep within that file
  int file_index = i / timesteps_per_file;
  int local_timestep = i % timesteps_per_file;

  auto fid = H5Fopen( this->filenames[file_index].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if (fid >= 0) {
    for (const auto &var : variables) {

      // probe variable name, with support for format strings like "data_t%d"
      hid_t did = H5I_INVALID_HID;
      for (auto varname : var.possible_names) {
        // Check if varname contains a format specifier
        if (varname.find("%d") != std::string::npos) {
          // Format the variable name with local timestep index
          char formatted_name[256];
          snprintf(formatted_name, sizeof(formatted_name), varname.c_str(), local_timestep);
          varname = formatted_name;
        }

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
