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
template <typename StoragePolicy = native_storage>
struct substream_h5 : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_h5(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  bool has_unlimited_time_dimension = false;
  int timesteps_per_file = 1;  // Number of timesteps (datasets) per file
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_h5<StoragePolicy>::initialize(YAML::Node y)
{
  // Read timesteps_per_file if specified
  if (y["timesteps_per_file"]) {
    this->timesteps_per_file = y["timesteps_per_file"].as<int>();
  }

  if (!this->is_static) {
    // Total timesteps = number of files × timesteps per file
    this->total_timesteps = this->filenames.size() * this->timesteps_per_file;
  }
}

template <typename StoragePolicy>
inline void substream_h5<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  int file_index = i / this->timesteps_per_file;
  int local_timestep = i % this->timesteps_per_file;
  const std::string filename = this->filenames[file_index];

  auto fid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid < 0) return;

  for (const auto &var : this->variables) {
    hid_t did = H5I_INVALID_HID;
    std::string actual_varname;

    for (auto varname : var.possible_names) {
      if (varname.find("%d") != std::string::npos) {
        char formatted_name[256];
        snprintf(formatted_name, sizeof(formatted_name), varname.c_str(), local_timestep);
        varname = formatted_name;
      }
      did = H5Dopen2(fid, varname.c_str(), H5P_DEFAULT);
      if (did != H5I_INVALID_HID) {
        actual_varname = varname;
        break;
      }
    }

    if (did == H5I_INVALID_HID) {
      if (!var.is_optional) fatal("cannot read variable " + var.name);
      continue;
    }

    auto native_type = H5Dget_type(did);
    auto p = ndarray_base::new_by_h5_dtype(native_type);

#if NDARRAY_HAVE_MPI
    if (var.dist_type == VariableDistType::DISTRIBUTED) {
      hid_t space_id = H5Dget_space(did);
      int nd = H5Sget_simple_extent_ndims(space_id);
      std::vector<hsize_t> hdims(nd);
      H5Sget_simple_extent_dims(space_id, hdims.data(), NULL);
      H5Sclose(space_id);

      std::vector<size_t> gdims(nd);
      for (int d = 0; d < nd; d++) gdims[nd - 1 - d] = static_cast<size_t>(hdims[d]);

      if (var.has_custom_decomposition) p->decompose(this->comm, gdims, 0, var.custom_decomp.dims, var.custom_decomp.ghost);
      else p->decompose(this->comm, gdims);
    } else {
      p->set_replicated(this->comm);
    }
#endif

    H5Dclose(did);
    // Use auto-read (it will re-open file internally or we could optimize by passing fid)
    p->read_hdf5_auto(filename, actual_varname);

    if (var.multicomponents) {
      p->set_multicomponents(1); // Default to 1 for backward compatibility
    }

    g->set(var.name, p);
  }
  H5Fclose(fid);
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML && NDARRAY_HAVE_HDF5

#endif // _NDARRAY_STREAM_HDF5_HH
