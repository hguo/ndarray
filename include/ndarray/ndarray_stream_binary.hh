#ifndef _NDARRAY_STREAM_BINARY_HH
#define _NDARRAY_STREAM_BINARY_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML

#include <ndarray/ndarray_stream.hh>

namespace ftk {

/**
 * @brief Binary file substream
 *
 * Reads raw binary data from files with configurable:
 * - Data type
 * - Dimensions
 * - Byte order (endianness)
 * - File offsets
 *
 * Requires dimensions to be specified in YAML configuration.
 */
template <typename StoragePolicy = native_storage>
struct substream_binary : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_binary(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return true; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_binary<StoragePolicy>::initialize(YAML::Node y)
{
  this->total_timesteps = this->filenames.size();
}

template <typename StoragePolicy>
inline void substream_binary<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  const auto f = this->filenames[i]; // assume each file has only one timestep

  for (const auto &var : this->variables) {
    auto p = ndarray_base::new_by_dtype(var.dtype);

#if NDARRAY_HAVE_MPI
    // Configure distribution
    if (var.dist_type == VariableDistType::DISTRIBUTED) {
      std::vector<size_t> gdims;
      for (int d : var.dimensions) gdims.push_back(static_cast<size_t>(d));
      // Binary dimensions in YAML are usually C-order, reshapec handles it but decompose needs Fortran-order
      std::reverse(gdims.begin(), gdims.end());

      if (var.has_custom_decomposition) p->decompose(this->comm, gdims, 0, var.custom_decomp.dims, var.custom_decomp.ghost);
      else p->decompose(this->comm, gdims);
    } else {
      p->set_replicated(this->comm);
    }
#endif

    // Usevara auto-read
    p->read_binary_auto(f);
    g->set(var.name, p);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_BINARY_HH
