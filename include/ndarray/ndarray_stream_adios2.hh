#ifndef _NDARRAY_STREAM_ADIOS2_HH
#define _NDARRAY_STREAM_ADIOS2_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML && NDARRAY_HAVE_ADIOS2

#include <ndarray/ndarray_stream.hh>

namespace ftk {

/**
 * @brief ADIOS2 (BP format) substream for time-varying data
 *
 * Reads variables from ADIOS2 BP files with support for:
 * - Variable name aliasing (possible_names)
 * - Optional variables
 * - Multi-file timesteps
 * - High-performance parallel I/O
 */
template <typename StoragePolicy = native_storage>
struct substream_adios2 : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_adios2(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_adios2<StoragePolicy>::initialize(YAML::Node y)
{
  if (!is_static)
    this->total_timesteps = this->filenames.size();
}

template <typename StoragePolicy>
inline void substream_adios2<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  const auto f = this->filenames[i];

  adios2::Engine reader = this->stream_.io.Open(f, adios2::Mode::Read);
  auto available_variables = this->stream_.io.AvailableVariables(true);

  for (const auto &var : this->variables) {
    std::string actual_varname;
    for (const auto varname : var.possible_names) {
      if (available_variables.find(varname) != available_variables.end()) {
        actual_varname = varname;
        break;
      }
    }

    if (actual_varname.empty()) {
      if (var.is_optional)
        continue;
      else {
        nd::fatal("cannot find variable " + var.name);
        return;
      }
    } else {
      auto p = ndarray_base::new_from_bp(
          this->stream_.io,
          reader,
          actual_varname);
      g->set(var.name, p);
    }
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML && NDARRAY_HAVE_ADIOS2

#endif // _NDARRAY_STREAM_ADIOS2_HH
