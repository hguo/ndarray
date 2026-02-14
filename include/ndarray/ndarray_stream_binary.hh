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
struct substream_binary : public substream {
  substream_binary(stream& s) : substream(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return true; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);
};

///////////
// Implementation
///////////

inline void substream_binary::initialize(YAML::Node y)
{
  this->total_timesteps = filenames.size();
}

inline void substream_binary::read(int i, std::shared_ptr<ndarray_group> g)
{
  const auto f = filenames[i]; // assume each file has only one timestep
  FILE *fp = fopen(f.c_str(), "rb");

  for (const auto &var : variables) {
    auto p = ndarray_base::new_by_dtype( var.dtype );
    p->reshapec( var.dimensions );

    if (!var.is_offset_auto)
      fseek(fp, var.offset, SEEK_SET);

    p->read_binary_file( fp, var.endian );
    g->set(var.name, p);
  }

  fclose(fp);
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_BINARY_HH
