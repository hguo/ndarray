#ifndef _NDARRAY_STREAM_SYNTHETIC_HH
#define _NDARRAY_STREAM_SYNTHETIC_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML

#include <ndarray/ndarray_stream.hh>
#include <ndarray/synthetic.hh>

namespace ftk {

/**
 * @brief Base class for synthetic data substreams
 *
 * Generates synthetic test data instead of reading from files.
 * Useful for testing and benchmarking without real data files.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return false; }
  bool require_dimensions() { return true; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
};

/**
 * @brief Synthetic woven pattern data generator
 *
 * Generates 2D woven pattern test data with configurable:
 * - Scaling factor
 * - Time parameters (t0, delta)
 * - Dimensions
 *
 * Used for testing time-varying data workflows.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_woven : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_woven(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  double scaling_factor = 15.0;
  double t0 = 1e-4, delta = 0.1;
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_synthetic<StoragePolicy>::initialize(YAML::Node y)
{
  if (auto ytimesteps = y["timesteps"])
    this->total_timesteps = ytimesteps.as<int>();
  else
    this->total_timesteps = 10;
}

template <typename StoragePolicy>
inline void substream_synthetic_woven<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  if (auto yscaling_factor = y["scaling_factor"])
    this->scaling_factor = yscaling_factor.as<double>();

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  if (auto yt0 = y["t0"])
    this->t0 = yt0.as<double>();

  assert(this->has_dimensions());
}

template <typename StoragePolicy>
inline void substream_synthetic_woven<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) { // normally, there should be only one variable
    // Respect dtype from YAML configuration
    if (var.dtype == NDARRAY_DTYPE_FLOAT) {
      const auto arr = synthetic_woven_2D<float>(
          this->dimensions[0],
          this->dimensions[1],
          static_cast<float>(t0 + delta * i),
          static_cast<float>(scaling_factor));
      g->set(var.name, arr);
    } else if (var.dtype == NDARRAY_DTYPE_DOUBLE || var.is_dtype_auto) {
      // Default to double if dtype is auto or explicitly double
      const auto arr = synthetic_woven_2D<double>(
          this->dimensions[0],
          this->dimensions[1],
          t0 + delta * i,
          scaling_factor);
      g->set(var.name, arr);
    } else if (var.dtype == NDARRAY_DTYPE_INT) {
      const auto arr = synthetic_woven_2D<int>(
          this->dimensions[0],
          this->dimensions[1],
          static_cast<int>(t0 + delta * i),
          static_cast<int>(scaling_factor));
      g->set(var.name, arr);
    } else {
      // Fallback: use double for unsupported dtypes
      const auto arr = synthetic_woven_2D<double>(
          this->dimensions[0],
          this->dimensions[1],
          t0 + delta * i,
          scaling_factor);
      g->set(var.name, arr);
    }
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_SYNTHETIC_HH
