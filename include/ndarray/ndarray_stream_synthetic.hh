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
struct substream_synthetic : public substream {
  substream_synthetic(stream& s) : substream(s) {}
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
struct substream_synthetic_woven : public substream_synthetic {
  substream_synthetic_woven(stream& s) : substream_synthetic(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);

  double scaling_factor = 15.0;
  double t0 = 1e-4, delta = 0.1;
};

///////////
// Implementation
///////////

inline void substream_synthetic::initialize(YAML::Node y)
{
  if (auto ytimesteps = y["timesteps"])
    this->total_timesteps = ytimesteps.as<int>();
  else
    this->total_timesteps = 10;
}

inline void substream_synthetic_woven::initialize(YAML::Node y)
{
  substream_synthetic::initialize(y);

  if (auto yscaling_factor = y["scaling_factor"])
    this->scaling_factor = yscaling_factor.as<double>();

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  if (auto yt0 = y["t0"])
    this->t0 = yt0.as<double>();

  assert(has_dimensions());
}

inline void substream_synthetic_woven::read(int i, std::shared_ptr<ndarray_group> g)
{
  for (const auto &var : variables) { // normally, there should be only one variable
    const auto arr = synthetic_woven_2D<double>(
        this->dimensions[0],
        this->dimensions[1],
        t0 + delta * i,
        scaling_factor);
    g->set(var.name, arr);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_SYNTHETIC_HH
