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

/**
 * @brief Synthetic moving extremum data generator
 *
 * Generates N-D scalar field with a moving maximum/minimum.
 * The field is a quadratic distance function from a moving point.
 * Configurable:
 * - Initial position (x0)
 * - Direction vector (dir)
 * - Movement rate (implicit in timestep delta)
 * - Sign (1 for maximum, -1 for minimum)
 *
 * Used for testing critical point tracking.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_moving_extremum : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_moving_extremum(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  std::vector<double> x0;  // Initial position
  std::vector<double> dir; // Direction vector
  double delta = 0.1;      // Time step
  int sign = -1;           // -1 for maximum (default), 1 for minimum
};

/**
 * @brief Synthetic double gyre flow generator
 *
 * Generates 2D time-varying vector field of the double gyre flow.
 * Configurable:
 * - Flow parameters (A, omega, epsilon)
 * - Time parameters (t0, delta)
 * - Dimensions
 *
 * Used for testing vector field and flow analysis.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_double_gyre : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_double_gyre(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  double A = 0.1;
  double omega = M_PI * 0.2;
  double epsilon = 0.25;
  double t0 = 0.0, delta = 0.1;
};

/**
 * @brief Synthetic merger pattern generator
 *
 * Generates 2D time-varying scalar field showing two merging blobs.
 * Configurable:
 * - Time parameters (t0, delta)
 * - Dimensions
 *
 * Used for testing blob tracking and topological changes.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_merger : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_merger(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  double t0 = 0.0, delta = 0.1;
};

/**
 * @brief Synthetic moving ramp generator
 *
 * Generates N-D scalar field with a moving ramp (linear function).
 * Configurable:
 * - Initial position (x0)
 * - Movement rate
 * - Dimensions
 *
 * Used for testing ridge/valley tracking.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_moving_ramp : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_moving_ramp(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  double x0 = 0.0;
  double rate = 1.0;
  double delta = 0.1;
};

/**
 * @brief Synthetic tornado flow generator
 *
 * Generates 3D vector field of a tornado flow pattern.
 * Configurable:
 * - Dimensions (must be 3D)
 * - Time start
 *
 * Used for testing 3D flow visualization and analysis.
 */
template <typename StoragePolicy = native_storage>
struct substream_synthetic_tornado : public substream_synthetic<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;

  substream_synthetic_tornado(stream_type& s) : substream_synthetic<StoragePolicy>(s) {}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  int time_start = 0;
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
    // Get dimensions
    const std::vector<int>& dims = var.dimensions.empty() ? this->dimensions : var.dimensions;

#if NDARRAY_HAVE_MPI
    // Configure distribution before generating data
    ndarray_base* p_base = nullptr;
    std::shared_ptr<ndarray<double, StoragePolicy>> p_double;
    std::shared_ptr<ndarray<float, StoragePolicy>> p_float;
    std::shared_ptr<ndarray<int, StoragePolicy>> p_int;

    // Create array of appropriate type
    if (var.dtype == NDARRAY_DTYPE_FLOAT) {
      p_float = std::make_shared<ndarray<float, StoragePolicy>>();
      p_base = p_float.get();
    } else if (var.dtype == NDARRAY_DTYPE_INT) {
      p_int = std::make_shared<ndarray<int, StoragePolicy>>();
      p_base = p_int.get();
    } else {
      p_double = std::make_shared<ndarray<double, StoragePolicy>>();
      p_base = p_double.get();
    }

    // Configure distribution
    std::vector<size_t> yaml_dims_c_order;
    for (int d : dims) yaml_dims_c_order.push_back(static_cast<size_t>(d));

    if (var.dist_type == VariableDistType::DISTRIBUTED) {
      if (var.has_custom_decomposition) {
        p_base->decompose(this->comm, yaml_dims_c_order, 0, var.custom_decomp.pattern, var.custom_decomp.ghost);
      } else {
        p_base->decompose(this->comm, yaml_dims_c_order);
      }

      // Generate distributed data: compute global-to-local mapping
      // For distributed mode, generate only the local portion
      if (p_float) {
        auto& arr = *p_float;
        for (size_t j = 0; j < arr.shape(0); j++) {
          for (size_t i = 0; i < arr.shape(1); i++) {
            // Convert local indices to global indices
            auto global_idx = arr.local_to_global({j, i});
            arr(j, i) = std::sin(static_cast<float>(global_idx[0] + global_idx[1] * scaling_factor + t0 + delta * i));
          }
        }
      } else if (p_int) {
        auto& arr = *p_int;
        for (size_t j = 0; j < arr.shape(0); j++) {
          for (size_t i = 0; i < arr.shape(1); i++) {
            auto global_idx = arr.local_to_global({j, i});
            arr(j, i) = static_cast<int>(global_idx[0] + global_idx[1] * scaling_factor + t0 + delta * i);
          }
        }
      } else {
        auto& arr = *p_double;
        for (size_t j = 0; j < arr.shape(0); j++) {
          for (size_t i = 0; i < arr.shape(1); i++) {
            auto global_idx = arr.local_to_global({j, i});
            arr(j, i) = std::sin(global_idx[0] + global_idx[1] * scaling_factor + t0 + delta * i);
          }
        }
      }
    } else {
      // Replicated mode: generate full data on all ranks
      p_base->set_replicated(this->comm);
      p_base->reshapef(yaml_dims_c_order);

      if (p_float) {
        *p_float = synthetic_woven_2D<float, StoragePolicy>(
            dims[0], dims[1],
            static_cast<float>(t0 + delta * i),
            static_cast<float>(scaling_factor));
      } else if (p_int) {
        *p_int = synthetic_woven_2D<int, StoragePolicy>(
            dims[0], dims[1],
            static_cast<int>(t0 + delta * i),
            static_cast<int>(scaling_factor));
      } else {
        *p_double = synthetic_woven_2D<double, StoragePolicy>(
            dims[0], dims[1],
            t0 + delta * i,
            scaling_factor);
      }
    }

    // Store in group (use shared_ptr directly to preserve distribution metadata)
    if (p_float) g->set(var.name, p_float);
    else if (p_int) g->set(var.name, p_int);
    else g->set(var.name, p_double);

#else
    // Non-MPI mode: generate full data
    if (var.dtype == NDARRAY_DTYPE_FLOAT) {
      const auto arr = synthetic_woven_2D<float, StoragePolicy>(
          dims[0], dims[1],
          static_cast<float>(t0 + delta * i),
          static_cast<float>(scaling_factor));
      g->set(var.name, arr);
    } else if (var.dtype == NDARRAY_DTYPE_DOUBLE || var.is_dtype_auto) {
      const auto arr = synthetic_woven_2D<double, StoragePolicy>(
          dims[0], dims[1],
          t0 + delta * i,
          scaling_factor);
      g->set(var.name, arr);
    } else if (var.dtype == NDARRAY_DTYPE_INT) {
      const auto arr = synthetic_woven_2D<int, StoragePolicy>(
          dims[0], dims[1],
          static_cast<int>(t0 + delta * i),
          static_cast<int>(scaling_factor));
      g->set(var.name, arr);
    } else {
      const auto arr = synthetic_woven_2D<double, StoragePolicy>(
          dims[0], dims[1],
          t0 + delta * i,
          scaling_factor);
      g->set(var.name, arr);
    }
#endif
  }
}

template <typename StoragePolicy>
inline void substream_synthetic_moving_extremum<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  // Get dimensions
  const int ndims = this->dimensions.size();

  // Initialize x0 and dir with defaults
  x0.resize(ndims, 0.5);  // Default: center of domain
  dir.resize(ndims, 0.0);
  if (ndims > 0) dir[0] = 1.0;  // Default: move in x direction

  if (auto yx0 = y["x0"]) {
    if (yx0.IsSequence()) {
      for (size_t i = 0; i < std::min(yx0.size(), x0.size()); i++)
        x0[i] = yx0[i].as<double>();
    }
  }

  if (auto ydir = y["dir"]) {
    if (ydir.IsSequence()) {
      for (size_t i = 0; i < std::min(ydir.size(), dir.size()); i++)
        dir[i] = ydir[i].as<double>();
    }
  }

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  if (auto ysign = y["sign"])
    this->sign = ysign.as<int>();

  assert(this->has_dimensions());
}

template <typename StoragePolicy>
inline void substream_synthetic_moving_extremum<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) {
    const std::vector<int>& dims = var.dimensions.empty() ? this->dimensions : var.dimensions;
    const int ndims = dims.size();

    std::vector<size_t> yaml_dims_c_order;
    for (int d : dims) yaml_dims_c_order.push_back(static_cast<size_t>(d));

#if NDARRAY_HAVE_MPI
    ndarray<double, StoragePolicy> arr;

    // Configure distribution
    if (var.dist_type == VariableDistType::DISTRIBUTED) {
      if (var.has_custom_decomposition) {
        arr.decompose(this->comm, yaml_dims_c_order, 0, var.custom_decomp.pattern, var.custom_decomp.ghost);
      } else {
        arr.decompose(this->comm, yaml_dims_c_order);
      }

      // Generate distributed data: compute quadratic distance from moving point
      if (ndims == 2) {
        double center_x = x0[0] + dir[0] * delta * i;
        double center_y = x0[1] + dir[1] * delta * i;
        for (size_t j = 0; j < arr.shape(0); j++) {
          for (size_t i_idx = 0; i_idx < arr.shape(1); i_idx++) {
            auto global_idx = arr.local_to_global({j, i_idx});
            double x = static_cast<double>(global_idx[1]) / yaml_dims_c_order[1];
            double y = static_cast<double>(global_idx[0]) / yaml_dims_c_order[0];
            double dist2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
            arr(j, i_idx) = sign * dist2;
          }
        }
      } else if (ndims == 3) {
        double center_x = x0[0] + dir[0] * delta * i;
        double center_y = x0[1] + dir[1] * delta * i;
        double center_z = x0[2] + dir[2] * delta * i;
        for (size_t k = 0; k < arr.shape(0); k++) {
          for (size_t j = 0; j < arr.shape(1); j++) {
            for (size_t i_idx = 0; i_idx < arr.shape(2); i_idx++) {
              auto global_idx = arr.local_to_global({k, j, i_idx});
              double x = static_cast<double>(global_idx[2]) / yaml_dims_c_order[2];
              double y = static_cast<double>(global_idx[1]) / yaml_dims_c_order[1];
              double z = static_cast<double>(global_idx[0]) / yaml_dims_c_order[0];
              double dist2 = (x - center_x) * (x - center_x) +
                            (y - center_y) * (y - center_y) +
                            (z - center_z) * (z - center_z);
              arr(k, j, i_idx) = sign * dist2;
            }
          }
        }
      } else {
        fatal("moving_extremum only supports 2D and 3D");
      }
    } else {
      // Replicated mode: generate full data on all ranks
      arr.set_replicated(this->comm);
      arr.reshapef(yaml_dims_c_order);

      if (ndims == 2) {
        double x0_arr[2] = {x0[0], x0[1]};
        double dir_arr[2] = {dir[0], dir[1]};
        auto arr_native = synthetic_moving_extremum<double, 2>(
            yaml_dims_c_order, x0_arr, dir_arr, delta * i);
        if (sign == -1) {
          for (size_t j = 0; j < arr_native.nelem(); j++) arr_native[j] = -arr_native[j];
        }
        arr.from_array(arr_native);
      } else if (ndims == 3) {
        double x0_arr[3] = {x0[0], x0[1], x0[2]};
        double dir_arr[3] = {dir[0], dir[1], dir[2]};
        auto arr_native = synthetic_moving_extremum<double, 3>(
            yaml_dims_c_order, x0_arr, dir_arr, delta * i);
        if (sign == -1) {
          for (size_t j = 0; j < arr_native.nelem(); j++) arr_native[j] = -arr_native[j];
        }
        arr.from_array(arr_native);
      } else {
        fatal("moving_extremum only supports 2D and 3D");
      }
    }

    g->set(var.name, std::move(arr));  // Use move to preserve distribution metadata

#else
    // Non-MPI mode: generate full data
    if (ndims == 2) {
      double x0_arr[2] = {x0[0], x0[1]};
      double dir_arr[2] = {dir[0], dir[1]};
      auto arr_native = synthetic_moving_extremum<double, 2>(
          yaml_dims_c_order, x0_arr, dir_arr, delta * i);
      if (sign == -1) {
        for (size_t j = 0; j < arr_native.nelem(); j++) arr_native[j] = -arr_native[j];
      }
      ndarray<double, StoragePolicy> arr;
      arr.from_array(arr_native);
      g->set(var.name, arr);
    } else if (ndims == 3) {
      double x0_arr[3] = {x0[0], x0[1], x0[2]};
      double dir_arr[3] = {dir[0], dir[1], dir[2]};
      auto arr_native = synthetic_moving_extremum<double, 3>(
          yaml_dims_c_order, x0_arr, dir_arr, delta * i);
      if (sign == -1) {
        for (size_t j = 0; j < arr_native.nelem(); j++) arr_native[j] = -arr_native[j];
      }
      ndarray<double, StoragePolicy> arr;
      arr.from_array(arr_native);
      g->set(var.name, arr);
    } else {
      fatal("moving_extremum only supports 2D and 3D");
    }
#endif
  }
}

template <typename StoragePolicy>
inline void substream_synthetic_double_gyre<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  if (auto yA = y["A"])
    this->A = yA.as<double>();

  if (auto yomega = y["omega"])
    this->omega = yomega.as<double>();

  if (auto yepsilon = y["epsilon"])
    this->epsilon = yepsilon.as<double>();

  if (auto yt0 = y["t0"])
    this->t0 = yt0.as<double>();

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  assert(this->has_dimensions());
  assert(this->dimensions.size() == 2);
}

template <typename StoragePolicy>
inline void substream_synthetic_double_gyre<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) {
    auto arr_native = synthetic_double_gyre<double>(
        this->dimensions[0],
        this->dimensions[1],
        t0 + delta * i,
        false, A, omega, epsilon);
    // Convert to target storage policy
    ndarray<double, StoragePolicy> arr;
    arr.from_array(arr_native);
    g->set(var.name, arr);
  }
}

template <typename StoragePolicy>
inline void substream_synthetic_merger<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  if (auto yt0 = y["t0"])
    this->t0 = yt0.as<double>();

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  assert(this->has_dimensions());
  assert(this->dimensions.size() == 2);
}

template <typename StoragePolicy>
inline void substream_synthetic_merger<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) {
    auto arr_native = synthetic_merger_2D<double>(
        this->dimensions[0],
        this->dimensions[1],
        t0 + delta * i);
    // Convert to target storage policy
    ndarray<double, StoragePolicy> arr;
    arr.from_array(arr_native);
    g->set(var.name, arr);
  }
}

template <typename StoragePolicy>
inline void substream_synthetic_moving_ramp<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  if (auto yx0 = y["x0"])
    this->x0 = yx0.as<double>();

  if (auto yrate = y["rate"])
    this->rate = yrate.as<double>();

  if (auto ydelta = y["delta"])
    this->delta = ydelta.as<double>();

  assert(this->has_dimensions());
}

template <typename StoragePolicy>
inline void substream_synthetic_moving_ramp<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) {
    const int ndims = this->dimensions.size();

    // Convert dimensions from vector<int> to vector<size_t>
    std::vector<size_t> dims_size_t(this->dimensions.begin(), this->dimensions.end());

    if (ndims == 2) {
      auto arr_native = synthetic_moving_ramp<double, 2>(
          dims_size_t, x0, rate, delta * i);
      // Convert to target storage policy
      ndarray<double, StoragePolicy> arr;
      arr.from_array(arr_native);
      g->set(var.name, arr);
    } else if (ndims == 3) {
      auto arr_native = synthetic_moving_ramp<double, 3>(
          dims_size_t, x0, rate, delta * i);
      // Convert to target storage policy
      ndarray<double, StoragePolicy> arr;
      arr.from_array(arr_native);
      g->set(var.name, arr);
    } else {
      fatal("moving_ramp only supports 2D and 3D");
    }
  }
}

template <typename StoragePolicy>
inline void substream_synthetic_tornado<StoragePolicy>::initialize(YAML::Node y)
{
  substream_synthetic<StoragePolicy>::initialize(y);

  if (auto ytime_start = y["time_start"])
    this->time_start = ytime_start.as<int>();

  assert(this->has_dimensions());
  assert(this->dimensions.size() == 3);
}

template <typename StoragePolicy>
inline void substream_synthetic_tornado<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  for (const auto &var : this->variables) {
    auto arr_native = synthetic_tornado<double>(
        this->dimensions[0],
        this->dimensions[1],
        this->dimensions[2],
        time_start + i);
    // Convert to target storage policy
    ndarray<double, StoragePolicy> arr;
    arr.from_array(arr_native);
    g->set(var.name, arr);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_SYNTHETIC_HH
