#ifndef _DISTRIBUTED_NDARRAY_STREAM_HH
#define _DISTRIBUTED_NDARRAY_STREAM_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#include <ndarray/distributed_ndarray.hh>
#include <ndarray/distributed_ndarray_group.hh>
#include <ndarray/ndarray_stream.hh>
#include <mpi.h>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <string>
#include <vector>

namespace ftk {

/**
 * @brief Distributed stream for time-varying scientific data with YAML configuration
 *
 * Provides the same YAML-based stream interface as ndarray_stream, but returns
 * distributed_ndarray instances with automatic domain decomposition and parallel I/O.
 *
 * YAML Configuration Example:
 * @code{.yaml}
 * # Domain decomposition (optional, defaults to automatic)
 * decomposition:
 *   global_dims: [1000, 800, 600]  # Global domain dimensions
 *   nprocs: 0                       # 0 = use all available ranks
 *   pattern: []                     # Empty = automatic, or [2, 2, 1] for manual
 *   ghost: [1, 1, 1]               # Ghost layers per dimension
 *
 * # Data streams (same format as regular stream)
 * streams:
 *   - name: velocity
 *     format: netcdf
 *     filenames: simulation_*.nc
 *     vars:
 *       - name: u
 *       - name: v
 *       - name: w
 *
 *   - name: temperature
 *     format: binary
 *     filenames: temp_t*.bin
 *     dimensions: [1000, 800, 600]
 * @endcode
 *
 * Usage Example:
 * @code
 * ftk::distributed_stream<> stream(MPI_COMM_WORLD);
 * stream.parse_yaml("config.yaml");
 *
 * for (int t = 0; t < stream.n_timesteps(); t++) {
 *   auto group = stream.read(t);
 *   group["temperature"].exchange_ghosts();
 *   // ... process data ...
 * }
 * @endcode
 */
template <typename T = float, typename StoragePolicy = native_storage>
class distributed_stream {
public:
  using distributed_array_type = distributed_ndarray<T, StoragePolicy>;
  using distributed_group_type = distributed_ndarray_group<T, StoragePolicy>;
  using regular_stream_type = stream<StoragePolicy>;
  using regular_group_type = ndarray_group<StoragePolicy>;

  /**
   * @brief Constructor
   * @param comm MPI communicator
   */
  distributed_stream(MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm)
  {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);

    // Create underlying regular stream (will be used for metadata and file discovery)
    regular_stream_ = std::make_shared<regular_stream_type>(comm_);
  }

  /**
   * @brief Parse YAML configuration file
   *
   * YAML format supports:
   * - decomposition: Domain decomposition parameters
   *   - global_dims: [nx, ny, nz]
   *   - nprocs: number of processes (0 = all)
   *   - pattern: decomposition pattern (empty = auto, or [px, py, pz])
   *   - ghost: ghost layers per dimension [gx, gy, gz]
   *
   * - streams: List of substreams (same format as regular stream)
   *   - name, format, filenames, vars, dimensions, etc.
   *
   * @param yaml_file Path to YAML configuration file
   */
  void parse_yaml(const std::string& yaml_file)
  {
    YAML::Node config = YAML::LoadFile(yaml_file);

    // Parse decomposition parameters (optional)
    if (config["decomposition"]) {
      parse_decomposition(config["decomposition"]);
    }

    // Parse path prefix (optional)
    if (config["path_prefix"]) {
      path_prefix_ = config["path_prefix"].as<std::string>();
      regular_stream_->set_path_prefix(path_prefix_);
    }

    // Parse global dimensions from top-level (optional, can be overridden by decomposition)
    if (config["dimensions"] && global_dims_.empty()) {
      for (auto i = 0; i < config["dimensions"].size(); i++) {
        global_dims_.push_back(config["dimensions"][i].as<size_t>());
      }
    }

    // Parse streams/substreams
    if (config["streams"]) {
      for (auto i = 0; i < config["streams"].size(); i++) {
        regular_stream_->new_substream_from_yaml(config["streams"][i]);
      }
    }

    // If no explicit decomposition but we have dimensions, set them now
    if (!decomposition_set_ && !global_dims_.empty()) {
      set_decomposition(global_dims_, 0, {}, {});
    }

    n_timesteps_ = regular_stream_->total_timesteps();
  }

  /**
   * @brief Set domain decomposition parameters explicitly
   *
   * Can be called instead of or in addition to YAML configuration.
   * Overrides YAML decomposition if called after parse_yaml().
   *
   * @param global_dims Global array dimensions
   * @param nprocs Number of processes (0 = use communicator size)
   * @param decomp Decomposition pattern (empty = automatic)
   * @param ghost Ghost layers per dimension
   */
  void set_decomposition(const std::vector<size_t>& global_dims,
                        size_t nprocs = 0,
                        const std::vector<size_t>& decomp = {},
                        const std::vector<size_t>& ghost = {})
  {
    global_dims_ = global_dims;
    nprocs_decomp_ = (nprocs == 0) ? nprocs_ : nprocs;
    decomp_pattern_ = decomp;
    ghost_layers_ = ghost;
    decomposition_set_ = true;
  }

  /**
   * @brief Read all variables at specified timestep
   *
   * Reads data from all enabled substreams and returns a group containing
   * distributed arrays. Each array has the configured domain decomposition.
   *
   * @param timestep Timestep index
   * @return distributed_ndarray_group containing all variables
   */
  std::shared_ptr<distributed_group_type> read(int timestep)
  {
    if (!decomposition_set_) {
      throw std::runtime_error("Must set decomposition before reading (via YAML or set_decomposition())");
    }

    // Read using regular stream to get data
    auto regular_group = regular_stream_->read(timestep);

    // Convert to distributed group
    auto dist_group = std::make_shared<distributed_group_type>(comm_);

    // For each array in regular group, create distributed array
    for (const auto& name : regular_group->keys()) {
      distributed_array_type darray(comm_);

      // Set decomposition
      darray.decompose(global_dims_, nprocs_decomp_,
                       decomp_pattern_, ghost_layers_);

      // Get the data from regular array
      auto& regular_array = (*regular_group)[name];

      // Copy local portion to distributed array
      // Note: This assumes rank 0 has full data from regular stream
      // For true parallel reads, we need format-specific substreams

      if (rank_ == 0) {
        // Rank 0 has full data, distribute to others
        // For now, each rank reads independently using read_parallel
        // This will be optimized with proper substream integration
      }

      // For parallel formats (NetCDF, HDF5), read directly in parallel
      // This requires format detection and calling read_parallel
      std::string filename = get_filename_for_timestep(timestep);
      if (!filename.empty()) {
        darray.read_parallel(filename, name, timestep);
      }

      dist_group->add(name, std::move(darray));
    }

    return dist_group;
  }

  /**
   * @brief Read static (time-independent) variables
   * @return distributed_ndarray_group containing static variables
   */
  std::shared_ptr<distributed_group_type> read_static()
  {
    if (!decomposition_set_) {
      throw std::runtime_error("Must set decomposition before reading");
    }

    auto regular_group = regular_stream_->read_static();
    auto dist_group = std::make_shared<distributed_group_type>(comm_);

    // Convert each static array to distributed
    for (const auto& name : regular_group->keys()) {
      distributed_array_type darray(comm_);
      darray.decompose(global_dims_, nprocs_decomp_,
                       decomp_pattern_, ghost_layers_);

      // Read in parallel (for supported formats)
      // For unsupported formats, will need to scatter from rank 0

      dist_group->add(name, std::move(darray));
    }

    return dist_group;
  }

  /**
   * @brief Get total number of timesteps
   * @return Number of timesteps
   */
  int n_timesteps() const
  {
    return n_timesteps_;
  }

  /**
   * @brief Iterator-style interface for processing all timesteps
   *
   * Example:
   * @code
   * stream.for_each_timestep([](int t, auto& group) {
   *   group["temperature"].exchange_ghosts();
   *   // process...
   * });
   * @endcode
   *
   * @param callback Function to call for each timestep
   */
  template <typename Callback>
  void for_each_timestep(Callback callback)
  {
    for (int t = 0; t < n_timesteps_; t++) {
      auto group = read(t);
      callback(t, *group);
    }
  }

  /**
   * @brief Set path prefix for all file paths
   * @param prefix Path prefix string
   */
  void set_path_prefix(const std::string& prefix)
  {
    path_prefix_ = prefix;
    regular_stream_->set_path_prefix(prefix);
  }

  // Accessors
  int rank() const { return rank_; }
  int nprocs() const { return nprocs_; }
  MPI_Comm comm() const { return comm_; }

  const std::vector<size_t>& global_dims() const { return global_dims_; }
  const std::vector<size_t>& decomp_pattern() const { return decomp_pattern_; }
  const std::vector<size_t>& ghost_layers() const { return ghost_layers_; }

private:
  /**
   * @brief Parse decomposition section from YAML
   * @param y YAML node containing decomposition config
   */
  void parse_decomposition(YAML::Node y)
  {
    if (y["global_dims"]) {
      for (auto i = 0; i < y["global_dims"].size(); i++) {
        global_dims_.push_back(y["global_dims"][i].as<size_t>());
      }
    }

    if (y["nprocs"]) {
      nprocs_decomp_ = y["nprocs"].as<size_t>();
      if (nprocs_decomp_ == 0) nprocs_decomp_ = nprocs_;
    } else {
      nprocs_decomp_ = nprocs_;
    }

    if (y["pattern"]) {
      for (auto i = 0; i < y["pattern"].size(); i++) {
        decomp_pattern_.push_back(y["pattern"][i].as<size_t>());
      }
    }

    if (y["ghost"]) {
      for (auto i = 0; i < y["ghost"].size(); i++) {
        ghost_layers_.push_back(y["ghost"][i].as<size_t>());
      }
    }

    decomposition_set_ = true;
  }

  /**
   * @brief Get filename for specified timestep from underlying stream
   * @param timestep Timestep index
   * @return Filename string
   */
  std::string get_filename_for_timestep(int timestep) const
  {
    // Query first enabled substream for filename
    // This is simplified - full implementation would handle multiple substreams
    if (!regular_stream_->substreams.empty()) {
      for (auto& sub : regular_stream_->substreams) {
        if (sub->is_enabled && !sub->filenames.empty()) {
          int file_idx = sub->locate_timestep_file_index(timestep);
          if (file_idx >= 0 && file_idx < sub->filenames.size()) {
            return sub->filenames[file_idx];
          }
        }
      }
    }
    return "";
  }

  MPI_Comm comm_;
  int rank_;
  int nprocs_;

  // Decomposition parameters
  std::vector<size_t> global_dims_;
  size_t nprocs_decomp_;
  std::vector<size_t> decomp_pattern_;
  std::vector<size_t> ghost_layers_;
  bool decomposition_set_ = false;

  // Stream parameters
  std::string path_prefix_;
  int n_timesteps_ = 0;

  // Underlying regular stream for YAML parsing and file discovery
  std::shared_ptr<regular_stream_type> regular_stream_;
};

} // namespace ftk

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#endif // _DISTRIBUTED_NDARRAY_STREAM_HH
