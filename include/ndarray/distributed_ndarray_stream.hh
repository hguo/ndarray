#ifndef _DISTRIBUTED_NDARRAY_STREAM_HH
#define _DISTRIBUTED_NDARRAY_STREAM_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_MPI

#include <ndarray/distributed_ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <mpi.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace ftk {

/**
 * @brief Distributed stream for time-varying scientific data
 *
 * Provides unified interface for reading time-series data in distributed
 * memory settings. Each MPI rank reads its local portion automatically.
 *
 * Wraps ndarray_stream functionality with automatic domain decomposition
 * and parallel I/O for each timestep.
 *
 * @code
 * ftk::distributed_stream<> stream(MPI_COMM_WORLD);
 * stream.set_decomposition({1000, 800}, 0, {}, {1, 1});
 * stream.add_variable("temperature");
 * stream.set_input_source("data.nc");
 *
 * for (int t = 0; t < stream.n_timesteps(); t++) {
 *   auto vars = stream.read(t);
 *   auto& temperature = vars["temperature"];
 *   temperature.exchange_ghosts();
 *   // ... process data ...
 * }
 * @endcode
 */
template <typename T = float, typename StoragePolicy = native_storage>
class distributed_stream {
public:
  using distributed_array_type = distributed_ndarray<T, StoragePolicy>;
  using variable_map_type = std::map<std::string, distributed_array_type>;

  /**
   * @brief Constructor
   * @param comm MPI communicator
   */
  distributed_stream(MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm)
  {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);
  }

  /**
   * @brief Set domain decomposition parameters
   *
   * All variables will use the same decomposition.
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
   * @brief Add variable to read from stream
   * @param name Variable name
   */
  void add_variable(const std::string& name)
  {
    variable_names_.push_back(name);
  }

  /**
   * @brief Set input source (file pattern or YAML config)
   *
   * For file patterns:
   *   - "data.nc" - single file with multiple timesteps
   *   - "data_{timestep}.nc" - one file per timestep
   *   - "data.yaml" - YAML configuration file
   *
   * @param source Input source specification
   */
  void set_input_source(const std::string& source)
  {
    input_source_ = source;

    // Detect if it's a YAML config or direct file
    if (source.find(".yaml") != std::string::npos ||
        source.find(".yml") != std::string::npos) {
      use_yaml_config_ = true;
#if NDARRAY_HAVE_YAML
      parse_yaml_config(source);
#else
      throw std::runtime_error("YAML support not enabled");
#endif
    } else {
      use_yaml_config_ = false;
      // Direct file pattern
      file_pattern_ = source;
    }
  }

  /**
   * @brief Get number of timesteps
   *
   * For YAML config, reads from configuration.
   * For file patterns, must be set explicitly via set_n_timesteps().
   *
   * @return Number of timesteps
   */
  int n_timesteps() const
  {
    return n_timesteps_;
  }

  /**
   * @brief Set number of timesteps (for non-YAML sources)
   * @param n Number of timesteps
   */
  void set_n_timesteps(int n)
  {
    n_timesteps_ = n;
  }

  /**
   * @brief Read all variables at specified timestep
   *
   * Returns a map of variable names to distributed arrays.
   * Each array contains the local portion of data for this rank.
   *
   * @param timestep Timestep index
   * @return Map of variable name to distributed_ndarray
   */
  variable_map_type read(int timestep)
  {
    if (!decomposition_set_) {
      throw std::runtime_error("Must call set_decomposition() before reading");
    }

    variable_map_type vars;

    // Read each variable
    for (const auto& varname : variable_names_) {
      distributed_array_type darray(comm_);

      // Set decomposition
      darray.decompose(global_dims_, nprocs_decomp_,
                       decomp_pattern_, ghost_layers_);

      // Determine filename for this timestep
      std::string filename = get_filename(timestep);

      // Read in parallel
      darray.read_parallel(filename, varname, timestep);

      // Store in map
      vars.emplace(varname, std::move(darray));
    }

    return vars;
  }

  /**
   * @brief Read single variable at specified timestep
   * @param varname Variable name
   * @param timestep Timestep index
   * @return distributed_ndarray containing the variable data
   */
  distributed_array_type read_var(const std::string& varname, int timestep)
  {
    if (!decomposition_set_) {
      throw std::runtime_error("Must call set_decomposition() before reading");
    }

    distributed_array_type darray(comm_);
    darray.decompose(global_dims_, nprocs_decomp_,
                     decomp_pattern_, ghost_layers_);

    std::string filename = get_filename(timestep);
    darray.read_parallel(filename, varname, timestep);

    return darray;
  }

  /**
   * @brief Iterator-style interface for processing all timesteps
   *
   * Example:
   * @code
   * stream.for_each_timestep([](int t, auto& vars) {
   *   auto& temperature = vars["temperature"];
   *   temperature.exchange_ghosts();
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
      auto vars = read(t);
      callback(t, vars);
    }
  }

  // Accessors
  int rank() const { return rank_; }
  int nprocs() const { return nprocs_; }
  MPI_Comm comm() const { return comm_; }

private:
  /**
   * @brief Get filename for specified timestep
   * @param timestep Timestep index
   * @return Filename string
   */
  std::string get_filename(int timestep) const
  {
    if (use_yaml_config_) {
      // For YAML config, filenames are managed by underlying stream
      return file_pattern_;
    }

    // Check if pattern contains {timestep} placeholder
    std::string filename = file_pattern_;
    std::string placeholder = "{timestep}";
    size_t pos = filename.find(placeholder);
    if (pos != std::string::npos) {
      // Replace {timestep} with actual timestep number
      filename.replace(pos, placeholder.length(), std::to_string(timestep));
    }

    return filename;
  }

#if NDARRAY_HAVE_YAML
  /**
   * @brief Parse YAML configuration file
   * @param yaml_file Path to YAML file
   */
  void parse_yaml_config(const std::string& yaml_file)
  {
    // Use underlying ndarray_stream to parse YAML
    // This gets us the file patterns, variables, and timestep counts
    stream<StoragePolicy> base_stream(comm_);
    base_stream.parse_yaml(yaml_file);

    n_timesteps_ = base_stream.total_timesteps();

    // Extract file pattern from first substream
    // (simplified - full implementation would handle multiple substreams)
    if (!base_stream.substreams.empty()) {
      // Store reference to use during reads
      file_pattern_ = yaml_file;  // Keep YAML file for reference
    }
  }
#endif

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
  std::string input_source_;
  std::string file_pattern_;
  std::vector<std::string> variable_names_;
  int n_timesteps_ = 0;
  bool use_yaml_config_ = false;
};

} // namespace ftk

#endif // NDARRAY_HAVE_MPI

#endif // _DISTRIBUTED_NDARRAY_STREAM_HH
