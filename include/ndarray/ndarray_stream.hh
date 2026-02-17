#ifndef _NDARRAY_STREAM_HH
#define _NDARRAY_STREAM_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML

#include <ndarray/ndarray_group.hh>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif
#include <yaml-cpp/yaml.h>

namespace ftk {

/**
 * @brief Stream format types
 */
enum {
  SUBSTREAM_NONE = 0, // invalid
  SUBSTREAM_SYNTHETIC,
  SUBSTREAM_BINARY,
  SUBSTREAM_NETCDF,
  SUBSTREAM_PNETCDF,
  SUBSTREAM_HDF5,
  SUBSTREAM_ADIOS2,
  SUBSTREAM_VTU,
  SUBSTREAM_VTU_RESAMPLE,
  SUBSTREAM_VTI,
  SUBSTREAM_PNG,
  SUBSTREAM_AMIRA,
  SUBSTREAM_NUMPY
};

/**
 * @brief Stream direction (input or output)
 */
enum {
  SUBSTREAM_DIR_INPUT,
  SUBSTREAM_DIR_OUTPUT
};

/**
 * @brief Variable distribution configuration
 *
 * Specifies how a variable should be distributed across MPI ranks.
 * Default is REPLICATED (safe, works for all cases).
 */
enum class VariableDistType {
  DISTRIBUTED,  // Domain-decomposed across ranks
  REPLICATED,   // Full data on all ranks (default)
  AUTO          // Auto-detect based on size/usage
};

/**
 * @brief Variable-specific decomposition configuration
 *
 * Allows per-variable customization of domain decomposition.
 */
struct variable_decomposition {
  std::vector<size_t> dims;     // Variable-specific dimensions (empty = use stream default)
  std::vector<size_t> pattern;  // Decomposition pattern (empty = auto, 0 = don't split)
  std::vector<size_t> ghost;    // Ghost layers per dimension
};

/**
 * @brief Variable metadata for stream I/O
 *
 * Describes a variable in a stream, including its name, dimensions,
 * data type, format-specific properties, and distribution configuration.
 */
struct variable {
  std::string name;
  std::vector<std::string> possible_names; // will prioritize possible_names by ordering
  std::vector<std::string> name_patterns;  // wildcard patterns (e.g., "time*_avg_temperature")

  bool is_optional = false; // will be ignored if the format is binary

  bool is_dims_auto = true;
  std::vector<int> dimensions;
  unsigned char order = NDARRAY_ORDER_C;

  bool is_dtype_auto = true;
  int dtype = NDARRAY_DTYPE_UNKNOWN;

  bool is_multicomponents_auto = true;
  bool multicomponents = false;

  bool is_offset_auto = true; // only apply to binary
  size_t offset = 0;

#if NDARRAY_USE_LITTLE_ENDIAN
  bool endian = NDARRAY_ENDIAN_LITTLE;
#else
  bool endian = NDARRAY_ENDIAN_BIG;
#endif

  // NEW: Distribution configuration
#if NDARRAY_HAVE_MPI
  VariableDistType dist_type = VariableDistType::REPLICATED;  // Default: replicated
  bool has_custom_decomposition = false;
  variable_decomposition custom_decomp;
#endif

  void parse_yaml(YAML::Node);
};

template <typename StoragePolicy> struct substream;

/**
 * @brief Stream for time-varying multi-format scientific data
 *
 * Provides unified interface for reading time-series data from various
 * formats (NetCDF, HDF5, ADIOS2, VTK, binary, synthetic).
 *
 * @code
 * ftk::stream<> s;
 * s.set_input_source_netcdf_file("data.nc");
 * for (int t = 0; t < s.n_timesteps(); t++) {
 *   auto g = s.read(t);
 *   // process data
 * }
 * @endcode
 */
template <typename StoragePolicy = native_storage>
struct stream {
  using group_type = ndarray_group<StoragePolicy>;

  stream(MPI_Comm comm = MPI_COMM_WORLD);
  ~stream() {};

  /**
   * @brief Read data at specified timestep
   * @param i Timestep index
   * @return ndarray_group containing all variables for this timestep
   */
  std::shared_ptr<group_type> read(int i);

  /**
   * @brief Read static (time-independent) data
   * @return ndarray_group containing static variables
   */
  std::shared_ptr<group_type> read_static();

  /**
   * @brief Parse YAML configuration file
   * @param filename Path to YAML file describing stream
   */
  void parse_yaml(const std::string filename);

  /**
   * @brief Get total number of timesteps
   * @return Number of timesteps in stream
   */
  int total_timesteps() const;

  /**
   * @brief Create substream from YAML node
   * @param y YAML node describing substream
   */
  void new_substream_from_yaml(YAML::Node y);

  /**
   * @brief Set path prefix for all file paths
   * @param p Path prefix string
   */
  void set_path_prefix(const std::string p) { path_prefix = p; }

  /**
   * @brief Check if stream has dimensions specified
   * @return true if dimensions are set
   */
  bool has_dimensions() const { return !dimensions.empty(); }

#if NDARRAY_HAVE_MPI
  /**
   * @brief Set default decomposition for distributed variables
   *
   * Sets global decomposition parameters that apply to all variables
   * marked as DISTRIBUTED (unless they have custom decomposition).
   *
   * @param global_dims Global array dimensions
   * @param nprocs Number of processes (0 = use comm size)
   * @param decomp Decomposition pattern (empty = auto, 0 = don't split)
   * @param ghost Ghost layers per dimension
   */
  void set_default_decomposition(const std::vector<size_t>& global_dims,
                                  size_t nprocs = 0,
                                  const std::vector<size_t>& decomp = {},
                                  const std::vector<size_t>& ghost = {});

  /**
   * @brief Configure a variable's distribution type
   *
   * @param varname Variable name
   * @param type Distribution type (DISTRIBUTED, REPLICATED, AUTO)
   */
  void set_variable_distribution(const std::string& varname, VariableDistType type);

  /**
   * @brief Configure a variable with custom decomposition
   *
   * @param varname Variable name
   * @param decomp Custom decomposition configuration
   */
  void set_variable_decomposition(const std::string& varname,
                                   const variable_decomposition& decomp);
#endif

public:
  std::vector<std::shared_ptr<substream<StoragePolicy>>> substreams;
  bool has_adios2_substream = false;

  std::string path_prefix;

  MPI_Comm comm;
  int rank = 0;
  int nprocs = 1;

  // dimensions (all substreams and their variables will use the same dimensions if specified)
  std::vector<int> dimensions;

#if NDARRAY_HAVE_MPI
  // Default decomposition for distributed variables
  bool has_default_decomposition = false;
  std::vector<size_t> default_global_dims;
  size_t default_nprocs = 0;
  std::vector<size_t> default_decomp_pattern;
  std::vector<size_t> default_ghost_layers;

  // Per-variable configurations (maps variable name to config)
  std::map<std::string, VariableDistType> variable_dist_types;
  std::map<std::string, variable_decomposition> variable_decompositions;
#endif

#if NDARRAY_HAVE_ADIOS2
  adios2::ADIOS adios;
  adios2::IO io;
#endif
};

/**
 * @brief Base class for format-specific substreams
 *
 * A substream handles I/O for a specific file format within a stream.
 * Subclasses implement format-specific read operations.
 */
template <typename StoragePolicy = native_storage>
struct substream {
  using group_type = ndarray_group<StoragePolicy>;
  using stream_type = stream<StoragePolicy>;

  substream(stream_type& s) : stream_(s) {}
  virtual ~substream() {}

  /**
   * @brief Initialize substream from YAML configuration
   * @param y YAML node with substream config
   */
  virtual void initialize(YAML::Node y) = 0;

  /**
   * @brief Locate which file contains given timestep
   * @param i Timestep index
   * @return File index, or -1 if not found
   */
  virtual int locate_timestep_file_index(int i);

  /**
   * @brief Read data at timestep into group
   * @param i Timestep index
   * @param g Output ndarray_group
   */
  virtual void read(int i, std::shared_ptr<group_type> g) = 0;

  /**
   * @brief Get stream direction (input or output)
   * @return SUBSTREAM_DIR_INPUT or SUBSTREAM_DIR_OUTPUT
   */
  virtual int direction() = 0;

  /**
   * @brief Check if substream requires input files
   * @return true if files needed
   */
  virtual bool require_input_files() = 0;

  /**
   * @brief Check if substream requires dimensions to be specified
   * @return true if dimensions required
   */
  virtual bool require_dimensions() = 0;

  /**
   * @brief Check if substream has dimensions set
   * @return true if dimensions are set
   */
  bool has_dimensions() const { return !dimensions.empty(); }

  // status
  bool is_enabled = true;
  bool is_optional = false;

  // yaml properties
  bool is_static = false;
  std::string name;
  std::string filename_pattern;

  // files and timesteps
  std::vector<std::string> filenames;
  std::vector<int> timesteps_per_file, first_timestep_per_file;
  int total_timesteps = 0;
  int current_file_index = 0;

  // optional dimensions (all variables will use the same dimensions if specified)
  std::vector<int> dimensions;

  // reference to the parent stream
  stream_type &stream_;

  // variables
  std::vector<variable> variables;

  // communicator
  MPI_Comm comm = MPI_COMM_WORLD;
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline stream<StoragePolicy>::stream(MPI_Comm comm_) :
#if NDARRAY_HAVE_ADIOS2
  adios(comm_),
#endif
  comm(comm_)
{
#if NDARRAY_HAVE_MPI
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
#endif
}

#if NDARRAY_HAVE_MPI
template <typename StoragePolicy>
inline void stream<StoragePolicy>::set_default_decomposition(
  const std::vector<size_t>& global_dims,
  size_t nprocs_,
  const std::vector<size_t>& decomp,
  const std::vector<size_t>& ghost)
{
  has_default_decomposition = true;
  default_global_dims = global_dims;
  default_nprocs = (nprocs_ == 0) ? nprocs : nprocs_;
  default_decomp_pattern = decomp;
  default_ghost_layers = ghost;
}

template <typename StoragePolicy>
inline void stream<StoragePolicy>::set_variable_distribution(
  const std::string& varname,
  VariableDistType type)
{
  variable_dist_types[varname] = type;
}

template <typename StoragePolicy>
inline void stream<StoragePolicy>::set_variable_decomposition(
  const std::string& varname,
  const variable_decomposition& decomp)
{
  variable_decompositions[varname] = decomp;
  // Also mark as distributed
  variable_dist_types[varname] = VariableDistType::DISTRIBUTED;
}
#endif // NDARRAY_HAVE_MPI

template <typename StoragePolicy>
inline void stream<StoragePolicy>::parse_yaml(const std::string filename)
{
  YAML::Node yaml = YAML::LoadFile(filename);
  auto yroot = yaml["stream"];

  if (auto yprefix = yroot["path_prefix"]) {
    if (this->path_prefix.empty())
      this->path_prefix = yprefix.as<std::string>();
  }

  if (auto ydimensions = yroot["dimensions"]) { // this will override all dimensions in all substreams and variables
      for (auto i = 0; i < ydimensions.size(); i ++)
        dimensions.push_back(ydimensions[i].as<int>());
  }

#if NDARRAY_HAVE_MPI
  // Parse decomposition section (default for all distributed variables)
  if (auto ydecomp = yroot["decomposition"]) {
    std::vector<size_t> global_dims;
    std::vector<size_t> pattern;
    std::vector<size_t> ghost;
    size_t np = 0;

    if (auto ygdims = ydecomp["global_dims"]) {
      for (auto i = 0; i < ygdims.size(); i++)
        global_dims.push_back(ygdims[i].as<size_t>());
    }

    if (auto ynprocs = ydecomp["nprocs"])
      np = ynprocs.as<size_t>();

    if (auto ypattern = ydecomp["pattern"]) {
      for (auto i = 0; i < ypattern.size(); i++)
        pattern.push_back(ypattern[i].as<size_t>());
    }

    if (auto yghost = ydecomp["ghost"]) {
      for (auto i = 0; i < yghost.size(); i++)
        ghost.push_back(yghost[i].as<size_t>());
    }

    if (!global_dims.empty()) {
      set_default_decomposition(global_dims, np, pattern, ghost);
    }
  }

  // Parse per-variable configurations
  if (auto yvars = yroot["variables"]) {
    for (YAML::const_iterator it = yvars.begin(); it != yvars.end(); ++it) {
      std::string varname = it->first.as<std::string>();
      YAML::Node varconfig = it->second;

      // Parse distribution type
      if (auto ytype = varconfig["type"]) {
        std::string type_str = ytype.as<std::string>();
        if (type_str == "distributed") {
          set_variable_distribution(varname, VariableDistType::DISTRIBUTED);
        } else if (type_str == "replicated") {
          set_variable_distribution(varname, VariableDistType::REPLICATED);
        } else if (type_str == "auto") {
          set_variable_distribution(varname, VariableDistType::AUTO);
        }
      }
      // If no type specified, default is REPLICATED (safe)

      // Parse custom decomposition for this variable
      if (auto yvardecomp = varconfig["decomposition"]) {
        variable_decomposition decomp;

        if (auto ydims = yvardecomp["dims"]) {
          for (auto i = 0; i < ydims.size(); i++)
            decomp.dims.push_back(ydims[i].as<size_t>());
        }

        if (auto ypattern = yvardecomp["pattern"]) {
          for (auto i = 0; i < ypattern.size(); i++)
            decomp.pattern.push_back(ypattern[i].as<size_t>());
        }

        if (auto yghost = yvardecomp["ghost"]) {
          for (auto i = 0; i < yghost.size(); i++)
            decomp.ghost.push_back(yghost[i].as<size_t>());
        }

        set_variable_decomposition(varname, decomp);
      }
    }
  }
#endif // NDARRAY_HAVE_MPI

  if (auto ysubstreams = yroot["substreams"]) { // has substreams
    for (auto i = 0; i < ysubstreams.size(); i ++) {
      auto ysubstream = ysubstreams[i];

      if (auto yformat = ysubstream["format"]) {
        std::string format = yformat.as<std::string>();
        if (format == "adios2" && !has_adios2_substream) {
          // here's where adios2 is initialized
          has_adios2_substream = true;
#if NDARRAY_HAVE_ADIOS2
          io = adios.DeclareIO("BPReader");
#endif
        }
      } else
        fatal(nd::ERR_STREAM_FORMAT);

      new_substream_from_yaml(ysubstream);
    }
  }
}

template <typename StoragePolicy>
inline std::shared_ptr<typename stream<StoragePolicy>::group_type> stream<StoragePolicy>::read_static()
{
  std::shared_ptr<group_type> g(new group_type);

  for (auto sub : this->substreams)
    if (sub->is_enabled && sub->is_static) {
      sub->read(0, g);
    }

#if NDARRAY_HAVE_MPI
  // Configure MPI distribution for static arrays (same logic as read())
  if (nprocs > 1) {
    for (auto& kv : *g) {
      const std::string& varname = kv.first;
      auto& array = kv.second;

      VariableDistType dist_type = VariableDistType::REPLICATED;  // Default
      if (variable_dist_types.count(varname)) {
        dist_type = variable_dist_types.at(varname);
      }

      if (dist_type == VariableDistType::DISTRIBUTED) {
        variable_decomposition decomp;
        bool has_custom = variable_decompositions.count(varname) > 0;

        if (has_custom) {
          decomp = variable_decompositions.at(varname);
        }

        std::vector<size_t> dims = has_custom && !decomp.dims.empty()
                                    ? decomp.dims : default_global_dims;
        std::vector<size_t> pattern = has_custom && !decomp.pattern.empty()
                                       ? decomp.pattern : default_decomp_pattern;
        std::vector<size_t> ghost = has_custom && !decomp.ghost.empty()
                                     ? decomp.ghost : default_ghost_layers;
        size_t np = default_nprocs;

        if (dims.empty()) {
          dims.resize(array.nd());
          for (size_t d = 0; d < array.nd(); d++) {
            dims[d] = array.dim(d);
          }
        }

        array.decompose(comm, dims, np, pattern, ghost);

      } else if (dist_type == VariableDistType::REPLICATED) {
        array.set_replicated(comm);
      }
    }
  }
#endif // NDARRAY_HAVE_MPI

  return g;
}

template <typename StoragePolicy>
inline int stream<StoragePolicy>::total_timesteps() const
{
  for (auto sub : this->substreams)
    if (!sub->is_static)
      return sub->total_timesteps;

  return 0;
}

template <typename StoragePolicy>
inline std::shared_ptr<typename stream<StoragePolicy>::group_type> stream<StoragePolicy>::read(int i)
{
  std::shared_ptr<group_type> g(new group_type);

  for (auto &sub : this->substreams)
    if (sub->is_enabled && !sub->is_static)
      sub->read(i, g);

#if NDARRAY_HAVE_MPI
  // Configure MPI distribution for each array based on variable settings
  if (nprocs > 1) {
    for (auto& kv : *g) {
      const std::string& varname = kv.first;
      auto& array = kv.second;

      // Determine distribution type for this variable
      VariableDistType dist_type = VariableDistType::REPLICATED;  // Default
      if (variable_dist_types.count(varname)) {
        dist_type = variable_dist_types.at(varname);
      }

      // Configure array based on distribution type
      if (dist_type == VariableDistType::DISTRIBUTED) {
        // Get decomposition parameters for this variable
        variable_decomposition decomp;
        bool has_custom = variable_decompositions.count(varname) > 0;

        if (has_custom) {
          decomp = variable_decompositions.at(varname);
        }

        // Use custom or default decomposition
        std::vector<size_t> dims = has_custom && !decomp.dims.empty()
                                    ? decomp.dims : default_global_dims;
        std::vector<size_t> pattern = has_custom && !decomp.pattern.empty()
                                       ? decomp.pattern : default_decomp_pattern;
        std::vector<size_t> ghost = has_custom && !decomp.ghost.empty()
                                     ? decomp.ghost : default_ghost_layers;
        size_t np = default_nprocs;

        // If dimensions not specified, infer from array shape
        if (dims.empty()) {
          dims.resize(array.nd());
          for (size_t d = 0; d < array.nd(); d++) {
            dims[d] = array.dim(d);
          }
        }

        // Configure as distributed
        array.decompose(comm, dims, np, pattern, ghost);

      } else if (dist_type == VariableDistType::REPLICATED) {
        // Configure as replicated
        array.set_replicated(comm);

      }
      // AUTO: leave as serial for now (could implement heuristics later)
    }
  }
#endif // NDARRAY_HAVE_MPI

  return g;
}

inline void variable::parse_yaml(YAML::Node y)
{
  this->name = y["name"].as<std::string>();

  if (auto ypvar = y["possible_names"]) {
    for (auto j = 0; j < ypvar.size(); j ++)
      this->possible_names.push_back(ypvar[j].as<std::string>());
  } else
    this->possible_names.push_back( this->name );

  // Handle format-specific variable names (h5_name, nc_name, etc.)
  // These are added to possible_names to support format-specific naming
  if (auto yh5name = y["h5_name"]) {
    this->possible_names.push_back(yh5name.as<std::string>());
  }
  if (auto yncname = y["nc_name"]) {
    this->possible_names.push_back(yncname.as<std::string>());
  }

  if (auto ypatterns = y["name_patterns"]) {
    for (auto j = 0; j < ypatterns.size(); j ++)
      this->name_patterns.push_back(ypatterns[j].as<std::string>());
  }

  if (auto ydtype = y["dtype"]) {
    this->dtype = ndarray_base::str2dtype( ydtype.as<std::string>() );
    if (this->dtype != NDARRAY_DTYPE_UNKNOWN)
      this->is_dtype_auto = false;
  }

  if (auto ymulticomponents = y["multicomponents"]) {
    this->is_multicomponents_auto = false;
    this->multicomponents = ymulticomponents.as<bool>();
  }

  if (auto yoffset = y["offset"]) {
    if (yoffset.as<std::string>() == "auto")
      this->is_offset_auto = true;
    else {
      this->is_offset_auto = false;
      this->offset = yoffset.as<size_t>();
    }
  }

  if (auto yorder = y["dimension_order"]) {
    if (yorder.as<std::string>() == "f")
      this->order = NDARRAY_ORDER_F;
  }

  if (auto ydims = y["dimensions"]) {
    if (ydims.IsScalar()) { // auto
      this->is_dims_auto = true;
    } else if (ydims.IsSequence()) {
      for (auto i = 0; i < ydims.size(); i ++)
        this->dimensions.push_back( ydims[i].as<int>() );
    } else
      throw nd::ERR_STREAM_FORMAT;
  }

  if (auto yendian = y["endian"]) {
    if (yendian.as<std::string>() == "big")
      this->endian = NDARRAY_ENDIAN_BIG;
    else if (yendian.as<std::string>() == "little")
      this->endian = NDARRAY_ENDIAN_LITTLE;
  }

  if (auto yopt = y["optional"]) {
    this->is_optional = yopt.as<bool>();
  }
}

template <typename StoragePolicy>
inline int substream<StoragePolicy>::locate_timestep_file_index(int i)
{
  int fi = current_file_index;
  int ft = first_timestep_per_file[fi],
      nt = timesteps_per_file[fi];

  if (i >= ft && i < ft + nt)
    return fi;
  else {
    for (fi = 0; fi < filenames.size(); fi ++) {
      ft = first_timestep_per_file[fi];
      nt = timesteps_per_file[fi];

      if (i >= ft && i < ft + nt) {
        current_file_index = fi;
        return fi;
      }
    }
  }

  return -1; // not found
}

// Type aliases for convenience
using stream_native = stream<native_storage>;
using substream_native = substream<native_storage>;

#if NDARRAY_HAVE_XTENSOR
using stream_xtensor = stream<xtensor_storage>;
using substream_xtensor = substream<xtensor_storage>;
#endif

#if NDARRAY_HAVE_EIGEN
using stream_eigen = stream<eigen_storage>;
using substream_eigen = substream<eigen_storage>;
#endif

} // namespace ftk

// Include template implementations (must come after class definitions)
#include <ndarray/ndarray_group_stream.hh>

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_HH
