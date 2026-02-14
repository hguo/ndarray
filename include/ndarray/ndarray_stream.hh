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
 * @brief Variable metadata for stream I/O
 *
 * Describes a variable in a stream, including its name, dimensions,
 * data type, and format-specific properties.
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

  void parse_yaml(YAML::Node);
};

struct substream;

/**
 * @brief Stream for time-varying multi-format scientific data
 *
 * Provides unified interface for reading time-series data from various
 * formats (NetCDF, HDF5, ADIOS2, VTK, binary, synthetic).
 *
 * @code
 * ftk::stream s;
 * s.set_input_source_netcdf_file("data.nc");
 * for (int t = 0; t < s.n_timesteps(); t++) {
 *   auto g = s.read(t);
 *   // process data
 * }
 * @endcode
 */
struct stream {
  stream(MPI_Comm comm = MPI_COMM_WORLD);
  ~stream() {};

  /**
   * @brief Read data at specified timestep
   * @param i Timestep index
   * @return ndarray_group containing all variables for this timestep
   */
  std::shared_ptr<ndarray_group> read(int i);

  /**
   * @brief Read static (time-independent) data
   * @return ndarray_group containing static variables
   */
  std::shared_ptr<ndarray_group> read_static();

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

public:
  std::vector<std::shared_ptr<substream>> substreams;
  bool has_adios2_substream = false;

  std::string path_prefix;

  MPI_Comm comm;

  // dimensions (all substreams and their variables will use the same dimensions if specified)
  std::vector<int> dimensions;

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
struct substream {
  substream(stream& s) : stream_(s) {}
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
  virtual void read(int i, std::shared_ptr<ndarray_group> g) = 0;

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
  stream &stream_;

  // variables
  std::vector<variable> variables;

  // communicator
  MPI_Comm comm = MPI_COMM_WORLD;
};

///////////
// Implementation
///////////

inline stream::stream(MPI_Comm comm_) :
#if NDARRAY_HAVE_ADIOS2
  adios(comm_),
#endif
  comm(comm_)
{
}

inline void stream::parse_yaml(const std::string filename)
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

inline std::shared_ptr<ndarray_group> stream::read_static()
{
  std::shared_ptr<ndarray_group> g(new ndarray_group);

  for (auto sub : this->substreams)
    if (sub->is_enabled && sub->is_static) {
      sub->read(0, g);
    }

  return g;
}

inline int stream::total_timesteps() const
{
  for (auto sub : this->substreams)
    if (!sub->is_static)
      return sub->total_timesteps;

  return 0;
}

inline std::shared_ptr<ndarray_group> stream::read(int i)
{
  std::shared_ptr<ndarray_group> g(new ndarray_group);

  for (auto &sub : this->substreams)
    if (sub->is_enabled && !sub->is_static)
      sub->read(i, g);

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

inline int substream::locate_timestep_file_index(int i)
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

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

#endif // _NDARRAY_STREAM_HH
