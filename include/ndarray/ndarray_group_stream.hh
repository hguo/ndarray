#ifndef _NDARRAY_NDARRAY_GROUP_STREAM_HH
#define _NDARRAY_NDARRAY_GROUP_STREAM_HH

#include <ndarray/ndarray_group.hh>
#include <yaml-cpp/yaml.h>

namespace ndarray {

enum {
  STREAM_NONE = 0, // invalid
  STREAM_SYNTHETIC,
  STREAM_BINARY,
  STREAM_NETCDF,
  STREAM_PNETCDF,
  STREAM_HDF5,
  STREAM_ADIOS2,
  STREAM_VTU,
  STREAM_VTI,
  STREAM_PNG,
  STREAM_AMIRA,
  STREAM_NUMPY
};

struct variable {
  std::string name;
  std::set<std::string> possible_names;

  bool is_optional = false;
  bool is_dims_auto = true;
  std::vector<int> dimensions;

  void parse_yaml(YAML::Node);
};

struct substream {
  virtual ~substream() {}
  virtual void initialize(YAML::Node) = 0;
  
  static std::shared_ptr<substream> from_yaml(YAML::Node);

  virtual int locate_timestep_file_index(int);
  virtual void read(int, std::shared_ptr<ndarray_group>) = 0;

  // yaml properties
  bool is_static = false;
  std::string name;
  std::string pattern; // file name pattern

  // files and timesteps
  std::vector<std::string> filenames;
  std::vector<int> timesteps_per_file, first_timestep_per_file;
  int total_timesteps = 0;
  int current_file_index = 0;

  // variables
  std::vector<variable> variables;

  // communicator
  MPI_Comm comm = MPI_COMM_WORLD;
};

struct substream_binary : public substream {
  void initialize(YAML::Node) {}
  
  void read(int, std::shared_ptr<ndarray_group>) {}
};

struct substream_netcdf : public substream {
  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);

  bool has_unlimited_time_dimension = false;
};

struct stream {
  std::shared_ptr<ndarray_group> advance_timestep();
  std::shared_ptr<ndarray_group> get_static_group();

  void parse_yaml(const std::string filename);

public:
  std::shared_ptr<ndarray_group> static_group;
  
  std::vector<substream> substreams;
};

///////////
inline void stream::parse_yaml(const std::string filename)
{
  YAML::Node yaml = YAML::LoadFile(filename);
  auto yroot = yaml["stream"];

  if (yroot["substreams"]) { // has substreams
    auto ysubstreams = yroot["substreams"];
    for (auto i = 0; i < ysubstreams.size(); i ++) {
      // fprintf(stderr, "substream %d\n", i);
      
      auto ysubstream = ysubstreams[i];
      auto sub = substream::from_yaml(ysubstream);
    }
  }
}

inline void variable::parse_yaml(YAML::Node y)
{
  this->name = y["name"].as<std::string>();
  this->possible_names.insert( this->name );

  if (auto ypvar = y["possible_names"]) {
    for (auto j = 0; j < ypvar.size(); j ++)
      this->possible_names.insert(ypvar[j].as<std::string>());
  }

  if (auto ydims = y["dimensions"]) {
    if (ydims.IsScalar()) { // auto
      this->is_dims_auto = true;
    } else if (ydims.IsSequence()) {
      for (auto i = 0; i < ydims.size(); i ++) 
        this->dimensions.push_back( ydims[i].as<int>() );
    }
    // TODO: otherwise throw an exception
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

inline std::shared_ptr<substream> substream::from_yaml(YAML::Node y)
{
  std::shared_ptr<substream> sub;

  if (auto yformat = y["format"]) {
    std::string format = yformat.as<std::string>();
    if (format == "binary")
      sub.reset(new substream_binary);
    if (format == "netcdf")
      sub.reset(new substream_netcdf);
    else
      throw NDARRAY_ERR_STREAM_FORMAT;
  }
  
  if (auto yvars = y["vars"]) { // has variable list
    for (auto i = 0; i < yvars.size(); i ++) {
      variable var;
      var.parse_yaml(yvars[i]);

      sub->variables.push_back(var);
    }
  }

#if 0
  for (auto i = 0; i < sub->variables.size(); i ++) {
    fprintf(stderr, "var=%s\n", sub->variables[i].name.c_str());
    for (const auto varname : sub->variables[i].possible_names)
      fprintf(stderr, "--name=%s\n", varname.c_str());
  }
#endif

  if (auto yname = y["name"])
    sub->name = yname.as<std::string>();

  if (auto ypattern = y["pattern"])
    sub->pattern = ypattern.as<std::string>();

  if (auto ystatic = y["static"])
    sub->is_static = ystatic.as<bool>();

  // std::cerr << y << std::endl;
  sub->initialize(y);
  return sub;
}

inline void substream_netcdf::read(int i, std::shared_ptr<ndarray_group> g)
{
  int fi = this->locate_timestep_file_index(i);
  if (fi < -1) 
    return;

  const std::string f = filenames[fi];

#if NDARRAY_HAVE_NETCDF
  int ncid, rtn;
#if NC_HAS_PARALLEL
  rtn = nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
  if (rtn != NC_NOERR)
    NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#else
  NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#endif

  for (const auto &var : variables) {
    int varid = -1;

    for (const auto varname : var.possible_names) {
      int rtn = nc_inq_varid(ncid, var.name.c_str(), &varid);
      if (rtn == NC_NOERR) 
        break;
    }

    if (varid >= 0) { // succ
      int type;
      NC_SAFE_CALL( nc_inq_vartype(ncid, varid, &type) );

      std::shared_ptr<ndarray_base> p = ndarray_base::new_by_nc_datatype(type);
      p->try_read_netcdf(ncid, var.possible_names, comm);

      g->set(var.name, p);

#if 0
      int ndims, dimids[4];
      size_t dimlens[4];

      NC_SAFE_CALL( nc_inq_varndims(ncid, varids, &ndims) );
      NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids) );
      for (int i = 0; i < ndims; i ++) 
        NC_SAFE_CALL( nc_inq_dimlen(ncid, dimids[i], &dimlens[i]) );
#endif

      // std::shared_ptr<ndarray_base> arr(new ndarray_base);
    } else { // failed

    }
  }

  NC_SAFE_CALL( nc_close(ncid) );

#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void substream_netcdf::initialize(YAML::Node y) 
{
  this->filenames = glob(pattern);

  fprintf(stderr, "substream %s, found %zu files.\n", 
      this->name.c_str(), filenames.size());

#if NDARRAY_HAVE_NETCDF
  for (const auto f : this->filenames) {
    int ncid, rtn;
#if NC_HAS_PARALLEL
    rtn = nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
    if (rtn != NC_NOERR)
      NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#else
    NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#endif

    size_t nt = 0;
    int unlimited_recid;
    nc_inq_unlimdim(ncid, &unlimited_recid);

    if (unlimited_recid >= 0) {
      has_unlimited_time_dimension = true;

      NC_SAFE_CALL( nc_inq_dimlen(ncid, unlimited_recid, &nt) );
    } else {
      has_unlimited_time_dimension = false;
    }

    NC_SAFE_CALL( nc_close(ncid) );
    
    timesteps_per_file.push_back(nt);
    first_timestep_per_file.push_back( this->total_timesteps );
    this->total_timesteps += nt;
    fprintf(stderr, "filename=%s, nt=%zu\n", f.c_str(), nt);
  }
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

}

#endif
