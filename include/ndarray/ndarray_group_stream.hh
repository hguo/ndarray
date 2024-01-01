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
  std::vector<std::string> possible_names; // will prioritize possible_names by ordering

  bool is_optional = false; // will be ignored if the format is binary
  
  bool is_dims_auto = true;
  std::vector<int> dimensions;
  unsigned char order = NDARRAY_ORDER_C;

  bool is_dtype_auto = true;
  int dtype = NDARRAY_DTYPE_UNKNOWN;

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

struct stream {
  stream(MPI_Comm comm = MPI_COMM_WORLD);
  ~stream() {};

  std::shared_ptr<ndarray_group> read(int);
  std::shared_ptr<ndarray_group> read_static();

  void parse_yaml(const std::string filename);
  int total_timesteps() const;
  
  void new_substream_from_yaml(YAML::Node);

public:
  std::vector<std::shared_ptr<substream>> substreams;
  bool has_adios2_substream = false;

  std::string path_prefix;

  MPI_Comm comm;

#if NDARRAY_HAVE_ADIOS2
  adios2::ADIOS adios;
  adios2::IO io;
#endif
};

struct substream {
  substream(stream& s) : stream_(s) {}
  virtual ~substream() {}
  virtual void initialize(YAML::Node) = 0;

  virtual int locate_timestep_file_index(int);
  virtual void read(int, std::shared_ptr<ndarray_group>) = 0;

  void glob();

  // yaml properties
  bool is_static = false;
  std::string name;
  std::string pattern; // file name pattern

  // files and timesteps
  std::vector<std::string> filenames;
  std::vector<int> timesteps_per_file, first_timestep_per_file;
  int total_timesteps = 0;
  int current_file_index = 0;

  // reference to the parent stream
  stream &stream_;

  // variables
  std::vector<variable> variables;

  // communicator
  MPI_Comm comm = MPI_COMM_WORLD;
};

struct substream_binary : public substream {
  substream_binary(stream& s) : substream(s) {}
  void initialize(YAML::Node);
  
  void read(int, std::shared_ptr<ndarray_group>);
};

struct substream_netcdf : public substream {
  substream_netcdf(stream& s) : substream(s) {}
  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);

  bool has_unlimited_time_dimension = false;
};

struct substream_h5 : public substream {
  substream_h5(stream& s) : substream(s) {}
  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);
  
  bool has_unlimited_time_dimension = false;
};

struct substream_adios2 : public substream {
  substream_adios2(stream& s) : substream(s) {}
  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);
};

struct substream_vti : public substream {
  substream_vti(stream& s) : substream(s) {}
  void initialize(YAML::Node);
  void read(int, std::shared_ptr<ndarray_group>);
};

struct substream_vti_o : public substream {
  substream_vti_o(stream& s) : substream(s) {}
};


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
    this->path_prefix = yprefix.as<std::string>();
  }

  if (auto ysubstreams = yroot["substreams"]) { // has substreams
    for (auto i = 0; i < ysubstreams.size(); i ++) {
      // fprintf(stderr, "substream %d\n", i);
      
      auto ysubstream = ysubstreams[i];
    
      std::string format = ysubstream["format"].as<std::string>();
      if (format == "adios2" && !has_adios2_substream) {
        // here's where adios2 is initialized
        has_adios2_substream = true;
#if NDARRAY_HAVE_ADIOS2
        io = adios.DeclareIO("BPReader");
#endif
      }

      new_substream_from_yaml(ysubstream);
    }
  }
}

inline std::shared_ptr<ndarray_group> stream::read_static()
{
  std::shared_ptr<ndarray_group> g(new ndarray_group);
  
  for (auto sub : this->substreams)
    if (sub->is_static)
      sub->read(0, g);

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
    if (!sub->is_static)
      sub->read(i, g);

  return g;
}

inline void stream::new_substream_from_yaml(YAML::Node y)
{
  std::shared_ptr<substream> sub;

  if (auto yformat = y["format"]) {
    std::string format = yformat.as<std::string>();
    if (format == "binary")
      sub.reset(new substream_binary(*this));
    else if (format == "netcdf")
      sub.reset(new substream_netcdf(*this));
    else if (format == "h5")
      sub.reset(new substream_h5(*this));
    else if (format == "adios2")
      sub.reset(new substream_adios2(*this));
    else if (format == "vti")
      sub.reset(new substream_vti(*this));
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

  substreams.push_back(sub);
}


///////////
inline void variable::parse_yaml(YAML::Node y)
{
  this->name = y["name"].as<std::string>();

  if (auto ypvar = y["possible_names"]) {
    for (auto j = 0; j < ypvar.size(); j ++)
      this->possible_names.push_back(ypvar[j].as<std::string>());
  } else 
    this->possible_names.push_back( this->name );

  if (auto ydtype = y["dtype"]) {
    this->dtype = ndarray_base::str2dtype( ydtype.as<std::string>() );
    if (this->dtype != NDARRAY_DTYPE_UNKNOWN)
      this->is_dtype_auto = false;
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
      throw NDARRAY_ERR_STREAM_FORMAT;
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


///////////
inline void substream::glob()
{
  const auto pattern_ = stream_.path_prefix + pattern;
  this->filenames = ::ndarray::glob(pattern_);
  fprintf(stderr, "substream %s, pattern=%s, found %zu files.\n", 
      this->name.c_str(), pattern_.c_str(), filenames.size());
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


///////////
inline void substream_binary::initialize(YAML::Node y)
{
  glob();
  this->total_timesteps = filenames.size();
}

inline void substream_binary::read(int i, std::shared_ptr<ndarray_group> g)
{
  const auto f = filenames[i]; // assume each vti has only one timestep
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

///////////
inline void substream_vti::initialize(YAML::Node y) 
{
  glob();
  this->total_timesteps = filenames.size();
}

inline void substream_vti::read(int i, std::shared_ptr<ndarray_group> g)
{
  const auto f = filenames[i]; // assume each vti has only one timestep

#if NDARRAY_HAVE_VTK
  vtkSmartPointer<vtkXMLImageDataReader> reader = vtkXMLImageDataReader::New();
  reader->SetFileName(f.c_str());
  reader->Update();
  vtkSmartPointer<vtkImageData> vti = reader->GetOutput();

  for (const auto &var : variables) {
    auto array = vti->GetPointData()->GetArray( var.name.c_str() );
    // array->PrintSelf(std::cerr, vtkIndent(4));

    std::shared_ptr<ndarray_base> p = ndarray_base::new_from_vtk_data_array(array);
    g->set(var.name, p);
  }

#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_VTK);
#endif
}

///////////
inline void substream_adios2::initialize(YAML::Node y)
{
  glob();

  if (!is_static)
    this->total_timesteps = this->filenames.size();
}

inline void substream_adios2::read(int i, std::shared_ptr<ndarray_group> g)
{
  const auto f = this->filenames[i];
#if NDARRAY_HAVE_ADIOS2
  adios2::Engine reader = this->stream_.io.Open(f, adios2::Mode::Read);
  auto available_variables = stream_.io.AvailableVariables(true);

  for (const auto &var : variables) {
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
        fatal("cannot find variable " + var.name);
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

#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_ADIOS2);
#endif
}

///////////
inline void substream_h5::initialize(YAML::Node y) 
{
  glob();

  if (!is_static)
    this->total_timesteps = this->filenames.size();
}

inline void substream_h5::read(int i, std::shared_ptr<ndarray_group> g)
{
#if NDARRAY_HAVE_HDF5
  auto fid = H5Fopen( this->filenames[i].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
  if (fid >= 0) {
    for (const auto &var : variables) {
      
      // probe variable name
      hid_t did = H5I_INVALID_HID;
      for (const auto varname : var.possible_names) {
        did = H5Dopen2(fid, varname.c_str(), H5P_DEFAULT);
        if (did != H5I_INVALID_HID)
          break;
      }

      if (did == H5I_INVALID_HID) {
        if (var.is_optional)
          continue;
        else {
          fatal("cannot read variable " + var.name);
          return;
        }
      } else { 
        // create a new array
        // auto native_type = H5Tget_native_type( H5Dget_type(did), H5T_DIR_DEFAULT );
        auto native_type = H5Dget_type(did);
        auto p = ndarray_base::new_by_h5_dtype( native_type );

        // actual read
        p->read_h5_did(did);
        // p->print_shapef(std::cerr); std::cerr << std::endl;
        g->set(var.name, p);

        H5Dclose(did);
      }
    }
    H5Fclose(fid);
  }
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_HDF5);
#endif
}

///////////
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
      int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);
      // fprintf(stderr, "ncid=%d, varname=%s, possible_name=%s, varid=%d\n", 
      //     ncid, var.name.c_str(), varname.c_str(), varid);

      if (rtn == NC_NOERR) 
        break;
    }

    if (varid >= 0) { // succ
      // create a new array
      int type;
      NC_SAFE_CALL( nc_inq_vartype(ncid, varid, &type) );
      std::shared_ptr<ndarray_base> p = ndarray_base::new_by_nc_dtype(type);
   
      // check if the variable has unlimited dimension
      int unlimited_recid;
      NC_SAFE_CALL( nc_inq_unlimdim(ncid, &unlimited_recid) );
      
      int ndims, dimids[4];
      NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );
      NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids) );

      bool time_varying = false;
      if (unlimited_recid >= 0) // has unlimied dimension
        for (int i = 0; i < ndims; i ++)
          if (dimids[i] == unlimited_recid)
            time_varying = true;

      if (time_varying)
        p->read_netcdf_timestep(ncid, varid, i - first_timestep_per_file[fi], comm);
      else 
        p->read_netcdf(ncid, varid, comm);

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
  glob();

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
