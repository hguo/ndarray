#ifndef _NDARRAY_NDARRAY_BASE_HH
#define _NDARRAY_NDARRAY_BASE_HH

#include <ndarray/config.hh>
#include <ndarray/error.hh>
#include <ndarray/device.hh>
#include <ndarray/lattice.hh>
#include <ndarray/util.hh>
#include <vector>
#include <array>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <random>

#if NDARRAY_HAVE_VTK
#include <vtkSmartPointer.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLPUnstructuredGridReader.h>
#include <vtkDataReader.h>
#include <vtkNew.h>
#endif

#if NDARRAY_HAVE_NETCDF
#include <netcdf.h>
#include <netcdf_meta.h>
#if NC_HAS_PARALLEL
#include <netcdf_par.h>
#endif
#endif

#if NDARRAY_HAVE_HDF5
#include <hdf5.h>
#endif

#if NDARRAY_HAVE_ADIOS2
#include <adios2.h>
#endif

#if NDARRAY_HAVE_ADIOS1
#include <adios.h>
#include <adios_read.h>
#include <adios_error.h>
#endif

namespace ftk {

enum {
  NDARRAY_ENDIAN_LITTLE = 0,
  NDARRAY_ENDIAN_BIG = 1
};

enum {
  NDARRAY_ORDER_C = 0,
  NDARRAY_ORDER_F = 1
};

enum {
  NDARRAY_DTYPE_UNKNOWN,
  NDARRAY_DTYPE_CHAR,
  NDARRAY_DTYPE_INT,
  NDARRAY_DTYPE_FLOAT,
  NDARRAY_DTYPE_DOUBLE,
  NDARRAY_DTYPE_UNSIGNED_CHAR,
  NDARRAY_DTYPE_UNSIGNED_INT
};

enum {
  NDARRAY_ADIOS2_STEPS_UNSPECIFIED = -1, 
  NDARRAY_ADIOS2_STEPS_ALL = -2
};

template <typename T> struct ndarray;

// the non-template base class for ndarray
struct ndarray_base {
  virtual ~ndarray_base() {}

  static std::string dtype2str(int dtype);
  static int str2dtype(const std::string str);

  static std::shared_ptr<ndarray_base> new_by_dtype(int type);
  static std::shared_ptr<ndarray_base> new_by_dtype(const std::string str) { return new_by_dtype(str2dtype(str)); }
  static std::shared_ptr<ndarray_base> new_by_nc_dtype(int typep);
  static std::shared_ptr<ndarray_base> new_by_vtk_dtype(int typep);
  static std::shared_ptr<ndarray_base> new_by_adios2_dtype(const std::string type);
#if NDARRAY_HAVE_HDF5
  static std::shared_ptr<ndarray_base> new_by_h5_dtype(hid_t native_type);
#endif

  virtual int type() const = 0;

  virtual size_t size() const = 0;
  virtual bool empty() const = 0;

  size_t nd() const {return dims.size();}
  
  size_t dimf(size_t i) const {return dims[i];}
  size_t shapef(size_t i) const {return dimf(i);}
  const std::vector<size_t> &shapef() const {return dims;}
  
  std::ostream& print_shapef(std::ostream& os) const;

  size_t dimc(size_t i) const {return dims[dims.size() - i -1];}
  size_t shapec(size_t i) const {return dimc(i);}
  const std::vector<size_t> shapec() const {std::vector<size_t> dc(dims); std::reverse(dc.begin(), dc.end()); return dc;}
  
  [[deprecated]] size_t dim(size_t i) const { return dimf(i); }
  [[deprecated]] size_t shape(size_t i) const {return dimf(i);}
  [[deprecated]] const std::vector<size_t> &shape() const {return shapef();}

  size_t nelem() const { if (empty()) return 0; else return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()); }
  virtual size_t elem_size() const = 0;
 
  virtual const void* pdata() const = 0;
  virtual void* pdata() = 0;

  virtual void flip_byte_order() = 0;

  virtual void reshapef(const std::vector<size_t> &dims_) = 0;
  void reshapef(const std::vector<int>& dims);
  void reshapef(size_t ndims, const size_t sizes[]);
  
  void reshapec(const std::vector<size_t> &dims_);
  void reshapec(const std::vector<int>& dims);
  void reshapec(size_t ndims, const size_t sizes[]);
  
  [[deprecated]] virtual void reshape(const std::vector<size_t> &dims_) { reshapef(dims_); }
  [[deprecated]] void reshape(const std::vector<int>& dims) { reshapef(dims); }
  [[deprecated]] void reshape(size_t ndims, const size_t sizes[]) { reshapef(ndims, sizes); }

  void reshape(const ndarray_base& array); //! copy shape from another array
  // template <typename T> void reshape(const ndarray<T>& array); //! copy shape from another array

public:
  size_t indexf(const std::vector<size_t>& idx) const;
  size_t indexf(const std::vector<int>& idx) const;
  size_t indexf(const size_t idx[]) const;
  
  template <typename uint=size_t>
  std::vector<uint> from_indexf(uint i) const {return lattice().from_integer(i);}

  lattice get_lattice() const;

public:
  size_t indexc(const std::vector<size_t>& idx) const;
  size_t indexc(const std::vector<int>& idx) const;
  size_t indexc(const size_t idx[]) const;
  
  template <typename uint=size_t>
  std::vector<uint> from_indexc(uint i) const; // FIXME {return lattice().from_integer(i);}

// public: // accessor
//   virtual double ac(size_t i0) const = 0;
//   virtual double ac(size_t i0, size_t i1) const = 0;
//   virtual double ac(size_t i0, size_t i1, size_t i2) const = 0;
//   virtual double ac(size_t i0, size_t i1, size_t i2, size_t i3) const = 0;
//   virtual double ac(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const = 0;
//   virtual double ac(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const = 0;
//   virtual double ac(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const = 0;

//   virtual double af(size_t i0) const = 0;
//   virtual double af(size_t i0, size_t i1) const = 0;
//   virtual double af(size_t i0, size_t i1, size_t i2) const = 0;
//   virtual double af(size_t i0, size_t i1, size_t i2, size_t i3) const = 0;
//   virtual double af(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const = 0;
//   virtual double af(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const = 0;
//   virtual double af(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const = 0;

public:  
  void set_multicomponents(size_t c=1) {ncd = c;}
  void make_multicomponents(); // make a non-multicomponent array to an array with 1 component
  size_t multicomponents() const {return ncd;}
  size_t ncomponents() const;

  void set_has_time(bool b) { tv = b; }
  bool has_time() const { return tv; }

public: // binary i/o
  void read_binary_file(const std::string& filename, int endian = NDARRAY_ENDIAN_LITTLE);
  virtual void read_binary_file(FILE *fp, int endian = NDARRAY_ENDIAN_LITTLE) = 0;
  void to_binary_file(const std::string& filename);
  virtual void to_binary_file(FILE *fp) = 0;

public: // netcdf
  void read_netcdf(const std::string& filename, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, int ndims, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(const std::string& filename, const std::string& varname, MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, const std::string& varname, MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, MPI_Comm comm=MPI_COMM_WORLD);

  void read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm=MPI_COMM_WORLD); // assuming the variable has an unlimited dimension
  
  // void to_netcdf(int ncid, const std::string& varname);
  // void to_netcdf(int ncid, int varid);
  void to_netcdf(int ncid, int varid, const size_t starts[], const size_t sizes[]) const;
  void to_netcdf(int ncid, int varid) const;
  void to_netcdf_multivariate(int ncid, int varids[]) const;
  void to_netcdf_unlimited_time(int ncid, int varid) const;
  void to_netcdf_multivariate_unlimited_time(int ncid, int varids[]) const;

  template <typename ContainerType> // std::vector<std::string>
  static int probe_netcdf_varid(int ncid, const ContainerType& possible_varnames, MPI_Comm comm);

  template <typename ContainerType> // std::vector<std::string>
  bool try_read_netcdf(int ncid, const ContainerType& possible_varnames, const size_t st[], const size_t sz[], MPI_Comm comm = MPI_COMM_WORLD);

  template <typename ContainerType> // std::vector<std::string>
  bool try_read_netcdf(int ncid, const ContainerType& possible_varnames, MPI_Comm comm = MPI_COMM_WORLD);

  virtual int nc_dtype() const = 0;

public: // h5 i/o
  bool read_h5(const std::string& filename, const std::string& name);
#if NDARRAY_HAVE_HDF5
  bool read_h5(hid_t fid, const std::string& name);
  virtual bool read_h5_did(hid_t did) = 0;
#endif

public: // adios2 i/o
  virtual void read_bp(
      const std::string& filename, 
      const std::string& varname, 
      int step = NDARRAY_ADIOS2_STEPS_UNSPECIFIED, 
      MPI_Comm comm = MPI_COMM_WORLD) = 0;

#if NDARRAY_HAVE_ADIOS2
  virtual void read_bp(
      adios2::IO &io, 
      adios2::Engine& reader, 
      const std::string &varname, 
      int step = NDARRAY_ADIOS2_STEPS_UNSPECIFIED) = 0; // read all

  static std::shared_ptr<ndarray_base> new_from_bp(
      adios2::IO &io,
      adios2::Engine &reader,
      const std::string &varname,
      int step = NDARRAY_ADIOS2_STEPS_UNSPECIFIED);
#endif

public: // adios1 io
  virtual bool read_bp_legacy(const std::string& filename, const std::string& varname, MPI_Comm comm) = 0;

public: // vti i/o
  void read_vtk_image_data_file(const std::string& filename, const std::string array_name=std::string());
  virtual void read_vtk_image_data_file_sequence(const std::string& pattern) = 0;
#if NDARRAY_HAVE_VTK
  virtual void from_vtk_image_data(vtkSmartPointer<vtkImageData> d, const std::string array_name=std::string()) = 0;
  virtual void from_vtu(vtkSmartPointer<vtkUnstructuredGrid> d, const std::string array_name=std::string()) = 0;
  virtual void from_vtk_data_array(vtkSmartPointer<vtkDataArray> da) = 0;
#endif

public: // vtk data array
#if NDARRAY_HAVE_VTK
  static std::shared_ptr<ndarray_base> new_from_vtk_data_array(vtkSmartPointer<vtkDataArray> da);
  static std::shared_ptr<ndarray_base> new_from_vtk_image_data(vtkSmartPointer<vtkImageData> da, const std::string varname);
  vtkSmartPointer<vtkDataArray> to_vtk_data_array(std::string varname=std::string()) const; 
  virtual int vtk_data_type() const = 0;
#endif

public: // device/host
  virtual void to_device(int device, int id=0) = 0;
  virtual void to_host() = 0;

protected:
  std::vector<size_t> dims, s;
  size_t ncd = 0; // number of dimensions for components.  For 3D vector field, nd=4, ncd=1.  For 3D jacobian field, nd=5, ncd=2
  bool tv = false; // wheter the last dimension is time
};

////////
inline lattice ndarray_base::get_lattice() const {
  std::vector<size_t> st(nd(), 0), sz(dims);
  return lattice(st, sz);
}

inline size_t ndarray_base::ncomponents() const {
  size_t rtn = 1;
  for (size_t i = 0; i < multicomponents(); i ++)
    rtn *= dims[i];
  return rtn;
}

inline void ndarray_base::reshapec(const std::vector<size_t>& dims_)
{
  std::vector<size_t> dims(dims_);
  std::reverse(dims.begin(), dims.end());
  reshapef(dims);
}

inline void ndarray_base::reshapec(const std::vector<int>& dims)
{
  std::vector<size_t> mydimsc;
  for (int i = 0; i < dims.size(); i ++)
    mydimsc.push_back(dims[i]);
  reshapec(mydimsc);
}

inline void ndarray_base::reshapec(size_t ndims, const size_t dims[])
{
  std::vector<size_t> mydimsc(dims, dims+ndims);
  reshapec(mydimsc);
}

inline void ndarray_base::reshapef(const std::vector<int>& dims)
{
  std::vector<size_t> mydims;
  for (int i = 0; i < dims.size(); i ++)
    mydims.push_back(dims[i]);
  reshapef(mydims);
}

inline void ndarray_base::reshapef(size_t ndims, const size_t dims[])
{
  std::vector<size_t> mydims(dims, dims+ndims);
  reshapef(mydims);
}

inline void ndarray_base::reshape(const ndarray_base& array)
{
  reshapef(array.shapef());
}

inline size_t ndarray_base::indexf(const size_t idx[]) const {
  size_t i(idx[0]);
  for (size_t j = 1; j < nd(); j ++)
    i += idx[j] * s[j];
  return i;
}

inline size_t ndarray_base::indexf(const std::vector<size_t>& idx) const {
  size_t i(idx[0]);
  for (size_t j = 1; j < nd(); j ++)
    i += idx[j] * s[j];
  return i;
}

inline size_t ndarray_base::indexf(const std::vector<int>& idx) const {
  std::vector<size_t> myidx(idx.begin(), idx.end());
  return indexf(myidx);
}

inline void ndarray_base::make_multicomponents()
{
  std::vector<size_t> s = shapef();
  s.insert(s.begin(), 1);
  reshapef(s);
  set_multicomponents();
}

inline void ndarray_base::read_binary_file(const std::string& filename, int endian)
{
  FILE *fp = fopen(filename.c_str(), "rb");
  read_binary_file(fp, endian);
  fclose(fp);
}

inline void ndarray_base::to_binary_file(const std::string& f)
{
  FILE *fp = fopen(f.c_str(), "wb");
  to_binary_file(fp);
  fclose(fp);
}

inline void ndarray_base::read_vtk_image_data_file(const std::string& filename, const std::string array_name)
{
#if NDARRAY_HAVE_VTK
  vtkNew<vtkXMLImageDataReader> reader;
  reader->SetFileName(filename.c_str());
  reader->Update();
  from_vtk_image_data(reader->GetOutput(), array_name);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_VTK);
#endif
}
  
#if NDARRAY_HAVE_VTK
inline std::shared_ptr<ndarray_base> ndarray_base::new_from_vtk_image_data(
    vtkSmartPointer<vtkImageData> vti,
    std::string varname)
{
  if (!vti)
    fatal("the input vtkImageData is null");

  vtkSmartPointer<vtkDataArray> arr = vti->GetPointData()->GetArray(varname.c_str());

  auto p = new_by_vtk_dtype( arr->GetDataType());
  p->from_vtk_image_data(vti, varname);

  return p;
}

inline std::shared_ptr<ndarray_base> ndarray_base::new_from_vtk_data_array(vtkSmartPointer<vtkDataArray> da)
{
  if (!da)
    fatal("the input vtkDataArray is null");

  auto p = new_by_vtk_dtype( da->GetDataType() );
  p->from_vtk_data_array(da);

  return p;
}
#endif

inline bool ndarray_base::read_h5(const std::string& filename, const std::string& name)
{
#if NDARRAY_HAVE_HDF5
  auto fid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid < 0) return false; else {
    bool succ = read_h5(fid, name);
    H5Fclose(fid);
    return succ;
  }
#else 
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_HDF5);
  return false;
#endif
}

#if NDARRAY_HAVE_HDF5
inline bool ndarray_base::read_h5(hid_t fid, const std::string& name)
{
  auto did = H5Dopen2(fid, name.c_str(), H5P_DEFAULT);
  if (did < 0) return false; else {
    bool succ = read_h5_did(did);
    H5Dclose(did);
    return succ;
  }
}
#endif

#if NDARRAY_HAVE_ADIOS2
inline std::shared_ptr<ndarray_base> ndarray_base::new_from_bp(
      adios2::IO &io,
      adios2::Engine &reader,
      const std::string &varname,
      int step)
{
  std::shared_ptr<ndarray_base> p = 
    new_by_adios2_dtype( io.VariableType(varname) );

  p->read_bp(io, reader, varname, step);

  return p;
}
#endif

inline void ndarray_base::read_bp(const std::string& filename, const std::string& varname, int step, MPI_Comm comm)
{
#if NDARRAY_HAVE_ADIOS2
#if ADIOS2_USE_MPI
  adios2::ADIOS adios(comm);
#else
  adios2::ADIOS adios;
#endif
  adios2::IO io = adios.DeclareIO("BPReader");
  adios2::Engine reader = io.Open(filename, adios2::Mode::Read); // , MPI_COMM_SELF);
  
  read_bp(io, reader, varname, step);
  reader.Close();
  
  // empty array; try legacy reader
  if (empty()) {
#if NDARRAY_HAVE_ADIOS1
    read_bp_legacy(filename, varname, comm);
#else
    throw NDARRAY_ERR_ADIOS2;
#endif
  }
  
  // if (empty()) read_bp_legacy(filename, varname, comm); 
#else
  warn(NDARRAY_ERR_NOT_BUILT_WITH_ADIOS2);
  read_bp_legacy(filename, varname, comm);
#endif
}

#if NDARRAY_HAVE_VTK
inline vtkSmartPointer<vtkDataArray> ndarray_base::to_vtk_data_array(std::string varname) const
{
  vtkSmartPointer<vtkDataArray> d = vtkDataArray::CreateDataArray(this->vtk_data_type());
  if (varname.length() > 0)
    d->SetName( varname.c_str() );

  // fprintf(stderr, "to_vtk_data_array, ncd=%zu\n", ncd);

  if (ncd == 1) {
    d->SetNumberOfComponents(shapef(0));
    d->SetNumberOfTuples( std::accumulate(dims.begin()+1, dims.end(), 1, std::multiplies<size_t>()) );
  }
  else if (ncd == 0) {
    d->SetNumberOfComponents(1);
    d->SetNumberOfTuples(nelem());
  } else {
    fatal(NDARRAY_ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS);
  }
  memcpy(d->GetVoidPointer(0), this->pdata(), elem_size() * nelem()); // nelem());
  return d;
}
#endif

inline void ndarray_base::read_netcdf(const std::string& filename, const std::string& varname, MPI_Comm comm)
{
#if NDARRAY_HAVE_NETCDF
  int ncid, varid;
#if NC_HAS_PARALLEL
  int rtn = nc_open_par(filename.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
  if (rtn != NC_NOERR)
    NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#else
  NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#endif

  {
    int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);
    if (rtn == NC_ENOTVAR)
      throw NDARRAY_ERR_NETCDF_MISSING_VARIABLE;
  }

  // NC_SAFE_CALL( nc_inq_varid(ncid, varname.c_str(), &varid) );
  
  read_netcdf(ncid, varid, comm);
  NC_SAFE_CALL( nc_close(ncid) );
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, int varid, int ndims, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#if NDARRAY_HAVE_NETCDF
  std::vector<size_t> mysizes(sizes, sizes+ndims);
  std::reverse(mysizes.begin(), mysizes.end());
  reshapef(mysizes);

  if (nc_dtype() == NC_INT) {
    NC_SAFE_CALL( nc_get_vara_int(ncid, varid, starts, sizes, (int*)pdata()) );
  } else if (nc_dtype() == NC_FLOAT) {
    NC_SAFE_CALL( nc_get_vara_float(ncid, varid, starts, sizes, (float*)pdata()) );
  } else if (nc_dtype() == NC_DOUBLE) {
    NC_SAFE_CALL( nc_get_vara_double(ncid, varid, starts, sizes, (double*)pdata()) );
  } else if (nc_dtype() == NC_UINT) {
    NC_SAFE_CALL( nc_get_vara_uint(ncid, varid, starts, sizes, (unsigned int*)pdata()) );
  } else if (nc_dtype() == NC_CHAR) {
    NC_SAFE_CALL( nc_get_vara_text(ncid, varid, starts, sizes, (char*)pdata()) );
  } else
    fatal(NDARRAY_ERR_NOT_IMPLEMENTED);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::to_netcdf(int ncid, int varid) const
{
  std::vector<size_t> starts(dims.size(), 0), sizes(dims);
  std::reverse(sizes.begin(), sizes.end());

  to_netcdf(ncid, varid, &starts[0], &sizes[0]);
}

inline void ndarray_base::to_netcdf(int ncid, int varid, const size_t st[], const size_t sz[]) const
{
#ifdef NDARRAY_HAVE_NETCDF
  fprintf(stderr, "st=%zu, %zu, %zu, %zu, sz=%zu, %zu, %zu, %zu\n", 
      st[0], st[1], st[2], st[3], sz[0], sz[1], sz[2], sz[3]);
  
  if (nc_dtype() == NC_DOUBLE) {
    NC_SAFE_CALL( nc_put_vara_double(ncid, varid, st, sz, (double*)pdata()) );
  } else if (nc_dtype() == NC_FLOAT) {
    NC_SAFE_CALL( nc_put_vara_float(ncid, varid, st, sz, (float*)pdata()) );
  } else if (nc_dtype() == NC_INT) {
    NC_SAFE_CALL( nc_put_vara_int(ncid, varid, st, sz, (int*)pdata()) );
  } else 
    fatal(NDARRAY_ERR_NOT_IMPLEMENTED);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

#if 0
inline void ndarray_base::to_netcdf_multivariate(int ncid, int varids[]) const
{
  const size_t nv = dims[0], ndims = nd()-1;
  std::vector<size_t> d(dims.begin()+1, dims.end());

  for (int i = 0; i < nv; i ++) {
    ndarray<T> subarray(d);
    for (size_t j = 0; j < subarray.nelem(); j ++) 
      subarray[j] = p[j*nv + i];
    subarray.to_netcdf(ncid, varids[i]);
  }
}
#endif

inline void ndarray_base::to_netcdf_unlimited_time(int ncid, int varid) const
{
  std::vector<size_t> starts(dims.size()+1, 0), sizes(dims);
  sizes.push_back(1);
  std::reverse(sizes.begin(), sizes.end());
 
  // fprintf(stderr, "starts={%zu, %zu, %zu}, sizes={%zu, %zu, %zu}\n", 
  //     starts[0], starts[1], starts[2], sizes[0], sizes[1], sizes[2]);

  to_netcdf(ncid, varid, &starts[0], &sizes[0]);
}

#if 0
inline void ndarray_base::to_netcdf_multivariate_unlimited_time(int ncid, int varids[]) const
{
  const size_t nv = dims[0], ndims = nd()-1;
  std::vector<size_t> d(dims.begin()+1, dims.end());

  for (int i = 0; i < nv; i ++) {
    ndarray<T> subarray(d);
    for (size_t j = 0; j < subarray.nelem(); j ++) 
      subarray[j] = p[j*nv + i];
    subarray.to_netcdf_unlimited_time(ncid, varids[i]);
  }
}
#endif

inline void ndarray_base::read_netcdf(int ncid, int varid, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ndims;
  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );

  std::vector<size_t> mysizes(sizes, sizes+ndims);
  std::reverse(mysizes.begin(), mysizes.end());
  reshapef(mysizes);

  read_netcdf(ncid, varid, ndims, starts, sizes, comm);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ndims;
  int dimids[4];
  size_t st[4] = {0}, sz[4] = {0};

  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );
  NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids) );

  for (int i = 0; i < ndims; i ++)
    NC_SAFE_CALL( nc_inq_dimlen(ncid, dimids[i], &sz[i]) );

  st[0] = t;
  sz[0] = 1;
 
  read_netcdf(ncid, varid, st, sz, comm);
  set_has_time(true);

#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, int varid, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ndims;
  int dimids[4];
  size_t starts[4] = {0}, sizes[4] = {0};

  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );
  NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids) );

  for (int i = 0; i < ndims; i ++)
    NC_SAFE_CALL( nc_inq_dimlen(ncid, dimids[i], &sizes[i]) );
  
  read_netcdf(ncid, varid, starts, sizes, comm);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, const std::string& varname, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int varid;
  const int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);
    
  if (rtn == NC_EBADID)
    throw NDARRAY_ERR_NETCDF_FILE_NOT_OPEN;
  else if (rtn == NC_ENOTVAR)
    throw NDARRAY_ERR_NETCDF_MISSING_VARIABLE;
  else // no error; variable found
    read_netcdf(ncid, varid, comm);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int varid;
  NC_SAFE_CALL( nc_inq_varid(ncid, varname.c_str(), &varid) );
  read_netcdf(ncid, varid, starts, sizes, comm);
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

template <typename ContainerType> // std::vector<std::string>
inline int ndarray_base::probe_netcdf_varid(
    int ncid, 
    const ContainerType& possible_varnames,
    MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int varid = -1;

  for (const auto varname : possible_varnames) {
    const int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);

    if (rtn == NC_EBADID)
      return false; // throw NDARRAY_ERR_NETCDF_FILE_NOT_OPEN;
    else if (rtn == NC_ENOTVAR)
      continue;
    else // no error; variable found
      break;
  }

  return varid;
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
  return -1;
#endif
}

template <typename ContainerType> // std::vector<std::string>
inline bool ndarray_base::try_read_netcdf(int ncid, 
    const ContainerType& possible_varnames, 
    MPI_Comm comm)
{
  int varid = probe_netcdf_varid(ncid, possible_varnames, comm);
  
  if (varid >= 0) {
    read_netcdf(ncid, varid, comm);
    return true;
  } else 
    return false;
}
  
template <typename ContainerType> // std::vector<std::string>
inline bool ndarray_base::try_read_netcdf(int ncid, 
    const ContainerType& possible_varnames, 
    const size_t st[], 
    const size_t sz[], 
    MPI_Comm comm)
{
  int varid = probe_netcdf_varid(ncid, possible_varnames, comm);
  
  if (varid >= 0) {
    read_netcdf(ncid, varid, st, sz, comm);
    return true;
  } else 
    return false;
}

inline void ndarray_base::read_netcdf(const std::string& filename, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ncid, varid;
#if NC_HAS_PARALLEL
  int rtn = nc_open_par(filename.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
  if (rtn != NC_NOERR)
    NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#else
  NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#endif

  NC_SAFE_CALL( nc_inq_varid(ncid, varname.c_str(), &varid) );
  read_netcdf(ncid, varid, starts, sizes, comm);
  NC_SAFE_CALL( nc_close(ncid) );
#else
  fatal(NDARRAY_ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

std::ostream& ndarray_base::print_shapef(std::ostream& os) const
{
  os << "nd=" << nd() << ", array_dims={";
  for (size_t i = 0; i < dims.size(); i ++) 
    if (i < dims.size()-1) os << dims[i] << ", ";
    else os << dims[i] << "}, ";
  
  os << "size=" << this->size() << ", "
     << "multicomponents=" << this->ncd << ", "
     << "time_varying=" << this->tv;

#if 0
  os << "prod={";
  for (size_t i = 0; i < s.size(); i ++) 
    if (i < s.size()-1) os << s[i] << ", ";
    else os << s[i] << "}, ";
  
#endif

  return os;
}

inline int ndarray_base::str2dtype(const std::string str)
{
  if (str == "char")
    return NDARRAY_DTYPE_CHAR;
  else if (str == "uchar")
    return NDARRAY_DTYPE_UNSIGNED_CHAR;
  else if (str == "int" || str == "int32")
    return NDARRAY_DTYPE_INT;
  else if (str == "uint" || str == "uint32")
    return NDARRAY_DTYPE_UNSIGNED_INT;
  else if (str == "float" || str == "float32")
    return NDARRAY_DTYPE_FLOAT;
  else if (str == "double" || str == "float64")
    return NDARRAY_DTYPE_DOUBLE;
  else 
    return NDARRAY_DTYPE_UNKNOWN;
}

} // namespace ndarray

#endif
