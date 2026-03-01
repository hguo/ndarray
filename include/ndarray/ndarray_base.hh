#ifndef _NDARRAY_NDARRAY_BASE_HH
#define _NDARRAY_NDARRAY_BASE_HH

#include <ndarray/config.hh>
#include <ndarray/error.hh>
#include <ndarray/device.hh>
#include <ndarray/lattice.hh>
#include <ndarray/util.hh>
#include <ndarray/murmurhash2.hh>
#include <vector>
#include <array>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <random>

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

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
#if NDARRAY_HAVE_NETCDF_PARALLEL
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

// Forward declaration with storage policy template parameter
struct native_storage;  // Forward declare default storage policy
template <typename T, typename StoragePolicy = native_storage> struct ndarray;

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

  // NOTE: dims now stores C-order (last varies fastest) internally like mdspan
  // dimf/shapef return Fortran-order (user-facing), dimc/shapec return C-order

  size_t dimf(size_t i) const {return dims[dims.size() - i - 1];}  // Reverse index
  size_t shapef(size_t i) const {return dimf(i);}
  const std::vector<size_t> shapef() const {
    std::vector<size_t> f_dims(dims);
    std::reverse(f_dims.begin(), f_dims.end());
    return f_dims;
  }

  std::ostream& print_shapef(std::ostream& os) const;

  size_t dimc(size_t i) const {return dims[i];}  // Direct access (C-order)
  size_t shapec(size_t i) const {return dimc(i);}
  const std::vector<size_t>& shapec() const {return dims;}  // Reference (C-order)

  /**
   * @brief Convert C-order shape to Fortran-order (for I/O API boundaries)
   * @param c_shape Dimensions in C-order (last varies fastest)
   * @return Dimensions in Fortran-order (first varies fastest)
   */
  static std::vector<size_t> c_to_f_order(const std::vector<size_t>& c_shape) {
    std::vector<size_t> f_shape(c_shape);
    std::reverse(f_shape.begin(), f_shape.end());
    return f_shape;
  }

  /**
   * @brief Convert C-order shape array to Fortran-order
   */
  static std::vector<size_t> c_to_f_order(const size_t* c_shape, size_t ndims) {
    std::vector<size_t> f_shape(c_shape, c_shape + ndims);
    std::reverse(f_shape.begin(), f_shape.end());
    return f_shape;
  }

  /**
   * @brief Convert Fortran-order shape to C-order (alias for shapec for static use)
   * @param f_shape Dimensions in Fortran-order
   * @return Dimensions in C-order
   */
  static std::vector<size_t> f_to_c_order(const std::vector<size_t>& f_shape) {
    std::vector<size_t> c_shape(f_shape);
    std::reverse(c_shape.begin(), c_shape.end());
    return c_shape;
  }

  [[deprecated]] size_t dim(size_t i) const { return dimf(i); }
  [[deprecated]] size_t shape(size_t i) const {return dimf(i);}
  [[deprecated]] std::vector<size_t> shape() const {return shapef();}

  size_t nelem() const { if (dims.empty()) return 0; else return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()); }
  virtual size_t elem_size() const = 0;

  virtual const void* pdata() const = 0;
  virtual void* pdata() = 0;

  virtual void flip_byte_order() = 0;

  // Fortran-order reshape: first dimension varies fastest (column-major)
  // Use with f() for element access
  virtual void reshapef(const std::vector<size_t> &dims_) = 0;
  void reshapef(const std::vector<int>& dims);
  void reshapef(size_t ndims, const size_t sizes[]);

  // C-order reshape: last dimension varies fastest (row-major)
  // Dimensions are reversed internally, then stored as Fortran-order
  // Use with c() for element access - consistent with NumPy default (C-order)
  void reshapec(const std::vector<size_t> &dims_);
  void reshapec(const std::vector<int>& dims);
  void reshapec(size_t ndims, const size_t sizes[]);

  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  virtual void reshape(const std::vector<size_t> &dims_) { reshapef(dims_); }
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(const std::vector<int>& dims) { reshapef(dims); }
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
  // Multicomponent array support for vector/tensor fields
  //
  // Component dimensions are ALWAYS the FIRST dimensions in the array shape.
  // This places all components at a single spatial point contiguous in memory.
  //
  // Examples:
  //   Scalar 3D field:  reshapef(nx, ny, nz),       set_multicomponents(0)
  //   Vector 3D field:  reshapef(3, nx, ny, nz),    set_multicomponents(1)
  //   Tensor 2D field:  reshapef(3, 3, nx, ny),     set_multicomponents(2)
  //
  // Access with f(): component indices come FIRST
  //   scalar.f(x, y, z)          // Scalar field
  //   velocity.f(c, x, y, z)     // Vector field: c ∈ [0, ncomp)
  //   jacobian.f(i, j, x, y)     // Tensor field: i,j ∈ [0, 3)
  //
  // DISTRIBUTED ARRAYS (MPI):
  //   Component dimensions are NOT partitioned across ranks (replicated).
  //   Use decomp[i]=0 for component dimensions in decompose().
  //   Example: velocity.decompose(comm, {3,100,200}, 0, {0,4,2}, {0,1,1})
  //            → All 3 components on every rank, spatial dims partitioned 4×2

  // Set number of component dimensions (0=scalar, 1=vector, 2=tensor)
  // Must be called after reshapef() for multicomponent arrays
  void set_multicomponents(size_t c=1) {n_component_dims = c;}

  // Convert scalar array to vector array with 1 component: [nx, ny] → [1, nx, ny]
  // Useful for treating scalar fields uniformly with vector fields
  void make_multicomponents();

  // Get number of component dimensions
  // Returns: 0 (scalar), 1 (vector), or 2 (tensor)
  size_t multicomponents() const {return n_component_dims;}

  // Get total number of components (product of first n_component_dims dimensions)
  // Examples:
  //   shape [3, 100, 200] with multicomponents()=1 → ncomponents()=3
  //   shape [3, 3, 64, 64] with multicomponents()=2 → ncomponents()=9
  size_t ncomponents() const;

  // Mark whether the last dimension represents time
  // When is_time_varying=true, the array is time-series data: [...spatial_dims..., time_dim]
  //
  // DISTRIBUTED ARRAYS (MPI):
  //   Time dimension is NOT partitioned across ranks (replicated).
  //   Use decomp[last]=0 for time dimension in decompose().
  //   Example: temp.decompose(comm, {100,200,50}, 0, {4,2,0}, {1,1,0})
  //            → All 50 timesteps on every rank, spatial dims partitioned 4×2
  void set_has_time(bool b) { is_time_varying = b; }

  // Check if last dimension is time
  bool has_time() const { return is_time_varying; }

public: // binary i/o
  void read_binary_file(const std::string& filename, int endian = NDARRAY_ENDIAN_LITTLE);
  virtual void read_binary_file(FILE *fp, int endian = NDARRAY_ENDIAN_LITTLE) = 0;
  void to_binary_file(const std::string& filename);
  virtual void to_binary_file(FILE *fp) = 0;

  virtual void read_binary_auto(const std::string& filename) = 0;
  virtual void write_binary_auto(const std::string& filename) = 0;

public: // netcdf
  void read_netcdf(const std::string& filename, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, int ndims, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, const size_t starts[], const size_t sizes[], MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(const std::string& filename, const std::string& varname, MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, const std::string& varname, MPI_Comm comm=MPI_COMM_WORLD);
  void read_netcdf(int ncid, int varid, MPI_Comm comm=MPI_COMM_WORLD);

  // Read a single timestep from NetCDF variable with unlimited dimension (typically time)
  // Automatically sets has_time(true) after reading
  void read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm=MPI_COMM_WORLD);

  // void to_netcdf(int ncid, const std::string& varname);
  // void to_netcdf(int ncid, int varid);
  void to_netcdf(int ncid, int varid, const size_t starts[], const size_t sizes[]) const;
  void to_netcdf(int ncid, int varid) const;
  void to_netcdf_multivariate(int ncid, int varids[]) const;
  void to_netcdf_unlimited_time(int ncid, int varid) const;
  void to_netcdf_multivariate_unlimited_time(int ncid, int varids[]) const;

  template <typename ContainerType=std::vector<std::string>> // std::vector<std::string>
  static int probe_netcdf_varid(int ncid, const ContainerType& possible_varnames, MPI_Comm comm);

  template <typename ContainerType=std::vector<std::string>> // std::vector<std::string>
  bool try_read_netcdf(int ncid, const ContainerType& possible_varnames, const size_t st[], const size_t sz[], MPI_Comm comm = MPI_COMM_WORLD);

  template <typename ContainerType=std::vector<std::string>> // std::vector<std::string>
  bool try_read_netcdf(int ncid, const ContainerType& possible_varnames, MPI_Comm comm = MPI_COMM_WORLD);

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_NETCDF
  virtual void read_netcdf_auto(const std::string& filename, const std::string& varname) = 0;
  virtual void write_netcdf_auto(const std::string& filename, const std::string& varname) = 0;
#endif

  virtual int nc_dtype() const = 0;

public: // pnetcdf i/o
#if NDARRAY_HAVE_PNETCDF
  virtual void read_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz) = 0;
  virtual void write_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz) const = 0;
#if NDARRAY_HAVE_MPI
  virtual void read_pnetcdf_auto(const std::string& filename, const std::string& varname) = 0;
  virtual void write_pnetcdf_auto(const std::string& filename, const std::string& varname) = 0;
#endif
#endif

public: // h5 i/o
  bool read_h5(const std::string& filename, const std::string& name);
#if NDARRAY_HAVE_HDF5
  bool read_h5(hid_t fid, const std::string& name);
  virtual void read_h5_did(hid_t did) = 0;
#endif
#if NDARRAY_HAVE_HDF5
  virtual void read_hdf5_auto(const std::string& filename, const std::string& varname) = 0;
  virtual void write_hdf5_auto(const std::string& filename, const std::string& varname) = 0;
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

public: // decomposition
#if NDARRAY_HAVE_MPI
  virtual void decompose(MPI_Comm comm,
                         const std::vector<size_t>& global_dims,
                         size_t nprocs = 0,
                         const std::vector<size_t>& decomp = {},
                         const std::vector<size_t>& ghost = {}) = 0;
  virtual void set_replicated(MPI_Comm comm) = 0;
#endif

protected:
  std::vector<size_t> dims, s;

  // n_component_dims: number of leading dimensions that represent components (not spatial/temporal)
  // n_component_dims=0: scalar field, all dimensions are spatial (e.g., [nx, ny, nz])
  // n_component_dims=1: vector field, first dimension is components (e.g., [3, nx, ny, nz] for 3D velocity)
  // n_component_dims=2: tensor field, first two dimensions are components (e.g., [3, 3, nx, ny] for stress tensor)
  // Total components = product of first n_component_dims dimensions
  size_t n_component_dims = 0;

  // is_time_varying: whether the last dimension represents time
  // is_time_varying=false: static/spatial data only (e.g., [nx, ny, nz])
  // is_time_varying=true: time-series data, last dimension is timesteps (e.g., [nx, ny, nt])
  // Combined: [component_dims..., spatial_dims..., time_dim]
  //           <----- n_component_dims -----> <-- nd()-n_component_dims-1 --> <-- is_time_varying -->
  bool is_time_varying = false;
};

////////
inline lattice ndarray_base::get_lattice() const {
  std::vector<size_t> st(nd(), 0), sz(dims);
  return lattice(st, sz);
}

inline size_t ndarray_base::ncomponents() const {
  // Compute total number of components by multiplying the first n_component_dims dimensions
  // Examples:
  //   [3, 100, 200] with multicomponents()=1 → ncomponents()=3
  //   [3, 3, 64, 64] with multicomponents()=2 → ncomponents()=9
  //   [100, 200] with multicomponents()=0 → ncomponents()=1 (scalar)
  size_t rtn = 1;
  for (size_t i = 0; i < multicomponents(); i++)
    rtn *= dims[i];
  return rtn;
}

inline void ndarray_base::reshapec(const std::vector<size_t>& dims_)
{
  // dims_ is in C-order, convert to Fortran-order for reshapef
  // reshapef will then convert back to C-order for storage
  std::vector<size_t> f_dims(dims_);
  std::reverse(f_dims.begin(), f_dims.end());
  reshapef(f_dims);
}

inline void ndarray_base::reshapec(const std::vector<int>& dims)
{
  std::vector<size_t> mydimsc;
  for (size_t i = 0; i < dims.size(); i ++)
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
  for (size_t i = 0; i < dims.size(); i ++)
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

// Fortran-order indexing: first index varies fastest
// With C-order strides, reverse indices and use C-order formula
inline size_t ndarray_base::indexf(const size_t idx[]) const {
  std::vector<size_t> reversed_idx(nd());
  for (size_t j = 0; j < nd(); j++)
    reversed_idx[j] = idx[nd() - 1 - j];
  return indexc(reversed_idx.data());
}

inline size_t ndarray_base::indexf(const std::vector<size_t>& idx) const {
  std::vector<size_t> reversed_idx(idx.rbegin(), idx.rend());
  return indexc(reversed_idx);
}

inline size_t ndarray_base::indexf(const std::vector<int>& idx) const {
  std::vector<size_t> myidx(idx.begin(), idx.end());
  return indexf(myidx);
}

// C-order indexing: last index varies fastest
// With C-order strides, direct formula
inline size_t ndarray_base::indexc(const size_t idx[]) const {
  size_t i = 0;
  for (size_t j = 0; j < nd(); j++)
    i += idx[j] * s[j];
  return i;
}

inline size_t ndarray_base::indexc(const std::vector<size_t>& idx) const {
  size_t i = 0;
  for (size_t j = 0; j < nd(); j++)
    i += idx[j] * s[j];
  return i;
}

inline size_t ndarray_base::indexc(const std::vector<int>& idx) const {
  std::vector<size_t> myidx(idx.begin(), idx.end());
  return indexc(myidx);
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
  if (!fp) throw std::runtime_error("Cannot open file for reading: " + filename);
  read_binary_file(fp, endian);
  fclose(fp);
}

inline void ndarray_base::to_binary_file(const std::string& f)
{
  FILE *fp = fopen(f.c_str(), "wb");
  if (!fp) throw std::runtime_error("Cannot open file for writing: " + f);
  to_binary_file(fp);
  fflush(fp);
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
  throw feature_not_available(ERR_NOT_BUILT_WITH_VTK, "VTK support not enabled in this build");
#endif
}

#if NDARRAY_HAVE_VTK
inline std::shared_ptr<ndarray_base> ndarray_base::new_from_vtk_image_data(
    vtkSmartPointer<vtkImageData> vti,
    std::string varname)
{
  if (!vti)
    throw vtk_error("Input vtkImageData is null");

  vtkSmartPointer<vtkDataArray> arr = vti->GetPointData()->GetArray(varname.c_str());

  auto p = new_by_vtk_dtype( arr->GetDataType());
  p->from_vtk_image_data(vti, varname);

  return p;
}

inline std::shared_ptr<ndarray_base> ndarray_base::new_from_vtk_data_array(vtkSmartPointer<vtkDataArray> da)
{
  if (!da)
    throw vtk_error("Input vtkDataArray is null");

  auto p = new_by_vtk_dtype( da->GetDataType() );
  p->from_vtk_data_array(da);

  return p;
}
#endif

inline bool ndarray_base::read_h5(const std::string& filename, const std::string& name)
{
#if NDARRAY_HAVE_HDF5
  auto fid = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fid < 0) {
    throw hdf5_error(ERR_HDF5_IO, "Cannot open HDF5 file: " + filename);
  }

  try {
    read_h5(fid, name);
    H5Fclose(fid);
    return true;
  } catch (...) {
    H5Fclose(fid);
    throw;  // Re-throw the exception after cleanup
  }
#else
  fatal(ERR_NOT_BUILT_WITH_HDF5);
  return false;
#endif
}

#if NDARRAY_HAVE_HDF5
inline bool ndarray_base::read_h5(hid_t fid, const std::string& name)
{
  auto did = H5Dopen2(fid, name.c_str(), H5P_DEFAULT);
  if (did < 0) {
    throw hdf5_error(ERR_HDF5_IO, "Cannot open HDF5 dataset: " + name);
  }

  read_h5_did(did);
  H5Dclose(did);
  return true;
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
  adios2::Engine reader = io.Open(filename, adios2::Mode::ReadRandomAccess);

  read_bp(io, reader, varname, step);
  reader.Close();

  // empty array; try legacy reader
  if (empty()) {
#if NDARRAY_HAVE_ADIOS1
    read_bp_legacy(filename, varname, comm);
#else
    throw ERR_ADIOS2;
#endif
  }

  // if (empty()) read_bp_legacy(filename, varname, comm);
#else
  throw feature_not_available(ERR_NOT_BUILT_WITH_ADIOS2, "ADIOS2 support not enabled in this build");
#endif
}

#if NDARRAY_HAVE_VTK
inline vtkSmartPointer<vtkDataArray> ndarray_base::to_vtk_data_array(std::string varname) const
{
  vtkSmartPointer<vtkDataArray> d = vtkDataArray::CreateDataArray(this->vtk_data_type());
  if (varname.length() > 0)
    d->SetName( varname.c_str() );

  // fprintf(stderr, "to_vtk_data_array, n_component_dims=%zu\n", n_component_dims);

  if (n_component_dims == 1) {
    d->SetNumberOfComponents(shapef(0));
    d->SetNumberOfTuples( std::accumulate(dims.begin()+1, dims.end(), 1, std::multiplies<size_t>()) );
  }
  else if (n_component_dims == 0) {
    d->SetNumberOfComponents(1);
    d->SetNumberOfTuples(nelem());
  } else {
    throw vtk_error(ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS, "VTK output only supports one dimension for components (vectors/tensors with >1D components not supported)");
  }
  memcpy(d->GetVoidPointer(0), this->pdata(), elem_size() * nelem()); // nelem());
  return d;
}
#endif

inline void ndarray_base::read_netcdf(const std::string& filename, const std::string& varname, MPI_Comm comm)
{
#if NDARRAY_HAVE_NETCDF
  int ncid, varid;
#if NDARRAY_HAVE_NETCDF_PARALLEL
  int rtn = nc_open_par(filename.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
  if (rtn != NC_NOERR)
    NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#else
  NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid) );
#endif

  try {
    int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);
    if (rtn == NC_ENOTVAR)
      throw ERR_NETCDF_MISSING_VARIABLE;

    read_netcdf(ncid, varid, comm);
  } catch (...) {
    nc_close(ncid);
    throw;
  }
  NC_SAFE_CALL( nc_close(ncid) );
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, int varid, int ndims, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#if NDARRAY_HAVE_NETCDF
  // NetCDF uses C-order, ndarray now stores C-order internally - direct use!
  std::vector<size_t> ndarray_sizes(sizes, sizes + ndims);
  reshapec(ndarray_sizes);

  // No conversion needed - starts/sizes already in C-order
  std::vector<size_t> nc_starts(starts, starts + ndims);
  std::vector<size_t> nc_sizes(sizes, sizes + ndims);

  if (nc_dtype() == NC_INT) {
    NC_SAFE_CALL( nc_get_vara_int(ncid, varid, nc_starts.data(), nc_sizes.data(), (int*)pdata()) );
  } else if (nc_dtype() == NC_FLOAT) {
    NC_SAFE_CALL( nc_get_vara_float(ncid, varid, nc_starts.data(), nc_sizes.data(), (float*)pdata()) );

    // fill value
    nc_type vr_type;
    size_t vr_len;
    int rtn = nc_inq_att(ncid, varid, "_FillValue", &vr_type, &vr_len);
    float fillvalue;
    if (rtn == NC_NOERR) { // has fill value
      NC_SAFE_CALL( nc_get_att_float(ncid, varid, "_FillValue", &fillvalue) );

      float *data = (float*)pdata();
      for (size_t i = 0; i < nelem(); i ++)
        if (data[i] == fillvalue)
          data[i] = std::nan("");
    }

  } else if (nc_dtype() == NC_DOUBLE) {
    NC_SAFE_CALL( nc_get_vara_double(ncid, varid, nc_starts.data(), nc_sizes.data(), (double*)pdata()) );
  } else if (nc_dtype() == NC_UINT) {
    NC_SAFE_CALL( nc_get_vara_uint(ncid, varid, nc_starts.data(), nc_sizes.data(), (unsigned int*)pdata()) );
  } else if (nc_dtype() == NC_CHAR) {
    NC_SAFE_CALL( nc_get_vara_text(ncid, varid, nc_starts.data(), nc_sizes.data(), (char*)pdata()) );
  } else
    fatal(ERR_NOT_IMPLEMENTED);

#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::to_netcdf(int ncid, int varid) const
{
  // Pass ndarray's native dimension order; overload will handle NetCDF reversal
  std::vector<size_t> starts(dims.size(), 0);
  to_netcdf(ncid, varid, &starts[0], &dims[0]);
}

inline void ndarray_base::to_netcdf(int ncid, int varid, const size_t st[], const size_t sz[]) const
{
#ifdef NDARRAY_HAVE_NETCDF
  // Query NetCDF variable dimensionality to know array sizes
  int nc_ndims;
  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &nc_ndims) );

  // NetCDF uses C-order, ndarray now stores C-order - direct use!
  std::vector<size_t> nc_starts(st, st + nc_ndims);
  std::vector<size_t> nc_sizes(sz, sz + nc_ndims);

  if (nc_dtype() == NC_DOUBLE) {
    NC_SAFE_CALL( nc_put_vara_double(ncid, varid, nc_starts.data(), nc_sizes.data(), (double*)pdata()) );
  } else if (nc_dtype() == NC_FLOAT) {
    NC_SAFE_CALL( nc_put_vara_float(ncid, varid, nc_starts.data(), nc_sizes.data(), (float*)pdata()) );
  } else if (nc_dtype() == NC_INT) {
    NC_SAFE_CALL( nc_put_vara_int(ncid, varid, nc_starts.data(), nc_sizes.data(), (int*)pdata()) );
  } else
    fatal(ERR_NOT_IMPLEMENTED);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
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
  // Pass ndarray's native dimension order plus time dim; overload will handle NetCDF reversal
  std::vector<size_t> starts(dims.size()+1, 0);
  std::vector<size_t> sizes(dims);
  sizes.push_back(1);

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

  // Delegate to overload which handles dimension conversion and reshape
  read_netcdf(ncid, varid, ndims, starts, sizes, comm);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ndims;
  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );

  std::vector<int> dimids(ndims);
  std::vector<size_t> st(ndims, 0), sz(ndims, 0);

  NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids.data()) );

  for (int i = 0; i < ndims; i ++)
    NC_SAFE_CALL( nc_inq_dimlen(ncid, dimids[i], &sz[i]) );

  st[0] = t;
  sz[0] = 1;

  read_netcdf(ncid, varid, st.data(), sz.data(), comm);
  set_has_time(true);

#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, int varid, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int ndims;
  NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );

  std::vector<int> dimids(ndims);
  std::vector<size_t> starts(ndims, 0), sizes(ndims, 0);

  NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids.data()) );

  for (int i = 0; i < ndims; i ++)
    NC_SAFE_CALL( nc_inq_dimlen(ncid, dimids[i], &sizes[i]) );

  read_netcdf(ncid, varid, starts.data(), sizes.data(), comm);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, const std::string& varname, MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int varid;
  const int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);

  if (rtn == NC_EBADID)
    throw ERR_NETCDF_FILE_NOT_OPEN;
  else if (rtn == NC_ENOTVAR)
    throw ERR_NETCDF_MISSING_VARIABLE;
  else // no error; variable found
    read_netcdf(ncid, varid, comm);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline void ndarray_base::read_netcdf(int ncid, const std::string& varname, const size_t starts[], const size_t sizes[], MPI_Comm comm)
{
#ifdef NDARRAY_HAVE_NETCDF
  int varid;
  NC_SAFE_CALL( nc_inq_varid(ncid, varname.c_str(), &varid) );
  read_netcdf(ncid, varid, starts, sizes, comm);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
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
      return false; // throw ERR_NETCDF_FILE_NOT_OPEN;
    else if (rtn == NC_ENOTVAR)
      continue;
    else // no error; variable found
      break;
  }

  return varid;
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
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
#if NDARRAY_HAVE_NETCDF_PARALLEL
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
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif
}

inline std::ostream& ndarray_base::print_shapef(std::ostream& os) const
{
  os << "nd=" << nd() << ", array_dims={";
  for (size_t i = 0; i < dims.size(); i ++)
    if (i < dims.size()-1) os << dims[i] << ", ";
    else os << dims[i] << "}, ";

  os << "size=" << this->size() << ", "
     << "n_component_dims=" << this->n_component_dims << ", "
     << "is_time_varying=" << this->is_time_varying;

#if 0
  os << "prod={";
  for (size_t i = 0; i < s.size(); i ++)
    if (i < s.size()-1) os << s[i] << ", ";
    else os << s[i] << "}, ";

#endif

  return os;
}

inline std::string ndarray_base::dtype2str(int dtype)
{
  if (dtype == NDARRAY_DTYPE_CHAR)
    return "char";
  else if (dtype == NDARRAY_DTYPE_UNSIGNED_CHAR)
    return "uchar";
  else if (dtype == NDARRAY_DTYPE_INT)
    return "int32";
  else if (dtype == NDARRAY_DTYPE_UNSIGNED_INT)
    return "uint32";
  else if (dtype == NDARRAY_DTYPE_FLOAT)
    return "float32";
  else if (dtype == NDARRAY_DTYPE_DOUBLE)
    return "float64";
  else
    return "";
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

} // namespace ftk

#endif
