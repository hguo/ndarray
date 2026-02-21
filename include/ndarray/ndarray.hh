#ifndef _NDARRAY_NDARRAY_HH
#define _NDARRAY_NDARRAY_HH

#include <ndarray/config.hh>
#include <ndarray/ndarray_base.hh>
#include <ndarray/storage/storage_policy.hh>
#include <ndarray/storage/native_storage.hh>
#include <ndarray/device.hh>

#if NDARRAY_HAVE_XTENSOR
#include <ndarray/storage/xtensor_storage.hh>
#endif

#if NDARRAY_HAVE_EIGEN
#include <ndarray/storage/eigen_storage.hh>
#endif

#include <ndarray/lattice_partitioner.hh>

#if NDARRAY_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <ndarray/ndarray_cuda.hh>
#endif

#if NDARRAY_HAVE_SYCL
#include <CL/sycl.hpp>
#endif

#if NDARRAY_HAVE_MPI
#include <mpi.h>
// #include <ndarray/external/bil/bil.h>
#if NDARRAY_HAVE_CUDA
#include <ndarray/ndarray_mpi_gpu.hh>
#endif
#endif

#if NDARRAY_HAVE_PNETCDF
#include <pnetcdf.h>
#endif

#if NDARRAY_HAVE_PYBIND11
#include <pybind11/numpy.h>
#endif

#if NDARRAY_HAVE_PNG
#include <ndarray/ndarray_png.hh>
#endif

namespace ftk {

// Default argument already specified in forward declaration (ndarray_base.hh)
template <typename T, typename StoragePolicy>
struct ndarray : public ndarray_base {
  using storage_type = typename StoragePolicy::template container_type<T>;
  int type() const;

  ndarray() {}
  ndarray(const std::vector<size_t> &dims) {reshapef(dims);}
  ndarray(const std::vector<size_t> &dims, T val) {reshapef(dims, val);}

  // Construct 1D array from std::vector data (disabled when T is size_t to avoid ambiguity)
  template <typename U = T, typename std::enable_if<!std::is_same<U, size_t>::value, int>::type = 0>
  explicit ndarray(const std::vector<T> &data) {copy_vector(data);}

  [[deprecated]] ndarray(const lattice& l) {reshapef(l.sizes());}
  [[deprecated]] ndarray(const T *a, const std::vector<size_t> &shape);

  template <typename T1> ndarray(const ndarray<T1>& array1) { from_array<T1>(array1); }
  template <typename T1, typename OtherPolicy> ndarray(const ndarray<T1, OtherPolicy>& array1) { from_array<T1, OtherPolicy>(array1); }
  ndarray(const ndarray<T, StoragePolicy>& a) { dims = a.dims; s = a.s; n_component_dims = a.n_component_dims; is_time_varying = a.is_time_varying; storage_ = a.storage_; }

  template <typename T1> ndarray<T, StoragePolicy>& operator=(const ndarray<T1>& array1) { from_array<T1>(array1); return *this; }
  template <typename T1, typename OtherPolicy> ndarray<T, StoragePolicy>& operator=(const ndarray<T1, OtherPolicy>& array1) { from_array<T1, OtherPolicy>(array1); return *this; }
  ndarray<T, StoragePolicy>& operator=(const ndarray<T, StoragePolicy>& a) { dims = a.dims; s = a.s; n_component_dims = a.n_component_dims; is_time_varying = a.is_time_varying; storage_ = a.storage_; return *this; }

  std::ostream& print(std::ostream& os) const;

  size_t size() const {return storage_.size();}
  bool empty() const  {return storage_.size() == 0;}
  size_t elem_size() const { return sizeof(T); }

  void fill(T value); //! fill with a constant value
  void fill(const std::vector<T>& values); //! fill values with std::vector
  void fill(const std::vector<std::vector<T>>& values); //! fill values

  // Note: Only available for native_storage backend
  template <typename SP = StoragePolicy>
  typename std::enable_if<std::is_same<SP, native_storage>::value, const std::vector<T>&>::type
  std_vector() const {return storage_.data_;}

  const T* data() const {return storage_.data();}
  T* data() {return storage_.data();}

  void flip_byte_order(T&);
  void flip_byte_order();

  const void* pdata() const {return storage_.data();}
  void* pdata() {return storage_.data();}

  void swap(ndarray& x);

  template <typename T1> void reshape(const ndarray<T1>& array); //! copy shape from another array

  void reshapef(const std::vector<size_t> &dims_);
  void reshapef(const std::vector<size_t> &dims, T val);
  template <typename I> void reshapef(const int ndims, const I sz[]);

  void reshapef(size_t n0) {reshapef(std::vector<size_t>({n0}));}
  void reshapef(size_t n0, size_t n1) {reshapef({n0, n1});}
  void reshapef(size_t n0, size_t n1, size_t n2) {reshapef({n0, n1, n2});}
  void reshapef(size_t n0, size_t n1, size_t n2, size_t n3) {reshapef({n0, n1, n2, n3});}
  void reshapef(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) {reshapef({n0, n1, n2, n3, n4});}
  void reshapef(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5) {reshapef({n0, n1, n2, n3, n4, n5});}
  void reshapef(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6) {reshapef({n0, n1, n2, n3, n4, n5, n6});}

  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(const std::vector<size_t> &dims_) {reshapef(dims_);}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(const std::vector<size_t> &dims, T val) {reshapef(dims, val);}
  template <typename I> [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(const int ndims, const I sz[]) {reshapef(ndims, sz);}

  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0) {reshapef(std::vector<size_t>({n0}));}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1) {reshapef({n0, n1});}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1, size_t n2) {reshapef({n0, n1, n2});}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1, size_t n2, size_t n3) {reshapef({n0, n1, n2, n3});}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) {reshapef({n0, n1, n2, n3, n4});}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5) {reshapef({n0, n1, n2, n3, n4, n5});}
  [[deprecated("Use reshapef() for Fortran-order (first index varies fastest) or reshapec() for C-order/NumPy compatibility (last index varies fastest)")]]
  void reshape(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6) {reshapef({n0, n1, n2, n3, n4, n5, n6});}

  void reset() { storage_.resize(0); dims.clear(); s.clear(); set_multicomponents(0); set_has_time(false); }

  ndarray<T, StoragePolicy> slice(const lattice&) const;
  ndarray<T, StoragePolicy> slice(const std::vector<size_t>& starts, const std::vector<size_t> &sizes) const;

  // Extract a single timestep from time-series data (assumes last dim is time)
  // Returns (n-1)-dimensional array without time dimension
  // Example: [nx, ny, nt] → [nx, ny]
  ndarray<T, StoragePolicy> slice_time(size_t t) const;

  // Extract all timesteps as a vector of (n-1)-dimensional arrays
  // Example: [nx, ny, nt] → vector of nt arrays of shape [nx, ny]
  std::vector<ndarray<T, StoragePolicy>> slice_time() const;

  // merge multiple arrays into a multicomponent array
  static ndarray<T, StoragePolicy> concat(const std::vector<ndarray<T, StoragePolicy>>& arrays);
  static ndarray<T, StoragePolicy> stack(const std::vector<ndarray<T, StoragePolicy>>& arrays);

public: // Column-major (Fortran-style) access: f(i0, i1, ...) where i0 varies fastest
  // For a 2D array reshaped as (n0, n1):
  //   f(i0, i1) accesses element at memory location: i0 + i1*n0
  //   This matches Fortran's column-major convention where the first index varies fastest
  T& f(const std::vector<size_t>& idx) {return storage_[indexf(idx)];}
  const T& f(const std::vector<size_t>& idx) const {return storage_[indexf(idx)];}

  T& f(const size_t idx[]) {return storage_[indexf(idx)];}
  const T& f(const size_t idx[]) const {return storage_[indexf(idx)];}

  T& f(const std::vector<int>& idx) {return storage_[indexf(idx)];}
  const T& f(const std::vector<int>& idx) const {return storage_[indexf(idx)];}

  // Fortran-order: first index varies fastest
  // With C-order strides, delegate to c() with reversed indices
  T& f(size_t i0) {return storage_[i0];}
  T& f(size_t i0, size_t i1) {return c(i1, i0);}
  T& f(size_t i0, size_t i1, size_t i2) {return c(i2, i1, i0);}
  T& f(size_t i0, size_t i1, size_t i2, size_t i3) {return c(i3, i2, i1, i0);}
  T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {return c(i4, i3, i2, i1, i0);}
  T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) {return c(i5, i4, i3, i2, i1, i0);}
  T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) {return c(i6, i5, i4, i3, i2, i1, i0);}

  const T& f(size_t i0) const {return storage_[i0];}
  const T& f(size_t i0, size_t i1) const {return c(i1, i0);}
  const T& f(size_t i0, size_t i1, size_t i2) const {return c(i2, i1, i0);}
  const T& f(size_t i0, size_t i1, size_t i2, size_t i3) const {return c(i3, i2, i1, i0);}
  const T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const {return c(i4, i3, i2, i1, i0);}
  const T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const {return c(i5, i4, i3, i2, i1, i0);}
  const T& f(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const {return c(i6, i5, i4, i3, i2, i1, i0);}

public: // Row-major (C-style) access: c(i0, i1, ...) where the last index varies fastest
  // For a 2D array reshaped as (n0, n1):
  //   c(i0, i1) accesses element at memory location: i1 + i0*n1
  //   This matches C's row-major convention where the last index varies fastest
  //
  // IMPORTANT: c() is consistent with NumPy's default C-order behavior
  // Use reshapec() with c() for NumPy compatibility
  //
  // Note: Both f() and c() access the same underlying storage but with different indexing schemes
  T& c(const std::vector<size_t>& idx) {return storage_[indexc(idx)];}
  const T& c(const std::vector<size_t>& idx) const {return storage_[indexc(idx)];}

  T& c(const size_t idx[]) {return storage_[indexc(idx)];}
  const T& c(const size_t idx[]) const {return storage_[indexc(idx)];}

  T& c(const std::vector<int>& idx) {return storage_[indexc(idx)];}
  const T& c(const std::vector<int>& idx) const {return storage_[indexc(idx)];}

  // C-order: last index varies fastest
  // With C-order strides, this is direct indexing
  T& c(size_t i0) {return storage_[i0];}
  T& c(size_t i0, size_t i1) {return storage_[i0*s[0]+i1*s[1]];}
  T& c(size_t i0, size_t i1, size_t i2) {return storage_[i0*s[0]+i1*s[1]+i2*s[2]];}
  T& c(size_t i0, size_t i1, size_t i2, size_t i3) {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]];}
  T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]];}
  T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]];}
  T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]+i6*s[6]];}

  const T& c(size_t i0) const {return storage_[i0];}
  const T& c(size_t i0, size_t i1) const {return storage_[i0*s[0]+i1*s[1]];}
  const T& c(size_t i0, size_t i1, size_t i2) const {return storage_[i0*s[0]+i1*s[1]+i2*s[2]];}
  const T& c(size_t i0, size_t i1, size_t i2, size_t i3) const {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]];}
  const T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]];}
  const T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]];}
  const T& c(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const {return storage_[i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]+i6*s[6]];}


public:
  friend std::ostream& operator<<(std::ostream& os, const ndarray<T, StoragePolicy>& arr) {arr.print(os); return os;}
  friend bool operator==(const ndarray<T, StoragePolicy>& lhs, const ndarray<T, StoragePolicy>& rhs) {
    if (lhs.dims != rhs.dims) return false;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs.storage_[i] != rhs.storage_[i]) return false;
    }
    return true;
  }

  ndarray<T, StoragePolicy>& operator+=(const ndarray<T, StoragePolicy>& x);
  ndarray<T, StoragePolicy>& operator-=(const ndarray<T, StoragePolicy>& x);

  template <typename T1> ndarray<T, StoragePolicy>& operator*=(const T1& x);
  template <typename T1> ndarray<T, StoragePolicy>& operator/=(const T1& x);

  template <typename T1, typename SP1> friend ndarray<T1, SP1> operator+(const ndarray<T1, SP1>& lhs, const ndarray<T1, SP1>& rhs);
  template <typename T1, typename SP1> friend ndarray<T1, SP1> operator-(const ndarray<T1, SP1>& lhs, const ndarray<T1, SP1>& rhs);

  // template <typename T1> friend ndarray<T> operator*(const ndarray<T>& lhs, const T1& rhs);
  template <typename T1> friend ndarray<T, StoragePolicy> operator*(const T1& lhs, const ndarray<T, StoragePolicy>& rhs) {return rhs * lhs;}
  template <typename T1> friend ndarray<T, StoragePolicy> operator/(const ndarray<T, StoragePolicy>& lhs, const T1& rhs);

public: // element access
  T& operator[](size_t i) {return storage_[i];}
  const T& operator[](size_t i) const {return storage_[i];}

public: // legacy compatibility (deprecated)
  // WARNING: at() uses FORTRAN-ORDER (first index varies fastest, column-major)
  //          at(i,j,k) is equivalent to f(i,j,k), NOT c(i,j,k)
  //          Use f() for Fortran-order or c() for NumPy/C-order compatibility
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(const std::vector<size_t>& idx) {return storage_[indexf(idx)];}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(const std::vector<size_t>& idx) const {return storage_[indexf(idx)];}

  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(const size_t idx[]) {return storage_[indexf(idx)];}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(const size_t idx[]) const {return storage_[indexf(idx)];}

  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(const std::vector<int>& idx) {return storage_[indexf(idx)];}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(const std::vector<int>& idx) const {return storage_[indexf(idx)];}

  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0) {return storage_[i0];}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1) {return f(i0, i1);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1, size_t i2) {return f(i0, i1, i2);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1, size_t i2, size_t i3) {return f(i0, i1, i2, i3);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {return f(i0, i1, i2, i3, i4);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) {return f(i0, i1, i2, i3, i4, i5);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) {return f(i0, i1, i2, i3, i4, i5, i6);}

  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0) const {return storage_[i0];}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1) const {return f(i0, i1);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1, size_t i2) const {return f(i0, i1, i2);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1, size_t i2, size_t i3) const {return f(i0, i1, i2, i3);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const {return f(i0, i1, i2, i3, i4);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const {return f(i0, i1, i2, i3, i4, i5);}
  [[deprecated("Use f() for Fortran-order (first index varies fastest) or c() for C-order/NumPy compatibility (last index varies fastest)")]]
  const T& at(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const {return f(i0, i1, i2, i3, i4, i5, i6);}

  [[deprecated]] double get(size_t i0) const { return at(i0); }
  [[deprecated]] double get(size_t i0, size_t i1) const { return at(i0, i1); }
  [[deprecated]] double get(size_t i0, size_t i1, size_t i2) const { return at(i0, i1, i2); }
  [[deprecated]] double get(size_t i0, size_t i1, size_t i2, size_t i3) const { return at(i0, i1, i2, i3); }
  [[deprecated]] double get(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const { return at(i0, i1, i2, i3, i4); }
  [[deprecated]] double get(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const { return at(i0, i1, i2, i3, i4, i5); }
  [[deprecated]] double get(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const { return at(i0, i1, i2, i3, i4, i5, i6); }

  [[deprecated]] T& operator()(const std::vector<size_t>& idx) {return at(idx);}
  [[deprecated]] T& operator()(const std::vector<int>& idx) {return at(idx);}
  [[deprecated]] T& operator()(size_t i0) {return storage_[i0];}
  [[deprecated]] T& operator()(size_t i0, size_t i1) {return storage_[i0+i1*s[1]];}
  [[deprecated]] T& operator()(size_t i0, size_t i1, size_t i2) {return storage_[i0+i1*s[1]+i2*s[2]];}
  [[deprecated]] T& operator()(size_t i0, size_t i1, size_t i2, size_t i3) {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]];}
  [[deprecated]] T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]];}
  [[deprecated]] T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]];}
  [[deprecated]] T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]+i6*s[6]];}

  [[deprecated]] T& operator()(const std::vector<size_t>& idx) const {return at(idx);}
  [[deprecated]] T& operator()(const std::vector<int>& idx) const {return at(idx);}
  [[deprecated]] const T& operator()(size_t i0) const {return storage_[i0];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1) const {return storage_[i0+i1*s[1]];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1, size_t i2) const {return storage_[i0+i1*s[1]+i2*s[2]];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1, size_t i2, size_t i3) const {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) const {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) const {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]];}
  [[deprecated]] const T& operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6) const {return storage_[i0+i1*s[1]+i2*s[2]+i3*s[3]+i4*s[4]+i5*s[5]+i6*s[6]];}


  template <typename F=float> // scalar multilinear interpolation
  bool mlerp(const F x[], T v[]) const;

  ndarray<T, StoragePolicy>& transpose(); // returns the ref to this
  ndarray<T, StoragePolicy> get_transpose() const; // only works for 2D arrays
  ndarray<T, StoragePolicy> get_transpose(const std::vector<size_t> order) const; // works for general tensors

  template <typename T1>
  void from_array(const ndarray<T1>& array1);

  template <typename T1, typename OtherPolicy>
  void from_array(const ndarray<T1, OtherPolicy>& array1);

  void from_array(const T* p, const std::vector<size_t>& shape);

  void from_vector(const std::vector<T> &array);
  void copy_vector(const std::vector<T> &array);

  template <typename Iterator>
  void copy(Iterator first, Iterator last);

public: // subarray
  ndarray<T, StoragePolicy> subarray(const lattice&) const;

public: // construction from data
  // Create 1D array from std::vector data
  static ndarray<T, StoragePolicy> from_vector_data(const std::vector<T>& data) {
    ndarray<T, StoragePolicy> arr;
    arr.copy_vector(data);
    return arr;
  }

  // Create N-D array from std::vector data with specified shape
  static ndarray<T, StoragePolicy> from_vector_data(const std::vector<T>& data, const std::vector<size_t>& shape) {
    ndarray<T, StoragePolicy> arr;
    arr.reshapef(shape);
    size_t n = std::min(data.size(), arr.size());
    for (size_t i = 0; i < n; i++) {
      arr[i] = data[i];
    }
    return arr;
  }

public: // file i/o; automatically determine format based on extensions
  static ndarray<T, StoragePolicy> from_file(const std::string& filename, const std::string varname="", MPI_Comm comm = MPI_COMM_WORLD);
  bool read_file(const std::string& filename, const std::string varname="", MPI_Comm comm = MPI_COMM_WORLD);
  bool to_file(const std::string& filename, const std::string varname="", MPI_Comm comm = MPI_COMM_WORLD) const;

public: // i/o for binary file
  void read_binary_file(const std::string& filename, int endian = NDARRAY_ENDIAN_LITTLE) { ndarray_base::read_binary_file(filename, endian); }
  void read_binary_file(FILE *fp, int endian = NDARRAY_ENDIAN_LITTLE);
  void read_binary_file_sequence(const std::string& pattern, int endian = NDARRAY_ENDIAN_LITTLE);
  void to_vector(std::vector<T> &out_vector) const;
  void to_binary_file(const std::string& filename) { ndarray_base::to_binary_file(filename); }
  void to_binary_file(FILE *fp);

  template <typename T1> void to_binary_file2(const std::string& f) const;

  void from_bov(const std::string& filename);
  void to_bov(const std::string& filename) const;

  void bil_add_block_raw(const std::string& filename, const std::vector<size_t>& SZ, const lattice& ext);

public: // i/o for vtk image data
  void to_vtk_image_data_file(const std::string& filename, const std::string varname=std::string()) const;
  void read_vtk_image_data_file_sequence(const std::string& pattern);
#if NDARRAY_HAVE_VTK
  int vtk_data_type() const;
  void from_vtu(vtkSmartPointer<vtkUnstructuredGrid> d, const std::string array_name=std::string());
  void from_vtk_image_data(vtkSmartPointer<vtkImageData> d, const std::string array_name=std::string()) { from_vtk_regular_data<>(d, array_name); }
  void from_vtk_array(vtkSmartPointer<vtkAbstractArray> d);
  void from_vtk_data_array(vtkSmartPointer<vtkDataArray> d);
  vtkSmartPointer<vtkImageData> to_vtk_image_data(std::string varname=std::string()) const;
  // vtkSmartPointer<vtkDataArray> to_vtk_data_array(std::string varname=std::string()) const;  // moved to base

  template <typename VTK_REGULAR_DATA=vtkImageData> /*vtkImageData, vtkRectilinearGrid, or vtkStructuredGrid*/
  void from_vtk_regular_data(vtkSmartPointer<VTK_REGULAR_DATA> d, const std::string array_name=std::string());
#endif

public: // i/o for vtkStructuredGrid data
  void to_vtk_rectilinear_grid(const std::string& filename, const std::string varname=std::string()) const;

public: // i/o for hdf5
  static ndarray<T, StoragePolicy> from_h5(const std::string& filename, const std::string& name);
#if NDARRAY_HAVE_HDF5
  void read_h5_did(hid_t did);
  void to_h5(const std::string& filename, const std::string& varname) const;
  static hid_t h5_mem_type_id();
#endif

public: // i/o for parallel-netcdf
#if NDARRAY_HAVE_PNETCDF
  void read_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz);
  void write_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz) const;
#endif

public: // i/o for adios2
  static ndarray<T, StoragePolicy> from_bp(const std::string& filename, const std::string& name, int step = -1, MPI_Comm comm = MPI_COMM_WORLD);
  void read_bp(const std::string& filename, const std::string& varname, int step = -1, MPI_Comm comm = MPI_COMM_WORLD) { this->ndarray_base::read_bp(filename, varname, step, comm); }

#if NDARRAY_HAVE_ADIOS2
  void read_bp(
      adios2::IO &io,
      adios2::Engine& reader,
      const std::string &varname,
      int step = -1); // read all

  void read_bp(
      adios2::IO &io,
      adios2::Engine& reader,
      adios2::Variable<T>& var,
      int step = -1);
#endif

public: // i/o for adios1
  static ndarray<T, StoragePolicy> from_bp_legacy(const std::string& filename, const std::string& varname, MPI_Comm comm);
  bool read_bp_legacy(const std::string& filename, const std::string& varname, MPI_Comm comm);
#if NDARRAY_HAVE_ADIOS1
  bool read_bp_legacy(ADIOS_FILE *fp, const std::string& varname);
#endif

public: // i/o for png
  void read_png(const std::string& filename);
  void to_png(const std::string& filename) const;

public: // i/o for amira, see https://www.csc.kth.se/~weinkauf/notes/amiramesh.html
  bool read_amira(const std::string& filename); // T must be floatt

public: // pybind11
#if NDARRAY_HAVE_PYBIND11
  ndarray(const pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> &numpy_array);
  void from_numpy(const pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> &numpy_array);
  pybind11::array_t<T, pybind11::array::c_style> to_numpy() const;
#endif
  void read_numpy(const std::string& filename);
  void to_numpy(const std::string& filename) const;

#if NDARRAY_HAVE_MPI
  static MPI_Datatype mpi_dtype();
#endif

  int nc_dtype() const;

public: // device management
  void to_device(int device, int id=0);
  void to_host();
  void copy_to_device(int device, int id=0);   // copy to device, keep host data
  void copy_from_device();                     // copy from device to host, keep device data

  bool is_on_device() const { return device_type != NDARRAY_DEVICE_HOST; }
  bool is_on_host() const { return device_type == NDARRAY_DEVICE_HOST; }
  int get_device_type() const { return device_type; }
  int get_device_id() const { return device_id; }

  void *get_devptr() { return devptr_.get(); }
  const void *get_devptr() const { return devptr_.get(); }

#if NDARRAY_HAVE_SYCL
  void set_sycl_queue(sycl::queue* q) { sycl_queue_ptr = q; }
  sycl::queue* get_sycl_queue() { return sycl_queue_ptr; }
#endif

public: // statistics & misc
  std::tuple<T, T> min_max() const;
  T maxabs() const;
  T resolution() const; // the min abs nonzero value

  ndarray<uint64_t> quantize() const; // quantization based on resolution

  ndarray<T, StoragePolicy> &perturb(T sigma); // add gaussian noise to the array
  ndarray<T, StoragePolicy> &clamp(T min, T max); // clamp data with min and max
  
  ndarray<T, StoragePolicy> &scale(T factor); // multiply all elements by factor
  ndarray<T, StoragePolicy> &add(const ndarray<T, StoragePolicy>& other); // element-wise addition

#if NDARRAY_HAVE_PNETCDF
  int pnc_dtype() const;
#endif

public: // Distribution-aware I/O (automatically chooses parallel/replicated/serial)
#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_NETCDF
  /**
   * @brief Read from NetCDF with automatic parallel/replicated/serial detection
   *
   * Behavior:
   * - Distributed: Parallel read (each rank reads local portion)
   * - Replicated: Rank 0 reads + MPI_Bcast
   * - Serial: Regular serial read
   */
  void read_netcdf_auto(const std::string& filename, const std::string& varname);
  void write_netcdf_auto(const std::string& filename, const std::string& varname);
#endif

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_PNETCDF
  void read_pnetcdf_auto(const std::string& filename, const std::string& varname);
  void write_pnetcdf_auto(const std::string& filename, const std::string& varname);
#endif

#if NDARRAY_HAVE_HDF5
  void read_hdf5_auto(const std::string& filename, const std::string& varname);
  void write_hdf5_auto(const std::string& filename, const std::string& varname);
#endif

  void read_binary_auto(const std::string& filename);
  void write_binary_auto(const std::string& filename);

public: // MPI and distributed memory support
#if NDARRAY_HAVE_MPI
  /**
   * @brief Configure array for distributed execution (domain decomposition)
   *
   * After calling decompose(), the array behaves as domain-decomposed:
   * - dims/shape return local dimensions (this rank's portion + ghosts)
   * - operator() and at() use local indices
   * - read/write methods do parallel I/O automatically
   * - exchange_ghosts() updates ghost layers via MPI
   *
   * IMPORTANT: Only spatial dimensions are decomposed. Component and time dimensions
   *            should use decomp[i]=0 to replicate them on all ranks.
   *
   * @param comm MPI communicator
   * @param global_dims Global array dimensions
   * @param nprocs Number of processes (0 = use comm size)
   * @param decomp Decomposition pattern (empty = auto, or specify per-dimension):
   *               - decomp[i] > 0: Split dimension i into decomp[i] pieces (spatial dims)
   *               - decomp[i] == 0: DON'T split dimension i - replicate on all ranks (components, time)
   *               - If empty, auto-decompose all dimensions
   *
   *               Example 1: velocity[3,1000,800,600] with decomp={0,4,2,1}
   *                         → All 3 components on every rank (not partitioned)
   *                         → Spatial dims partitioned 4×2×1 = 8 ranks
   *
   *               Example 2: temp_time[100,200,50] with decomp={4,2,0}
   *                         → Spatial dims partitioned 4×2 = 8 ranks
   *                         → All 50 timesteps on every rank (not partitioned)
   *
   * @param ghost Ghost layers per dimension (0 for non-decomposed dimensions)
   */
  void decompose(MPI_Comm comm,
                 const std::vector<size_t>& global_dims,
                 size_t nprocs = 0,
                 const std::vector<size_t>& decomp = {},
                 const std::vector<size_t>& ghost = {});

  /**
   * @brief Mark array as replicated across all ranks
   *
   * All ranks have full data, no domain decomposition.
   * I/O uses rank 0 read/write + MPI_Bcast for efficiency.
   *
   * @param comm MPI communicator
   */
  void set_replicated(MPI_Comm comm);

  /**
   * @brief Exchange ghost layers with neighboring ranks
   *
   * No-op if array is not distributed.
   */
  void exchange_ghosts();

  /**
   * @brief Check if array is distributed (domain-decomposed)
   */
  bool is_distributed() const;

  /**
   * @brief Check if array is replicated (full data on all ranks)
   */
  bool is_replicated() const;

  /**
   * @brief Check if array has MPI configuration
   */
  bool has_mpi_config() const { return dist_ != nullptr; }

  // Distribution-specific accessors (throw if not distributed)
  const lattice& global_lattice() const;
  const lattice& local_core() const;
  const lattice& local_extent() const;
  ndarray<T, StoragePolicy>& local_array() { return *this; }
  const ndarray<T, StoragePolicy>& local_array() const { return *this; }

  // Index conversion (throw if not distributed)
  std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const;
  std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const;
  bool is_local(const std::vector<size_t>& global_idx) const;

  // Global index access (convenience methods - throw if not in local core)
  // Fortran-order (column-major)
  T& at_global(size_t i0);
  T& at_global(size_t i0, size_t i1);
  T& at_global(size_t i0, size_t i1, size_t i2);
  T& at_global(size_t i0, size_t i1, size_t i2, size_t i3);
  const T& at_global(size_t i0) const;
  const T& at_global(size_t i0, size_t i1) const;
  const T& at_global(size_t i0, size_t i1, size_t i2) const;
  const T& at_global(size_t i0, size_t i1, size_t i2, size_t i3) const;

  // Explicit Fortran-order
  T& f_global(size_t i0);
  T& f_global(size_t i0, size_t i1);
  T& f_global(size_t i0, size_t i1, size_t i2);
  T& f_global(size_t i0, size_t i1, size_t i2, size_t i3);
  const T& f_global(size_t i0) const;
  const T& f_global(size_t i0, size_t i1) const;
  const T& f_global(size_t i0, size_t i1, size_t i2) const;
  const T& f_global(size_t i0, size_t i1, size_t i2, size_t i3) const;

  // C-order (row-major)
  T& c_global(size_t i0);
  T& c_global(size_t i0, size_t i1);
  T& c_global(size_t i0, size_t i1, size_t i2);
  T& c_global(size_t i0, size_t i1, size_t i2, size_t i3);
  const T& c_global(size_t i0) const;
  const T& c_global(size_t i0, size_t i1) const;
  const T& c_global(size_t i0, size_t i1, size_t i2) const;
  const T& c_global(size_t i0, size_t i1, size_t i2, size_t i3) const;

  // MPI accessors
  MPI_Comm comm() const;
  int rank() const;
  int nprocs() const;

private:
  // Distribution type
  enum class DistType { DISTRIBUTED, REPLICATED };

  // Distribution information (nullptr = serial, non-null = parallel)
  struct distribution_info {
    DistType type;
    MPI_Comm comm;
    int rank;
    int nprocs;

    // For DISTRIBUTED arrays only:
    lattice global_lattice_;
    lattice local_core_;
    lattice local_extent_;
    std::unique_ptr<lattice_partitioner> partitioner_;
    std::vector<size_t> decomp_pattern_;  // Stores which dims are decomposed (0 = not decomposed)
    std::vector<size_t> ghost_widths_;    // Number of ghost layers per dimension

    // Neighbor info for ghost exchange
    struct Neighbor {
      int rank;              // Neighbor's MPI rank
      int direction;         // Which face: 0=left, 1=right (dim 0); 2=down, 3=up (dim 1); etc.
      size_t send_count;     // Number of elements to send
      size_t recv_count;     // Number of elements to receive
    };

    std::vector<Neighbor> neighbors_;
    bool neighbors_identified_ = false;
  };

  std::unique_ptr<distribution_info> dist_;

  // Helper methods
  bool should_use_parallel_io() const { return dist_ && dist_->type == DistType::DISTRIBUTED; }
  bool should_use_replicated_io() const { return dist_ && dist_->type == DistType::REPLICATED; }
  void setup_ghost_exchange();
  size_t calculate_buffer_size(int neighbor_idx, int pass);
  void pack_boundary_data(int neighbor_idx, std::vector<T>& buffer, int pass = 0);
  void unpack_ghost_data(int neighbor_idx, const std::vector<T>& buffer, int pass = 0);
  MPI_Datatype mpi_datatype() const;

  // GPU-aware MPI support
  bool has_gpu_aware_mpi() const;
  void exchange_ghosts_cpu();      // CPU path (current implementation)
  void exchange_ghosts_gpu_staged();  // GPU with host staging fallback
#if NDARRAY_HAVE_CUDA
  void exchange_ghosts_gpu_direct();  // GPU direct (requires GPU-aware MPI, nvcc compilation)
#endif
#else
  bool should_use_parallel_io() const { return false; }
  bool should_use_replicated_io() const { return false; }
#endif

private:
  storage_type storage_;  // Replaces: std::vector<T> p

  int device_type = NDARRAY_DEVICE_HOST;
  int device_id = 0;
  device_ptr devptr_;  // RAII-managed device memory

#if NDARRAY_HAVE_SYCL
  sycl::queue* sycl_queue_ptr = nullptr;  // Optional: user can provide their own queue
#endif
};

//////////////////////////////////

template <typename T, typename StoragePolicy>
int ndarray<T, StoragePolicy>::type() const {
  if constexpr (std::is_same_v<T, double>) {
    return NDARRAY_DTYPE_DOUBLE;
  } else if constexpr (std::is_same_v<T, float>) {
    return NDARRAY_DTYPE_FLOAT;
  } else if constexpr (std::is_same_v<T, int>) {
    return NDARRAY_DTYPE_INT;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return NDARRAY_DTYPE_UNSIGNED_INT;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return NDARRAY_DTYPE_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, char>) {
    return NDARRAY_DTYPE_CHAR;
  } else {
    return NDARRAY_DTYPE_UNKNOWN;
  }
}

#if 0
template <typename T, typename StoragePolicy>
unsigned int ndarray<T, StoragePolicy>::hash() const
{
  unsigned int h0 = murmurhash2(storage_.data(), sizeof(T)*storage_.size(), 0);
  unsigned int h1 = murmurhash2(dims.data(), sizeof(size_t)*dims.size(), h0);
  return h1;
}
#endif

template <typename T, typename StoragePolicy>
template <typename T1>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::operator*=(const T1& x)
{
  for (auto i = 0; i < storage_.size(); i ++)
    storage_[i] *= x;
  return *this;
}

template <typename T, typename StoragePolicy>
template <typename T1>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::operator/=(const T1& x)
{
  for (auto i = 0; i < storage_.size(); i ++)
    storage_[i] /= x;
  return *this;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::operator+=(const ndarray<T, StoragePolicy>& x)
{
  if (empty()) *this = x;
  else {
    assert(this->shapef() == x.shapef());
    for (auto i = 0; i < storage_.size(); i ++)
      storage_[i] += x.storage_[i];
  }
  return *this;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> operator+(const ndarray<T, StoragePolicy>& lhs, const ndarray<T, StoragePolicy>& rhs)
{
  ndarray<T, StoragePolicy> array;
  array.reshape(lhs);

  for (auto i = 0; i < array.nelem(); i ++)
    array[i] = lhs[i] + rhs[i];
  return array;
}

template <typename T, typename StoragePolicy, typename T1>
ndarray<T, StoragePolicy> operator*(const ndarray<T, StoragePolicy>& lhs, const T1& rhs)
{
  ndarray<T, StoragePolicy> array;
  array.reshape(lhs);
  for (auto i = 0; i < array.nelem(); i ++)
    array[i] = lhs[i] * rhs;
  return array;
}

template <typename T, typename StoragePolicy, typename T1>
ndarray<T, StoragePolicy> operator/(const ndarray<T, StoragePolicy>& lhs, const T1& rhs)
{
  ndarray<T, StoragePolicy> array;
  array.reshapef(lhs);
  for (auto i = 0; i < array.nelem(); i ++)
    array[i] = lhs[i] / rhs;
  return array;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::fill(T v)
{
#if NDARRAY_HAVE_CUDA
  if (device_type == NDARRAY_DEVICE_CUDA) {
    launch_fill<T>(static_cast<T*>(devptr_.get()), nelem(), v);
    cudaDeviceSynchronize();
    return;
  }
#endif
  storage_.fill(v);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::fill(const std::vector<T>& values)
{
  storage_.resize(values.size());
  for (size_t i = 0; i < values.size(); i++) {
    storage_[i] = values[i];
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::to_vector(std::vector<T> &out_vector) const{
  out_vector.resize(storage_.size());
  for (size_t i = 0; i < storage_.size(); i++) {
    out_vector[i] = storage_[i];
  }
}

template <typename T, typename StoragePolicy>
template <typename T1>
void ndarray<T, StoragePolicy>::from_array(const ndarray<T1>& array1)
{
  reshapef(array1.shapef());
  for (auto i = 0; i < storage_.size(); i ++)
    storage_[i] = static_cast<T>(array1[i]);
  n_component_dims = array1.multicomponents();
  is_time_varying = array1.has_time();
}

// Conversion from different storage policy
template <typename T, typename StoragePolicy>
template <typename T1, typename OtherPolicy>
void ndarray<T, StoragePolicy>::from_array(const ndarray<T1, OtherPolicy>& array1)
{
  dims = array1.shapef();
  reshapef(dims);  // This will recalculate strides (s)

  n_component_dims = array1.multicomponents();
  is_time_varying = array1.has_time();

  for (size_t i = 0; i < size(); i++) {
    storage_[i] = static_cast<T>(array1.data()[i]);
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::from_array(const T *x, const std::vector<size_t>& shape)
{
  reshapef(shape);
  memcpy(&storage_[0], x, nelem() * sizeof(T));
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::from_vector(const std::vector<T> &in_vector){
  for (int i=0;i<nelem();++i)
    if (i<in_vector.size())
      storage_[i] = in_vector[i];
    else break;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::copy_vector(const std::vector<T> &array)
{
  reshapef({array.size()});
  for (size_t i = 0; i < array.size(); i++) {
    storage_[i] = array[i];
  }
}

template <typename T, typename StoragePolicy>
template <typename Iterator>
void ndarray<T, StoragePolicy>::copy(Iterator first, Iterator last)
{
  // For native_storage with iterators, use direct copy
  size_t count = std::distance(first, last);
  reshapef({count});
  size_t i = 0;
  for (auto it = first; it != last; ++it, ++i) {
    storage_[i] = *it;
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::swap(ndarray& x)
{
  dims.swap(x.dims);
  s.swap(x.s);
  std::swap(storage_, x.storage_);
  std::swap(x.n_component_dims, n_component_dims);
  std::swap(x.is_time_varying, is_time_varying);
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::subarray(const lattice& l0) const
{
  lattice l(l0);
  if (l0.nd_cuttable() < nd()) {
    for (int i = 0; i < n_component_dims; i ++) {
      l.starts_.insert(l.starts_.begin(), 0);
      l.sizes_.insert(l.starts_.begin(), this->shapef(i));
    }
  }

  ndarray<T, StoragePolicy> arr(l.sizes());
  for (auto i = 0; i < arr.nelem(); i ++) {
    auto idx = l.from_integer(i);
    arr[i] = f(idx);
  }

  arr.n_component_dims = n_component_dims;
  return arr;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::bil_add_block_raw(const std::string& filename,
    const std::vector<size_t>& SZ,
    const lattice& ext)
{
#if NDARRAY_HAVE_MPI
  reshapef(ext.sizes());
  std::vector<int> domain, st, sz;

  for (int i = 0; i < nd(); i ++) {
    domain.push_back(SZ[i]);
    st.push_back(ext.start(i));
    sz.push_back(ext.size(i));
  }

  // BIL_Add_block_raw(nd(), domain.data(), st.data(), sz.data(), filename.c_str(), mpi_dtype(), (void**)&storage_[0]);
#else
  fatal(ERR_NOT_BUILT_WITH_MPI);
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::flip_byte_order(T &x)
{
  T y;
  char *px = (char*)&x, *py = (char*)&y;

  for (int i = 0; i < sizeof(T); i ++)
    py[sizeof(T)-i-1] = px[i];

  x = y;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::flip_byte_order()
{
  for (auto i = 0; i < this->nelem(); i ++)
    flip_byte_order(storage_[i]);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_binary_file(FILE *fp, int endian)
{
  auto s = fread(&storage_[0], sizeof(T), nelem(), fp);

#if NDARRAY_USE_LITTLE_ENDIAN
  if (endian == NDARRAY_ENDIAN_BIG)
    flip_byte_order();
#endif

#if NDARRAY_USE_BIG_ENDIAN
  if (endian == NDARRAY_ENDIAN_LITTLE)
    flip_byte_order();
#endif

  if (s != nelem())
    warn(ERR_FILE_CANNOT_READ_EXPECTED_BYTES);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::to_binary_file(FILE *fp)
{
  fwrite(&storage_[0], sizeof(T), nelem(), fp);
}

template <typename T, typename StoragePolicy>
template <typename T1>
void ndarray<T, StoragePolicy>::to_binary_file2(const std::string& f) const
{
  ndarray<T1> array;
  array.template from_array<T>(*this);
  array.to_binary_file(f);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_binary_file_sequence(const std::string& pattern, int endian) // Note: endian parameter not implemented (maintenance mode)
{
  const auto filenames = glob(pattern);
  if (filenames.size() == 0) return;

  std::vector<size_t> mydims = dims;
  mydims[nd() - 1] = filenames.size();
  reshapef(mydims);

  size_t npt = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<size_t>());
  for (int i = 0; i < filenames.size(); i ++) {
    // fprintf(stderr, "loading %s\n", filenames[i].c_str());
    FILE *fp = fopen(filenames[i].c_str(), "rb");
    fread(&storage_[npt*i], sizeof(T), npt, fp);
    fclose(fp);
  }
}

#ifdef NDARRAY_HAVE_VTK
template <typename T, typename StoragePolicy>
inline int ndarray<T, StoragePolicy>::vtk_data_type() const {
  if constexpr (std::is_same_v<T, char>) return VTK_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>) return VTK_UNSIGNED_CHAR;
  else if constexpr (std::is_same_v<T, short>) return VTK_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>) return VTK_UNSIGNED_SHORT;
  else if constexpr (std::is_same_v<T, int>) return VTK_INT;
  else if constexpr (std::is_same_v<T, unsigned int>) return VTK_UNSIGNED_INT;
  else if constexpr (std::is_same_v<T, long>) return VTK_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>) return VTK_UNSIGNED_LONG;
  else if constexpr (std::is_same_v<T, float>) return VTK_FLOAT;
  else if constexpr (std::is_same_v<T, double>) return VTK_DOUBLE;
  else return VTK_VOID;  // Unknown type
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::from_vtk_array(vtkSmartPointer<vtkAbstractArray> d)
{
  vtkSmartPointer<vtkDataArray> da = vtkDataArray::SafeDownCast(d);
  from_vtk_data_array(da);
}

template<typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::from_vtk_data_array(
    vtkSmartPointer<vtkDataArray> da)
{
  const int nc = da->GetNumberOfComponents(),
            ne = da->GetNumberOfTuples();
  if (nc > 1) {
    reshapef(nc, ne);
    set_multicomponents(1);
  } else {
    reshapef(ne);
    set_multicomponents(0);
  }

  for (auto i = 0; i < ne; i ++) {
    double *tuple = da->GetTuple(i);
    for (auto j = 0; j < nc; j ++)
      storage_[i*nc+j] = tuple[j];
  }
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::from_vtu(
    vtkSmartPointer<vtkUnstructuredGrid> d,
    const std::string array_name)
{
  vtkSmartPointer<vtkDataArray> da = d->GetPointData()->GetArray(array_name.c_str());
  if (!da) da = d->GetPointData()->GetArray(0);
  from_vtk_data_array(da);
}

template <typename T, typename StoragePolicy>
template <typename VTK_REGULAR_DATA>
inline void ndarray<T, StoragePolicy>::from_vtk_regular_data(
    vtkSmartPointer<VTK_REGULAR_DATA> d,
    const std::string array_name)
{
  vtkSmartPointer<vtkDataArray> da = d->GetPointData()->GetArray(array_name.c_str());
  if (!da) da = d->GetPointData()->GetArray(0);

  const int nd = d->GetDataDimension(),
            nc = da->GetNumberOfComponents();

  int vdims[3];
  d->GetDimensions(vdims);

  if (nd == 2) {
    if (nc == 1) reshapef(vdims[0], vdims[1]); //scalar field
    else {
      reshapef(nc, vdims[0], vdims[1]); // vector field
      n_component_dims = 1; // multicomponent array
    };
  } else if (nd == 3) {
    if (nc == 1) reshapef(vdims[0], vdims[1], vdims[2]); // scalar field
    else {
      reshapef(nc, vdims[0], vdims[1], vdims[2]);
      n_component_dims = 1; // multicomponent array
    }
  } else {
    fprintf(stderr, "[NDARRAY] fatal error: unsupported data dimension %d.\n", nd);
    assert(false);
  }

  for (auto i = 0; i < da->GetNumberOfTuples(); i ++) {
    double *tuple = da->GetTuple(i);
    for (auto j = 0; j < nc; j ++)
      storage_[i*nc+j] = tuple[j];
  }
}


template<typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::to_vtk_image_data_file(const std::string& filename, const std::string varname) const
{
  // fprintf(stderr, "to_vtk_image_data_file, n_component_dims=%zu\n", n_component_dims);
  vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkXMLImageDataWriter::New();
  writer->SetFileName(filename.c_str());
  writer->SetInputData( to_vtk_image_data(varname) );
  writer->Write();
}

template<typename T, typename StoragePolicy>
inline vtkSmartPointer<vtkImageData> ndarray<T, StoragePolicy>::to_vtk_image_data(std::string varname) const
{
  vtkSmartPointer<vtkImageData> d = vtkImageData::New();
  
  if (n_component_dims == 1) { // vector field
    if (nd() == 3) d->SetDimensions(shapef(1), shapef(2), 1);
    else if (nd() == 4) d->SetDimensions(shapef(1), shapef(2), shapef(3));
    
    if (varname.empty()) varname = "vector";
  } else if (n_component_dims == 2) { // tensor field
    if (nd() == 4) d->SetDimensions(shapef(2), shapef(3), 1);
    else if (nd() == 5) d->SetDimensions(shapef(2), shapef(3), shapef(4));
    
    if (varname.empty()) varname = "tensor";
  } else { // scalar field (n_component_dims == 0)
    if (nd() == 2) d->SetDimensions(shapef(0), shapef(1), 1);
    else if (nd() == 3) d->SetDimensions(shapef(0), shapef(1), shapef(2));
    
    if (varname.empty()) varname = "scalar";
  }
  
  d->GetPointData()->SetScalars(to_vtk_data_array(varname));

  return d;
}

template<typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_vtk_image_data_file_sequence(const std::string& pattern)
{
  const auto filenames = glob(pattern);
  if (filenames.size() == 0) return;
  storage_.resize(0);

  ndarray<T, StoragePolicy> array;
  std::vector<ndarray<T, StoragePolicy>> arrays;
  for (int t = 0; t < filenames.size(); t ++) {
    array.read_vtk_image_data_file(filenames[t]);
    arrays.push_back(array);
  }

  auto dims = array.dims;
  dims.push_back(filenames.size());
  reshapef(dims);

  // Copy all arrays into storage
  size_t offset = 0;
  for (const auto& arr : arrays) {
    for (size_t i = 0; i < arr.size(); i++) {
      storage_[offset++] = arr.storage_[i];
    }
  }
}
#else
template<typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_vtk_image_data_file_sequence(const std::string& pattern)
{
  throw feature_not_available(ERR_NOT_BUILT_WITH_VTK, "VTK support not enabled in this build");
}

template<typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::to_vtk_image_data_file(const std::string& filename, const std::string) const
{
  throw feature_not_available(ERR_NOT_BUILT_WITH_VTK, "VTK support not enabled in this build");
}
#endif

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>::ndarray(const T *a, const std::vector<size_t> &dims_)
{
  from_array(a, dims_);
#if 0
  dims = dims_;
  s.resize(dims.size());

  for (size_t i = 0; i < nd(); i ++)
    if (i == 0) s[i] = 1;
    else s[i] = s[i-1]*dims[i-1];

  storage_.resize(s[nd()-1]);
  for (size_t i = 0; i < s[nd()-1]; i++) {
    storage_[i] = a[i];
  }
#endif
}

template <typename T, typename StoragePolicy>
template <typename I>
void ndarray<T, StoragePolicy>::reshapef(const int ndims, const I sz[])
{
  std::vector<size_t> sizes(ndims);
  for (int i = 0; i < ndims; i ++)
    sizes[i] = sz[i];
  reshapef(sizes);
}

template <typename T, typename StoragePolicy>
template <typename T1>
void ndarray<T, StoragePolicy>::reshape(const ndarray<T1>& array)
{
  reshapef(array.shapef());
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::reshapef(const std::vector<size_t> &dims_)
{
  if (device_type == NDARRAY_DEVICE_HOST) {
    // dims_ is Fortran-order (user-facing), convert to C-order for internal storage
    dims = f_to_c_order(dims_);
    s.resize(dims.size());

    if (dims.size() == 0)
      throw std::invalid_argument("Cannot reshape to empty dimensions");

    // C-order strides: last dimension varies fastest
    // s[nd-1] = 1, s[nd-2] = dims[nd-1], etc.
    for (int i = this->nd() - 1; i >= 0; i--)
      if (i == this->nd() - 1) this->s[i] = 1;
      else this->s[i] = this->s[i+1] * this->dims[i+1];

    size_t total_size = this->s[0] * this->dims[0];

    // Use reshape() if the storage backend supports it (xtensor, eigen)
    if constexpr (has_reshape<storage_type>::value) {
      storage_.reshape(dims_);  // Backend gets F-order
    } else {
      storage_.resize(total_size);
    }
  } else
    throw device_error(ERR_NDARRAY_RESHAPE_DEVICE, "Reshaping a device ndarray is not supported");
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::reshapef(const std::vector<size_t> &dims, T val)
{
  reshapef(dims);
  this->fill(val);
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::scale(T factor)
{
#if NDARRAY_HAVE_CUDA
  if (device_type == NDARRAY_DEVICE_CUDA) {
    launch_scale<T>(static_cast<T*>(devptr_.get()), nelem(), factor);
    cudaDeviceSynchronize();
    return *this;
  }
#endif
  for (size_t i = 0; i < storage_.size(); i++) storage_[i] *= factor;
  return *this;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::add(const ndarray<T, StoragePolicy>& other)
{
  if (this->dims != other.dims)
    throw std::invalid_argument("ndarray::add: dimension mismatch");
#if NDARRAY_HAVE_CUDA
  if (device_type == NDARRAY_DEVICE_CUDA && other.device_type == NDARRAY_DEVICE_CUDA) {
    launch_add<T>(static_cast<T*>(devptr_.get()), static_cast<const T*>(other.devptr_.get()), nelem());
    cudaDeviceSynchronize();
    return *this;
  }
#endif
  for (size_t i = 0; i < storage_.size(); i++) storage_[i] += other.storage_[i];
  return *this;
}

template <typename T, typename StoragePolicy>
std::tuple<T, T> ndarray<T, StoragePolicy>::min_max() const {
  T min = std::numeric_limits<T>::max(),
    max = std::numeric_limits<T>::min();

  for (size_t i = 0; i < nelem(); i ++) {
    min = std::min(min, f(i));
    max = std::max(max, f(i));
  }

  return std::make_tuple(min, max);
}

template <typename T, typename StoragePolicy>
T ndarray<T, StoragePolicy>::maxabs() const
{
  T r = 0;
  for (size_t i = 0; i < nelem(); i ++)
    r = std::max(r, std::abs(storage_[i]));

  return r;
}

template <typename T, typename StoragePolicy>
T ndarray<T, StoragePolicy>::resolution() const {
  T r = std::numeric_limits<T>::max();

  for (size_t i = 0; i < nelem(); i ++)
    if (storage_[i] != T(0))
      r = std::min(r, std::abs(storage_[i]));

  return r;
}

#if NDARRAY_HAVE_MPI
template <typename T, typename StoragePolicy>
inline MPI_Datatype ndarray<T, StoragePolicy>::mpi_dtype() {
  if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
  else if constexpr (std::is_same_v<T, int>) return MPI_INT;
  else if constexpr (std::is_same_v<T, unsigned int>) return MPI_UNSIGNED;
  else if constexpr (std::is_same_v<T, long>) return MPI_LONG;
  else if constexpr (std::is_same_v<T, unsigned long>) return MPI_UNSIGNED_LONG;
  else if constexpr (std::is_same_v<T, short>) return MPI_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>) return MPI_UNSIGNED_SHORT;
  else if constexpr (std::is_same_v<T, char>) return MPI_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>) return MPI_UNSIGNED_CHAR;
  else if constexpr (std::is_same_v<T, long long>) return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<T, unsigned long long>) return MPI_UNSIGNED_LONG_LONG;
  else return MPI_BYTE;  // Fallback for unknown types
}
#endif

#if NDARRAY_HAVE_NETCDF
template <typename T, typename StoragePolicy>
inline int ndarray<T, StoragePolicy>::nc_dtype() const {
  if constexpr (std::is_same_v<T, double>) return NC_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) return NC_FLOAT;
  else if constexpr (std::is_same_v<T, int>) return NC_INT;
  else if constexpr (std::is_same_v<T, unsigned int>) return NC_UINT;
  else if constexpr (std::is_same_v<T, unsigned long>) return NC_UINT;
  else if constexpr (std::is_same_v<T, unsigned char>) return NC_UBYTE;
  else if constexpr (std::is_same_v<T, char>) return NC_CHAR;
  else if constexpr (std::is_same_v<T, short>) return NC_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>) return NC_USHORT;
  else if constexpr (std::is_same_v<T, long long>) return NC_INT64;
  else if constexpr (std::is_same_v<T, unsigned long long>) return NC_UINT64;
  else return -1;  // Unknown type
}
#else
template <typename T, typename StoragePolicy>
inline int ndarray<T, StoragePolicy>::nc_dtype() const { return -1; } // linking without netcdf
#endif

#if NDARRAY_HAVE_PNETCDF
template <typename T, typename StoragePolicy>
inline int ndarray<T, StoragePolicy>::pnc_dtype() const {
  if constexpr (std::is_same_v<T, double>) return NC_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) return NC_FLOAT;
  else if constexpr (std::is_same_v<T, int>) return NC_INT;
  else if constexpr (std::is_same_v<T, unsigned int>) return NC_UINT;
  else if constexpr (std::is_same_v<T, unsigned long>) return NC_UINT64;
  else if constexpr (std::is_same_v<T, unsigned char>) return NC_UBYTE;
  else if constexpr (std::is_same_v<T, char>) return NC_CHAR;
  else if constexpr (std::is_same_v<T, short>) return NC_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>) return NC_USHORT;
  else if constexpr (std::is_same_v<T, long long>) return NC_INT64;
  else if constexpr (std::is_same_v<T, unsigned long long>) return NC_UINT64;
  else return -1;  // Unknown type
}
#endif

#if NDARRAY_HAVE_ADIOS2
template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_bp(adios2::IO &io, adios2::Engine &reader, adios2::Variable<T>& var, int step)
{
  if (var) {
    // std::cerr << var << std::endl;
    // std::cerr << var.Shape() << std::endl;

    if (step == NDARRAY_ADIOS2_STEPS_UNSPECIFIED) {
      // nothing to do
    } else if (step == NDARRAY_ADIOS2_STEPS_ALL) {
      const size_t nsteps = var.Steps();
      var.SetStepSelection({0, nsteps-1});
    } else
      var.SetStepSelection({step, 1});

    std::vector<size_t> adios_shape(var.Shape());

    if (adios_shape.size()) { // array type
      // ADIOS2 uses C-order, ndarray now stores C-order - direct use!
      reshapec(adios_shape);

      // SetSelection uses ADIOS2 C-order
      std::vector<size_t> zeros(adios_shape.size(), 0);
      var.SetSelection({zeros, adios_shape});

      // For non-native storage, read into temp storage then copy
      if constexpr (std::is_same_v<StoragePolicy, native_storage>) {
        reader.Get<T>(var, storage_.data_);
      } else {
        std::vector<T> temp;
        reader.Get<T>(var, temp);
        for (size_t i = 0; i < temp.size(); i++) {
          storage_[i] = temp[i];
        }
      }
    } else { // scalar type
      reshapef(1);
      if constexpr (std::is_same_v<StoragePolicy, native_storage>) {
        reader.Get<T>(var, storage_.data_);
      } else {
        std::vector<T> temp(1);
        reader.Get<T>(var, temp);
        storage_[0] = temp[0];
      }
    }
  } else {
    throw ERR_ADIOS2_VARIABLE_NOT_FOUND;
    // fatal(ERR_ADIOS2_VARIABLE_NOT_FOUND);
  }
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_bp(adios2::IO &io, adios2::Engine &reader, const std::string &varname, int step)
{
  auto var = io.template InquireVariable<T>(varname);
  read_bp(io, reader, var, step);
}
#endif

#if NDARRAY_HAVE_ADIOS1
template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::read_bp_legacy(ADIOS_FILE *fp, const std::string& varname)
{
  warn("reading bp file with legacy ADIOS1 API..");
  ADIOS_VARINFO *avi = adios_inq_var(fp, varname.c_str());
  if (avi == NULL)
    throw ERR_ADIOS2_VARIABLE_NOT_FOUND;

  adios_inq_var_stat(fp, avi, 0, 0);
  adios_inq_var_blockinfo(fp, avi);
  adios_inq_var_meshinfo(fp, avi);

  int nt = 1;
  uint64_t st[4] = {0, 0, 0, 0}, sz[4] = {0, 0, 0, 0};
  std::vector<size_t> mydims;

  for (int i = 0; i < avi->ndim; i++) {
    st[i] = 0;
    sz[i] = avi->dims[i];
    nt = nt * sz[i];
    mydims.push_back(sz[i]);
  }
  // fprintf(stderr, "%d, %d, %d, %d\n", sz[0], sz[1], sz[2], sz[3]);

  if (!mydims.empty()) {
    // ADIOS1 uses C-order, ndarray now stores C-order - direct use!
    reshapec(mydims);

    // Selection uses ADIOS1 C-order
    ADIOS_SELECTION *sel = adios_selection_boundingbox(avi->ndim, st, sz);
    assert(sel->type == ADIOS_SELECTION_BOUNDINGBOX);

    adios_schedule_read_byid(fp, sel, avi->varid, 0, 1, &storage_[0]);
    int retval = adios_perform_reads(fp, 1);

    adios_selection_delete(sel);
    return true; // avi->ndim;
  } else {
    // Note: Only adios_integer scalar type supported (maintenance mode)
    if (avi->type == adios_integer) {
      reshapef({1});
      storage_[0] = *((int*)avi->value);
      return true;
    }
    else return false;
  }
}
#endif

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::from_bp_legacy(const std::string& filename, const std::string& varname, MPI_Comm comm)
{
  ndarray<T, StoragePolicy> arr;
  arr.read_bp_legacy(filename, varname, comm);
  return arr;
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::read_bp_legacy(const std::string& filename, const std::string& varname, MPI_Comm comm)
{
#if NDARRAY_HAVE_ADIOS1
  adios_read_init_method( ADIOS_READ_METHOD_BP, comm, "" );
  ADIOS_FILE *fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm);
  // adios_read_bp_reset_dimension_order(fp, 0);

  bool succ = read_bp_legacy(fp, varname);

  adios_read_finalize_method (ADIOS_READ_METHOD_BP);
  adios_read_close(fp);
  return succ;
#else
  throw feature_not_available(ERR_NOT_BUILT_WITH_ADIOS1, "ADIOS1 support not enabled in this build");
#endif
}


template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::to_device(int dev, int id)
{
  if (dev == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
    if (this->device_type == NDARRAY_DEVICE_CUDA) { // already on gpu
      warn("array already on device");
    } else {
      this->device_type = NDARRAY_DEVICE_CUDA;
      this->device_id = id;

      devptr_.allocate(sizeof(T) * nelem(), NDARRAY_DEVICE_CUDA, id);
      cudaSetDevice(id);
      cudaMemcpy(devptr_.get(), storage_.data(), sizeof(T) * storage_.size(),
          cudaMemcpyHostToDevice);
      storage_.resize(0);
    }
#else
    fatal(ERR_NOT_BUILT_WITH_CUDA);
#endif
  } else if (dev == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
    if (this->device_type == NDARRAY_DEVICE_SYCL) {
      warn("array already on SYCL device");
    } else {
      this->device_type = NDARRAY_DEVICE_SYCL;
      this->device_id = id;

      // Use provided queue or create default queue
      sycl::queue* q = sycl_queue_ptr;
      bool own_queue = false;
      if (q == nullptr) {
        q = new sycl::queue(sycl::default_selector{});
        own_queue = true;
      }

      // Allocate device memory
      devptr_.allocate_sycl(sizeof(T) * nelem(), *q, id);

      // Copy data to device
      q->memcpy(devptr_.get(), storage_.data(), sizeof(T) * storage_.size()).wait();

      // Clean up temporary queue if we created it
      if (own_queue) delete q;

      storage_.resize(0);
    }
#else
    fatal(ERR_NOT_BUILT_WITH_SYCL);
#endif
  } else
    fatal(ERR_NDARRAY_UNKNOWN_DEVICE);
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::to_host()
{
  if (this->device_type == NDARRAY_DEVICE_HOST) {
    warn("array already on host");
  } else if (this->device_type == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
    if (this->device_type == NDARRAY_DEVICE_CUDA) {
      storage_.resize(nelem());

      cudaSetDevice(this->device_id);
      cudaMemcpy(storage_.data(), devptr_.get(), sizeof(T) * storage_.size(),
          cudaMemcpyDeviceToHost);
      devptr_.free();

      this->device_type = NDARRAY_DEVICE_HOST;
      this->device_id = 0;
    } else
      fatal("array not on device");
#else
    fatal(ERR_NOT_BUILT_WITH_CUDA);
#endif
  } else if (this->device_type == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
    storage_.resize(nelem());

    // Use provided queue or create default queue
    sycl::queue* q = sycl_queue_ptr;
    bool own_queue = false;
    if (q == nullptr) {
      q = new sycl::queue(sycl::default_selector{});
      own_queue = true;
    }

    // Copy data from device
    q->memcpy(storage_.data(), devptr_.get(), sizeof(T) * storage_.size()).wait();

    // Free device memory (RAII wrapper handles cleanup)
    devptr_.free();

    // Clean up temporary queue if we created it
    if (own_queue) delete q;

    this->device_type = NDARRAY_DEVICE_HOST;
    this->device_id = 0;
#else
    fatal(ERR_NOT_BUILT_WITH_SYCL);
#endif
  } else
    fatal(ERR_NDARRAY_UNKNOWN_DEVICE);
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::copy_to_device(int dev, int id)
{
  if (dev == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
    if (this->device_type == NDARRAY_DEVICE_CUDA) {
      warn("array already on CUDA device");
    } else if (this->device_type != NDARRAY_DEVICE_HOST) {
      fatal("array is on a different device type");
    } else {
      this->device_type = NDARRAY_DEVICE_CUDA;
      this->device_id = id;

      devptr_.allocate(sizeof(T) * nelem(), NDARRAY_DEVICE_CUDA, id);
      cudaSetDevice(id);
      cudaMemcpy(devptr_.get(), storage_.data(), sizeof(T) * storage_.size(),
          cudaMemcpyHostToDevice);
      // Note: storage_ is NOT cleared, keeping data on both host and device
    }
#else
    fatal(ERR_NOT_BUILT_WITH_CUDA);
#endif
  } else if (dev == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
    if (this->device_type == NDARRAY_DEVICE_SYCL) {
      warn("array already on SYCL device");
    } else if (this->device_type != NDARRAY_DEVICE_HOST) {
      fatal("array is on a different device type");
    } else {
      this->device_type = NDARRAY_DEVICE_SYCL;
      this->device_id = id;

      // Use provided queue or create default queue
      sycl::queue* q = sycl_queue_ptr;
      bool own_queue = false;
      if (q == nullptr) {
        q = new sycl::queue(sycl::default_selector{});
        own_queue = true;
      }

      // Allocate device memory
      devptr_.allocate_sycl(sizeof(T) * nelem(), *q, id);

      // Copy data to device
      q->memcpy(devptr_.get(), storage_.data(), sizeof(T) * storage_.size()).wait();

      // Clean up temporary queue if we created it
      if (own_queue) delete q;

      // Note: storage_ is NOT cleared, keeping data on both host and device
    }
#else
    fatal(ERR_NOT_BUILT_WITH_SYCL);
#endif
  } else
    fatal(ERR_NDARRAY_UNKNOWN_DEVICE);
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::copy_from_device()
{
  if (this->device_type == NDARRAY_DEVICE_HOST) {
    warn("array is on host, nothing to copy");
  } else if (this->device_type == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
    if (storage_.size() == 0) {
      storage_.resize(nelem());
    }

    cudaSetDevice(this->device_id);
    cudaMemcpy(storage_.data(), devptr_.get(), sizeof(T) * storage_.size(),
        cudaMemcpyDeviceToHost);
    // Note: device memory is NOT freed
#else
    fatal(ERR_NOT_BUILT_WITH_CUDA);
#endif
  } else if (this->device_type == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
    if (storage_.size() == 0) {
      storage_.resize(nelem());
    }

    // Use provided queue or create default queue
    sycl::queue* q = sycl_queue_ptr;
    bool own_queue = false;
    if (q == nullptr) {
      q = new sycl::queue(sycl::default_selector{});
      own_queue = true;
    }

    // Copy data from device
    q->memcpy(storage_.data(), devptr_.get(), sizeof(T) * storage_.size()).wait();

    // Clean up temporary queue if we created it
    if (own_queue) delete q;

    // Note: device memory is NOT freed
#else
    fatal(ERR_NOT_BUILT_WITH_SYCL);
#endif
  } else
    fatal(ERR_NDARRAY_UNKNOWN_DEVICE);
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::from_file(const std::string& filename, const std::string varname, MPI_Comm comm)
{
  ndarray<T, StoragePolicy> array;
  array.read_file(filename, varname, comm);
  return array;
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::read_file(const std::string& filename, const std::string varname, MPI_Comm comm)
{
  if (!file_exists(filename)) {
    warn(ERR_FILE_NOT_FOUND);
    return false;
  }

  auto ext = file_extension(filename);
  if (ext == FILE_EXT_BP) read_bp(filename, varname, -1, comm); // Note: step=-1 (last timestep) hardcoded
  else if (ext == FILE_EXT_NETCDF) read_netcdf(filename, varname, comm);
  else if (ext == FILE_EXT_VTI) read_vtk_image_data_file(filename, varname);
  else if (ext == FILE_EXT_HDF5) read_h5(filename, varname);
  else fatal(ERR_FILE_UNRECOGNIZED_EXTENSION);

  return true; // Note: read_* functions throw exceptions on error, so reaching here means success
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::from_bp(const std::string& filename, const std::string& name, int step, MPI_Comm comm)
{
  ndarray<T, StoragePolicy> array;
  array.read_bp(filename, name, step, comm);

  return array;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::from_h5(const std::string& filename, const std::string& name)
{
  ndarray<T, StoragePolicy> array;
  array.read_h5(filename, name);
  return array;
}

#if NDARRAY_HAVE_HDF5
template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_h5_did(hid_t did)
{
  auto sid = H5Dget_space(did); // space id
  auto type = H5Sget_simple_extent_type(sid);

  if (type == H5S_SIMPLE) {
    const int h5ndims = H5Sget_simple_extent_ndims(sid);
    std::vector<hsize_t> h5dims(h5ndims);
    H5Sget_simple_extent_dims(sid, h5dims.data(), NULL);

    std::vector<size_t> h5_dims(h5ndims);
    for (auto i = 0; i < h5ndims; i ++)
      h5_dims[i] = h5dims[i];

    // HDF5 uses C-order, ndarray now stores C-order - direct use!
    reshapec(h5_dims);

    // H5Dread with H5S_ALL reads entire dataset
    H5Dread(did, h5_mem_type_id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, storage_.data());
  } else if (type == H5S_SCALAR) {
    reshapef(1);
    H5Dread(did, h5_mem_type_id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, storage_.data());
  } else {
    throw hdf5_error(ERR_HDF5_IO, "Unsupported HDF5 extent type");
  }
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::to_h5(const std::string& filename, const std::string& varname) const
{
  hid_t dtype = h5_mem_type_id();
  if (dtype < 0) {
    throw hdf5_error(ERR_HDF5_UNSUPPORTED_TYPE, "Unsupported data type for HDF5 output");
  }

  hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file_id < 0) {
    throw hdf5_error(ERR_HDF5_IO, "Cannot create HDF5 file: " + filename);
  }

  const size_t nd = this->nd();

  // HDF5 uses C-order, ndarray now stores C-order - direct use!
  auto h5_dims_vec = shapec();  // Already in C-order
  std::vector<hsize_t> h5_dims(h5_dims_vec.begin(), h5_dims_vec.end());

  hid_t dataspace_id = H5Screate_simple(static_cast<int>(nd), h5_dims.data(), NULL);
  hid_t dataset_id = H5Dcreate2(file_id, varname.c_str(), dtype, dataspace_id,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  if (dataset_id < 0) {
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    throw hdf5_error(ERR_HDF5_IO, "Cannot create HDF5 dataset: " + varname);
  }

  H5Dwrite(dataset_id, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, storage_.data());
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);
  H5Fclose(file_id);
}

template <typename T, typename StoragePolicy>
inline hid_t ndarray<T, StoragePolicy>::h5_mem_type_id() {
  if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, int>) return H5T_NATIVE_INT;
  else if constexpr (std::is_same_v<T, unsigned long>) return H5T_NATIVE_ULONG;
  else if constexpr (std::is_same_v<T, unsigned int>) return H5T_NATIVE_UINT;
  else if constexpr (std::is_same_v<T, unsigned char>) return H5T_NATIVE_UCHAR;
  else if constexpr (std::is_same_v<T, char>) return H5T_NATIVE_CHAR;
  else if constexpr (std::is_same_v<T, short>) return H5T_NATIVE_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>) return H5T_NATIVE_USHORT;
  else if constexpr (std::is_same_v<T, long>) return H5T_NATIVE_LONG;
  else if constexpr (std::is_same_v<T, long long>) return H5T_NATIVE_LLONG;
  else if constexpr (std::is_same_v<T, unsigned long long>) return H5T_NATIVE_ULLONG;
  else {
    // Unsupported type - return an invalid type
    return -1;
  }
}
#endif

template <typename T, typename StoragePolicy>
inline ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::slice(const lattice& l) const
{
  ndarray<T, StoragePolicy> array(l.sizes());
  for (auto i = 0; i < l.n(); i ++) {
    auto idx = l.from_integer(i);
    array[i] = f(idx);
  }
  return array;
}

template <typename T, typename StoragePolicy>
inline ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::slice(const std::vector<size_t>& st, const std::vector<size_t>& sz) const
{
  return slice(lattice(st, sz));
}

template <typename T, typename StoragePolicy>
inline ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::slice_time(size_t t) const
{
  ndarray<T, StoragePolicy> array;
  std::vector<size_t> mydims(dims);
  mydims.resize(nd()-1);

  array.reshapef(mydims);
  memcpy(&array[0], &storage_[t * s[nd()-1]], s[nd()-1] * sizeof(T));

  return array;
}

template <typename T, typename StoragePolicy>
inline std::vector<ndarray<T, StoragePolicy>> ndarray<T, StoragePolicy>::slice_time() const
{
  std::vector<ndarray<T, StoragePolicy>> arrays;
  const size_t nt = shape(nd()-1);
  for (size_t i = 0; i < nt; i ++)
    arrays.push_back(slice_time(i));
  return arrays;
}

template <typename T, typename StoragePolicy>
std::ostream& ndarray<T, StoragePolicy>::print(std::ostream& os) const
{
  print_shapef(os);

  if (nd() == 1) {
    os << "[";
    for (size_t i = 0; i < dims[0]; i ++)
      if (i < dims[0]-1) os << f(i) << ", ";
      else os << f(i) << "]";
  } else if (nd() == 2) {
    os << "[";
    for (size_t j = 0; j < dims[1]; j ++) {
      os << "[";
      for (size_t i = 0; i < dims[0]; i ++)
        if (i < dims[0]-1) os << f(i, j) << ", ";
        else os << f(i, j) << "]";
      if (j < dims[1]-1) os << "], ";
      else os << "]";
    }
  } else if (nd() == 3) {
    os << "[";
    for (size_t k = 0; k < dims[2]; k ++) {
      for (size_t j = 0; j < dims[1]; j ++) {
        os << "[";
        for (size_t i = 0; i < dims[0]; i ++)
          if (i < dims[0]-1) os << f(i, j) << ", ";
          else os << f(i, j) << "]";
        if (j < dims[1]-1) os << "], ";
        else os << "]";
      }
      if (k < dims[2]-1) os << "], ";
      else os << "]";
    }
  }

  return os;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::transpose()
{
  *this = get_transpose();
  return *this;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::get_transpose() const
{
  ndarray<T, StoragePolicy> a;
  if (nd() == 0) return a;
  else if (nd() == 1) return *this;
  else if (nd() == 2) {
    a.reshapef(dimf(1), dimf(0));
    for (auto i = 0; i < dimf(0); i ++)
      for (auto j = 0; j < dimf(1); j ++)
        a.f(j, i) = f(i, j);
    return a;
  } else if (nd() == 3) {
    a.reshapef(dimf(2), dimf(1), dimf(0));
    for (auto i = 0; i < dimf(0); i ++)
      for (auto j = 0; j < dimf(1); j ++)
        for (auto k = 0; k < dimf(2); k ++)
          a.f(k, j, i) = f(i, j, k);
    return a;
  } else if (nd() == 4) {
    a.reshapef(dimf(3), dimf(2), dimf(1), dimf(0));
    for (auto i = 0; i < dimf(0); i ++)
      for (auto j = 0; j < dimf(1); j ++)
        for (auto k = 0; k < dimf(2); k ++)
          for (auto l = 0; l < dimf(3); l ++)
            a.f(l, k, j, i) = f(i, j, k, l);
    return a;
  } else {
    throw std::logic_error("Unsupported dimensionality for global index conversion (only 1D, 2D, 3D supported)");
  }
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::concat(const std::vector<ndarray<T, StoragePolicy>>& arrays)
{
  ndarray<T, StoragePolicy> result;
  std::vector<size_t> result_shape = arrays[0].shapef();
  result_shape.insert(result_shape.begin(), arrays.size());
  result.reshapef(result_shape);

  const auto n = arrays[0].nelem();
  const auto n1 = arrays.size();

  for (auto i = 0; i < n; i ++)
    for (auto j = 0 ; j < n1; j ++)
      result[i*n1 + j] = arrays[j][i];

  return result;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> ndarray<T, StoragePolicy>::stack(const std::vector<ndarray<T, StoragePolicy>>& arrays)
{
  ndarray<T, StoragePolicy> result;
  std::vector<size_t> result_shape = arrays[0].shapef();
  result_shape.push_back(arrays.size());
  result.reshapef(result_shape);

  const auto n = arrays[0].nelem();
  const auto n1 = arrays.size();

  for (auto j = 0 ; j < n1; j ++)
    for (auto i = 0; i < n; i ++)
      result[i + j*n] = arrays[j][i];

  return result;
}

#if NDARRAY_HAVE_PYBIND11
template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>::ndarray(const pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> &numpy_array)
{
  from_numpy(numpy_array);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::from_numpy(const pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast> &array)
{
  pybind11::buffer_info buf = array.request();
  std::vector<size_t> shape;
  for (auto i = 0; i < buf.ndim; i ++)
    shape.push_back(array.shape(i));
  reshapef(shape);

  from_array((T*)buf.ptr, shape);
}

template <typename T, typename StoragePolicy>
pybind11::array_t<T, pybind11::array::c_style> ndarray<T, StoragePolicy>::to_numpy() const
{
  auto result = pybind11::array_t<T>(nelem());
  result.resize(shapef());
  pybind11::buffer_info buf = result.request();

  T *ptr = (T*)buf.ptr;
  memcpy(ptr, data(), sizeof(T) * nelem());

  return result;
}
#endif

#if NDARRAY_HAVE_PNG
template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_png(const std::string& filename)
{
  std::vector<unsigned char> png_data;
  int width, height, channels;

  read_png_file(filename, png_data, width, height, channels);

  if (channels == 1) {
    // Grayscale: shape is (height, width)
    reshapef(height, width);
  } else {
    // RGB/RGBA: shape is (channels, height, width) - multicomponent
    reshapef(channels, height, width);
    set_multicomponents();
  }

  // Copy data with type conversion
  for (size_t i = 0; i < png_data.size(); i++) {
    storage_[i] = static_cast<T>(png_data[i]);
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::to_png(const std::string& filename) const
{
  std::vector<unsigned char> png_data;
  int width, height, channels;

  // Determine format from array shape
  if (nd() == 2) {
    // Grayscale: (height, width)
    height = dimf(0);
    width = dimf(1);
    channels = 1;
  } else if (nd() == 3 && multicomponents()) {
    // RGB/RGBA: (channels, height, width)
    channels = dimf(0);
    height = dimf(1);
    width = dimf(2);

    if (channels != 3 && channels != 4) {
      fatal("PNG write requires 1 (gray), 3 (RGB), or 4 (RGBA) channels");
    }
  } else {
    fatal("Unable to save to PNG: array must be 2D (grayscale) or 3D multicomponent (RGB/RGBA)");
  }

  // Convert data to unsigned char
  png_data.resize(nelem());
  for (size_t i = 0; i < nelem(); i++) {
    // Clamp to [0, 255]
    T val = storage_[i];
    if (val < T(0)) val = T(0);
    if (val > T(255)) val = T(255);
    png_data[i] = static_cast<unsigned char>(val);
  }

  write_png_file(filename, png_data.data(), width, height, channels);
}
#else
template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_png(const std::string& filename)
{
  fatal(ERR_NOT_BUILT_WITH_PNG);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::to_png(const std::string& filename) const
{
  fatal(ERR_NOT_BUILT_WITH_PNG);
}
#endif

// see https://www.csc.kth.se/~weinkauf/notes/amiramesh.html
template <>
inline bool ndarray<float>::read_amira(const std::string& filename)
{
  auto find_and_jump = [](const char* buffer, const char* SearchString) {
    const char* FoundLoc = strstr(buffer, SearchString);
    if (FoundLoc) return FoundLoc + strlen(SearchString);
    return buffer;
  };

  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    warn(ERR_FILE_CANNOT_OPEN, filename);
    return false;
  }

  char buffer[2048];
  fread(buffer, sizeof(char), 2047, fp);
  buffer[2047] = '\0';

  if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1")) {
    warn(ERR_FILE_FORMAT_AMIRA, filename);
    fclose(fp);
    return false;
  }

  int xDim(0), yDim(0), zDim(0);
  sscanf(find_and_jump(buffer, "define Lattice"), "%d %d %d", &xDim, &yDim, &zDim);
  printf("\tAmriaMesh grid dimensions: %d %d %d\n", xDim, yDim, zDim);

  float xmin(1.0f), ymin(1.0f), zmin(1.0f);
  float xmax(-1.0f), ymax(-1.0f), zmax(-1.0f);

  sscanf(find_and_jump(buffer, "BoundingBox"), "%g %g %g %g %g %g", &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);
  printf("\tBoundingBox in x-Direction: [%g ... %g]\n", xmin, xmax);
  printf("\tBoundingBox in y-Direction: [%g ... %g]\n", ymin, ymax);
  printf("\tBoundingBox in z-Direction: [%g ... %g]\n", zmin, zmax);

  const bool bIsUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
  printf("\tGridType: %s\n", bIsUniform ? "uniform" : "UNKNOWN");

  int NumComponents(0);
  if (strstr(buffer, "Lattice { float Data }"))
  { //Scalar field
    NumComponents = 1;
  } else {
    sscanf(find_and_jump(buffer, "Lattice { float["), "%d", &NumComponents);
  }
  printf("\tNumber of Components: %d\n", NumComponents);

  if (xDim <= 0 || yDim <= 0 || zDim <= 0
      || xmin > xmax || ymin > ymax || zmin > zmax
      || !bIsUniform || NumComponents <= 0)
  {
    warn(ERR_FILE_FORMAT_AMIRA);
    fclose(fp);
    return false;
  }

  const long idxStartData = strstr(buffer, "# Data section follows") - buffer;
  if (idxStartData > 0)
  {
    fseek(fp, idxStartData, SEEK_SET);
    fgets(buffer, 2047, fp);
    fgets(buffer, 2047, fp);

    reshapef(NumComponents, xDim, yDim, zDim);
    set_multicomponents();
    read_binary_file(fp);
  }

  fclose(fp);
  return 0;
}

template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy>& ndarray<T, StoragePolicy>::perturb(T sigma)
{
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, sigma};

  for (auto i = 0; i < nelem(); i ++)
    storage_[i] = storage_[i] + d(gen);

  return *this;
}

template <typename T, typename StoragePolicy>
template <typename F>
bool ndarray<T, StoragePolicy>::mlerp(const F x[], T v[]) const
{
  static const size_t maxn = 12; // some arbitrary large number for nd
  const size_t n = nd() - multicomponents(),
               nc = multicomponents();

  // fprintf(stderr, "n=%zu, nc=%zu\n", n, nc);
  int x0[maxn]; // corner
  F mu[maxn]; // unit coordinates
  for (size_t i = 0; i < n; i ++) {
    if (x[i] < 0 || x[i] >= dims[nc+i] - 1) return false; // out of bound

    x0[i] = std::floor(x[i]);
    mu[i] = x[i] - x0[i];
  }

  if (nc == 0)
    v[0] = F(0);
  else if (nc == 1)
    for (size_t i = 0; i < dims[0]; i ++)
      v[i] = 0;
  else
    fatal(ERR_NOT_IMPLEMENTED);

  // fprintf(stderr, "w=%f, %f\n", v[0], v[1]);
  // fprintf(stderr, "mu=%f, %f\n", mu[0], mu[1]);

  for (size_t cur = 0; cur < (1 << n); cur ++) {
    bool b[maxn];
    for (size_t i = 0; i < n; i ++)
      b[i] = (cur >> i) & 1; // check if ith bit is set

    F coef(1);
    size_t verts[maxn];
    for (size_t i = 0; i < n; i ++) {
      verts[nc+i] = x0[i] + b[i];

      if (b[i])
        coef *= mu[i];
      else
        coef *= (F(1) - mu[i]);
    }
    // fprintf(stderr, "coef=%f\n", coef);

    if (nc == 0) // univariate
      v[0] += coef * this->f(verts);
    else if (nc == 1) { // multiple channels
      for (int k = 0; k < dimf(0); k ++) {
        verts[0] = k;
        F val = this->f(verts);
        v[k] += coef * val;
        // fprintf(stderr, "k=%d, verts=%zu, %zu, %zu, coef=%f, val=%f\n", k, verts[0], verts[1], verts[2], coef, val);
      }
    } else
      fatal(ERR_NOT_IMPLEMENTED);
  }

  // fprintf(stderr, "v=%f, %f\n", v[0], v[1]);
  return true;
}

/////
template <typename T>
inline T mse(const ndarray<T>& x, const ndarray<T>& y)
{
  T r(0);
  const auto n = std::min(x.size(), y.size());
  size_t m = 0;
  for (auto i = 0; i < n; i ++) {
    if (std::isnan(x[i]) || std::isinf(x[i]) ||
        std::isnan(y[i]) || std::isinf(y[i]))
      continue;
    else {
      const T d = x[i] - y[i];
      r += d * d;
      m ++;
    }
  }
  return r / m;
}

template <typename T>
T rmse(const ndarray<T>& x, const ndarray<T>& y)
{
  return std::sqrt(mse(x, y));
}

template <typename T>
T psnr(const ndarray<T>& x, const ndarray<T>& xp)
{
  const auto min_max = x.min_max();
  const auto range = std::get<1>(min_max) - std::get<0>(min_max);
  return 20.0 * log10(range) - 10.0 * log10(mse(x, xp));
}

//////
// Factory methods (moved to the end of file to ensure ndarray<T> is fully defined)

///////////
// PNetCDF implementation
///////////

#if NDARRAY_HAVE_PNETCDF

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::read_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz)
{
  // Get variable dimensionality
  int ndims;
  PNC_SAFE_CALL(ncmpi_inq_varndims(ncid, varid, &ndims));

  // Get dimensions
  std::vector<MPI_Offset> dims(ndims);
  for (int i = 0; i < ndims; i++) {
    dims[i] = sz[i];
  }

  // Reshape array based on count
  std::vector<size_t> shape(ndims);
  for (int i = 0; i < ndims; i++) {
    shape[i] = static_cast<size_t>(sz[i]);
  }
  reshapec(shape);  // Use C order for NetCDF compatibility

  // Collective read based on type
  if (std::is_same<T, float>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_float_all(ncid, varid, st, sz, reinterpret_cast<float*>(storage_.data())));
  } else if (std::is_same<T, double>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_double_all(ncid, varid, st, sz, reinterpret_cast<double*>(storage_.data())));
  } else if (std::is_same<T, int>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_int_all(ncid, varid, st, sz, reinterpret_cast<int*>(storage_.data())));
  } else if (std::is_same<T, unsigned int>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_uint_all(ncid, varid, st, sz, reinterpret_cast<unsigned int*>(storage_.data())));
  } else if (std::is_same<T, short>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_short_all(ncid, varid, st, sz, reinterpret_cast<short*>(storage_.data())));
  } else if (std::is_same<T, long long>::value) {
    PNC_SAFE_CALL(ncmpi_get_vara_longlong_all(ncid, varid, st, sz, reinterpret_cast<long long*>(storage_.data())));
  } else {
    fatal("Unsupported type for read_pnetcdf_all");
  }
}

template <typename T, typename StoragePolicy>
inline void ndarray<T, StoragePolicy>::write_pnetcdf_all(int ncid, int varid, const MPI_Offset *st, const MPI_Offset *sz) const
{
  // Collective write based on type
  if (std::is_same<T, float>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_float_all(ncid, varid, st, sz, reinterpret_cast<const float*>(storage_.data())));
  } else if (std::is_same<T, double>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_double_all(ncid, varid, st, sz, reinterpret_cast<const double*>(storage_.data())));
  } else if (std::is_same<T, int>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_int_all(ncid, varid, st, sz, reinterpret_cast<const int*>(storage_.data())));
  } else if (std::is_same<T, unsigned int>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_uint_all(ncid, varid, st, sz, reinterpret_cast<const unsigned int*>(storage_.data())));
  } else if (std::is_same<T, short>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_short_all(ncid, varid, st, sz, reinterpret_cast<const short*>(storage_.data())));
  } else if (std::is_same<T, long long>::value) {
    PNC_SAFE_CALL(ncmpi_put_vara_longlong_all(ncid, varid, st, sz, reinterpret_cast<const long long*>(storage_.data())));
  } else {
    fatal("Unsupported type for write_pnetcdf_all");
  }
}

#endif // NDARRAY_HAVE_PNETCDF

//////////////////////////////////
// MPI and distributed memory support implementations
//////////////////////////////////

#if NDARRAY_HAVE_MPI

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::decompose(MPI_Comm comm,
                                          const std::vector<size_t>& global_dims,
                                          size_t nprocs,
                                          const std::vector<size_t>& decomp,
                                          const std::vector<size_t>& ghost)
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    this->reshapef(global_dims);
    return;
  }

  // Validate decomp size if provided
  if (!decomp.empty() && decomp.size() != global_dims.size()) {
    throw std::invalid_argument("decomp size must match global_dims size");
  }

  // Validate ghost size if provided
  if (!ghost.empty() && ghost.size() != global_dims.size()) {
    throw std::invalid_argument("ghost size must match global_dims size");
  }

  // Create distribution info
  dist_ = std::make_unique<distribution_info>();
  dist_->type = DistType::DISTRIBUTED;
  dist_->comm = comm;
  MPI_Comm_rank(comm, &dist_->rank);
  MPI_Comm_size(comm, &dist_->nprocs);

  // Create global lattice
  dist_->global_lattice_ = lattice(global_dims);

  // Store decomposition pattern and ghost widths
  dist_->decomp_pattern_ = decomp;
  if (ghost.empty()) {
    dist_->ghost_widths_.assign(global_dims.size(), 0);
  } else {
    dist_->ghost_widths_ = ghost;
  }

  // Use provided nprocs or communicator size
  size_t np = (nprocs == 0) ? dist_->nprocs : nprocs;

  // Create effective decomposition pattern
  // For dimensions with decomp[i] == 0, lattice_partitioner should not split
  std::vector<size_t> effective_decomp = decomp;
  std::vector<size_t> effective_ghost = ghost;

  // If empty, lattice_partitioner will auto-decompose
  // But we need to be careful about which dimensions to decompose

  // Create partitioner
  // Note: lattice_partitioner needs to handle decomp[i]==0 meaning "don't split"
  dist_->partitioner_ = std::make_unique<lattice_partitioner>(dist_->global_lattice_);

  // Partition the domain
  dist_->partitioner_->partition(np, effective_decomp, effective_ghost);

  // Get local core and extent for this rank
  dist_->local_core_ = dist_->partitioner_->get_core(dist_->rank);
  dist_->local_extent_ = dist_->partitioner_->get_ext(dist_->rank);

  // For non-decomposed dimensions (decomp[i]==0), ensure local == global
  if (!decomp.empty()) {
    for (size_t i = 0; i < decomp.size(); i++) {
      if (decomp[i] == 0) {
        // This dimension should NOT be decomposed
        // Verify that local_core has full extent
        if (dist_->local_core_.size(i) != global_dims[i]) {
          // lattice_partitioner didn't handle it correctly
          // This is a workaround - ideally lattice_partitioner should handle it
          std::cerr << "Warning: decomp[" << i << "]==0 but dimension was split. "
                    << "This may indicate lattice_partitioner doesn't support non-decomposed dimensions yet." << std::endl;
        }
      }
    }
  }

  // Reshape local storage to hold extent (core + ghosts)
  this->reshapef(dist_->local_extent_.sizes());

  // Setup ghost exchange topology
  setup_ghost_exchange();
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::set_replicated(MPI_Comm comm)
{
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) return;

  // Create distribution info
  dist_ = std::make_unique<distribution_info>();
  dist_->type = DistType::REPLICATED;
  dist_->comm = comm;
  MPI_Comm_rank(comm, &dist_->rank);
  MPI_Comm_size(comm, &dist_->nprocs);

  // No decomposition - global = local
  // Array will be reshaped by subsequent read/write operations
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::is_distributed() const
{
  return dist_ && dist_->type == DistType::DISTRIBUTED;
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::is_replicated() const
{
  return dist_ && dist_->type == DistType::REPLICATED;
}

template <typename T, typename StoragePolicy>
const lattice& ndarray<T, StoragePolicy>::global_lattice() const
{
  if (!is_distributed()) {
    throw std::runtime_error("Array is not distributed");
  }
  return dist_->global_lattice_;
}

template <typename T, typename StoragePolicy>
const lattice& ndarray<T, StoragePolicy>::local_core() const
{
  if (!is_distributed()) {
    throw std::runtime_error("Array is not distributed");
  }
  return dist_->local_core_;
}

template <typename T, typename StoragePolicy>
const lattice& ndarray<T, StoragePolicy>::local_extent() const
{
  if (!is_distributed()) {
    throw std::runtime_error("Array is not distributed");
  }
  return dist_->local_extent_;
}

template <typename T, typename StoragePolicy>
MPI_Comm ndarray<T, StoragePolicy>::comm() const
{
  if (!dist_) {
    throw std::runtime_error("Array has no MPI configuration");
  }
  return dist_->comm;
}

template <typename T, typename StoragePolicy>
int ndarray<T, StoragePolicy>::rank() const
{
  if (!dist_) return 0;
  return dist_->rank;
}

template <typename T, typename StoragePolicy>
int ndarray<T, StoragePolicy>::nprocs() const
{
  if (!dist_) return 1;
  return dist_->nprocs;
}

template <typename T, typename StoragePolicy>
std::vector<size_t> ndarray<T, StoragePolicy>::global_to_local(
  const std::vector<size_t>& global_idx) const
{
  if (!is_distributed()) {
    throw std::runtime_error("Array is not distributed");
  }

  std::vector<size_t> local_idx(global_idx.size());
  for (size_t d = 0; d < global_idx.size(); d++) {
    if (global_idx[d] < dist_->local_core_.start(d) ||
        global_idx[d] >= dist_->local_core_.start(d) + dist_->local_core_.size(d)) {
      throw std::out_of_range("Global index not in local core region");
    }
    local_idx[d] = global_idx[d] - dist_->local_core_.start(d) +
                   (dist_->local_extent_.start(d) - dist_->local_core_.start(d));
  }
  return local_idx;
}

template <typename T, typename StoragePolicy>
std::vector<size_t> ndarray<T, StoragePolicy>::local_to_global(
  const std::vector<size_t>& local_idx) const
{
  if (!is_distributed()) {
    throw std::runtime_error("Array is not distributed");
  }

  std::vector<size_t> global_idx(local_idx.size());
  for (size_t d = 0; d < local_idx.size(); d++) {
    global_idx[d] = local_idx[d] + dist_->local_core_.start(d) -
                    (dist_->local_extent_.start(d) - dist_->local_core_.start(d));
  }
  return global_idx;
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::is_local(const std::vector<size_t>& global_idx) const
{
  if (!is_distributed()) return true;

  for (size_t d = 0; d < global_idx.size(); d++) {
    if (global_idx[d] < dist_->local_core_.start(d) ||
        global_idx[d] >= dist_->local_core_.start(d) + dist_->local_core_.size(d)) {
      return false;
    }
  }
  return true;
}

// Global index access convenience methods

// 1D Fortran-order (at_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::at_global(size_t i0) {
  auto local_idx = global_to_local({i0});
  return this->f(local_idx[0]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::at_global(size_t i0) const {
  auto local_idx = global_to_local({i0});
  return this->f(local_idx[0]);
}

// 2D Fortran-order (at_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1) {
  auto local_idx = global_to_local({i0, i1});
  return this->f(local_idx[0], local_idx[1]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1) const {
  auto local_idx = global_to_local({i0, i1});
  return this->f(local_idx[0], local_idx[1]);
}

// 3D Fortran-order (at_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1, size_t i2) {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->f(local_idx[0], local_idx[1], local_idx[2]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1, size_t i2) const {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->f(local_idx[0], local_idx[1], local_idx[2]);
}

// 4D Fortran-order (at_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1, size_t i2, size_t i3) {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::at_global(size_t i0, size_t i1, size_t i2, size_t i3) const {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

// 1D Fortran-order (f_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::f_global(size_t i0) {
  auto local_idx = global_to_local({i0});
  return this->f(local_idx[0]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::f_global(size_t i0) const {
  auto local_idx = global_to_local({i0});
  return this->f(local_idx[0]);
}

// 2D Fortran-order (f_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1) {
  auto local_idx = global_to_local({i0, i1});
  return this->f(local_idx[0], local_idx[1]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1) const {
  auto local_idx = global_to_local({i0, i1});
  return this->f(local_idx[0], local_idx[1]);
}

// 3D Fortran-order (f_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1, size_t i2) {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->f(local_idx[0], local_idx[1], local_idx[2]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1, size_t i2) const {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->f(local_idx[0], local_idx[1], local_idx[2]);
}

// 4D Fortran-order (f_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1, size_t i2, size_t i3) {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::f_global(size_t i0, size_t i1, size_t i2, size_t i3) const {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

// 1D C-order (c_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::c_global(size_t i0) {
  auto local_idx = global_to_local({i0});
  return this->c(local_idx[0]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::c_global(size_t i0) const {
  auto local_idx = global_to_local({i0});
  return this->c(local_idx[0]);
}

// 2D C-order (c_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1) {
  auto local_idx = global_to_local({i0, i1});
  return this->c(local_idx[0], local_idx[1]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1) const {
  auto local_idx = global_to_local({i0, i1});
  return this->c(local_idx[0], local_idx[1]);
}

// 3D C-order (c_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1, size_t i2) {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->c(local_idx[0], local_idx[1], local_idx[2]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1, size_t i2) const {
  auto local_idx = global_to_local({i0, i1, i2});
  return this->c(local_idx[0], local_idx[1], local_idx[2]);
}

// 4D C-order (c_global)
template <typename T, typename StoragePolicy>
T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1, size_t i2, size_t i3) {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->c(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

template <typename T, typename StoragePolicy>
const T& ndarray<T, StoragePolicy>::c_global(size_t i0, size_t i1, size_t i2, size_t i3) const {
  auto local_idx = global_to_local({i0, i1, i2, i3});
  return this->c(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::setup_ghost_exchange()
{
  if (!is_distributed()) return;

  const int ndims = static_cast<int>(dist_->local_core_.nd());

  dist_->neighbors_.clear();
  dist_->neighbors_identified_ = false;

  // Identify neighbors in each dimension
  for (int dim = 0; dim < ndims; dim++) {
    // Skip if no ghost layers in this dimension
    const size_t ghost_width = dist_->ghost_widths_[dim];
    if (ghost_width == 0) continue;

    // Skip non-decomposed dimensions
    if (!dist_->decomp_pattern_.empty() &&
        static_cast<size_t>(dim) < dist_->decomp_pattern_.size() &&
        dist_->decomp_pattern_[dim] == 0) {
      continue;  // This dimension is not decomposed, no neighbors
    }

    // Check left neighbor (lower index in this dimension)
    if (dist_->local_core_.start(dim) > 0) {
      // There is a neighbor on the left
      std::vector<size_t> neighbor_point(ndims);
      for (int d = 0; d < ndims; d++) {
        if (d == dim) {
          neighbor_point[d] = dist_->local_core_.start(d) - 1;
        } else {
          neighbor_point[d] = dist_->local_core_.start(d);
        }
      }

      // Find which rank owns this point
      int neighbor_rank = -1;
      for (size_t p = 0; p < dist_->partitioner_->np(); p++) {
        const auto& core = dist_->partitioner_->get_core(p);
        if (core.contains(neighbor_point)) {
          neighbor_rank = static_cast<int>(p);
          break;
        }
      }

      if (neighbor_rank >= 0 && neighbor_rank != dist_->rank) {
        typename distribution_info::Neighbor neighbor;
        neighbor.rank = neighbor_rank;
        neighbor.direction = dim * 2;  // 0=left in dim 0, 2=left in dim 1, etc.

        // Calculate number of elements in the boundary face
        // Use core size in perpendicular dimensions (corners handled separately in multi-pass)
        size_t face_size = 1;
        for (int d = 0; d < ndims; d++) {
          if (d == dim) {
            face_size *= ghost_width;
          } else {
            face_size *= dist_->local_core_.size(d);
          }
        }

        neighbor.send_count = face_size;
        neighbor.recv_count = face_size;

        dist_->neighbors_.push_back(neighbor);
      }
    }

    // Check right neighbor (higher index in this dimension)
    size_t core_end = dist_->local_core_.start(dim) + dist_->local_core_.size(dim);
    size_t global_end = dist_->global_lattice_.start(dim) + dist_->global_lattice_.size(dim);
    if (core_end < global_end) {
      // There is a neighbor on the right
      std::vector<size_t> neighbor_point(ndims);
      for (int d = 0; d < ndims; d++) {
        if (d == dim) {
          neighbor_point[d] = core_end;
        } else {
          neighbor_point[d] = dist_->local_core_.start(d);
        }
      }

      // Find which rank owns this point
      int neighbor_rank = -1;
      for (size_t p = 0; p < dist_->partitioner_->np(); p++) {
        const auto& core = dist_->partitioner_->get_core(p);
        if (core.contains(neighbor_point)) {
          neighbor_rank = static_cast<int>(p);
          break;
        }
      }

      if (neighbor_rank >= 0 && neighbor_rank != dist_->rank) {
        typename distribution_info::Neighbor neighbor;
        neighbor.rank = neighbor_rank;
        neighbor.direction = dim * 2 + 1;  // 1=right in dim 0, 3=right in dim 1, etc.

        // Use core size in perpendicular dimensions (corners handled separately)
        size_t face_size = 1;
        for (int d = 0; d < ndims; d++) {
          if (d == dim) {
            face_size *= ghost_width;
          } else {
            face_size *= dist_->local_core_.size(d);
          }
        }

        neighbor.send_count = face_size;
        neighbor.recv_count = face_size;

        dist_->neighbors_.push_back(neighbor);
      }
    }
  }

  dist_->neighbors_identified_ = true;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::exchange_ghosts()
{
  if (!is_distributed()) return;  // No-op for serial or replicated arrays

  if (!dist_->neighbors_identified_ || dist_->neighbors_.empty()) {
    // No neighbors or not yet identified - nothing to exchange
    return;
  }

  // Route to appropriate implementation based on device state
  if (is_on_device()) {
    // GPU path
#if NDARRAY_HAVE_CUDA
    if (get_device_type() == NDARRAY_DEVICE_CUDA) {
      // Check environment variable to force host staging
      const char* force_staging = std::getenv("NDARRAY_FORCE_HOST_STAGING");
#ifdef __CUDACC__
      // GPU-direct path only available when compiled with nvcc
      if (force_staging && std::string(force_staging) == "1") {
        exchange_ghosts_gpu_staged();
      } else if (has_gpu_aware_mpi()) {
        exchange_ghosts_gpu_direct();
      } else {
        exchange_ghosts_gpu_staged();
      }
#else
      // When not compiled with nvcc, always use staged approach
      exchange_ghosts_gpu_staged();
#endif
      return;
    }
#endif
    // For other device types (SYCL, HIP), fall back to staged for now
    exchange_ghosts_gpu_staged();
  } else {
    // CPU path
    exchange_ghosts_cpu();
  }
}

template <typename T, typename StoragePolicy>
bool ndarray<T, StoragePolicy>::has_gpu_aware_mpi() const
{
#if NDARRAY_HAVE_CUDA
  // Check if GPU-aware MPI detection is disabled
  const char* disable = std::getenv("NDARRAY_DISABLE_GPU_AWARE_MPI");
  if (disable && std::string(disable) == "1") {
    return false;
  }

  // Method 1: Compile-time detection (MPICH, MVAPICH2, recent OpenMPI)
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return true;
#elif defined(MPIX_Query_cuda_support)
  // Method 2: Runtime query (some MPI implementations)
  return MPIX_Query_cuda_support() == 1;
#else
  // Method 3: Environment variable detection (fallback)
  // MPICH/MVAPICH2
  const char* mpich = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
  if (mpich && std::string(mpich) == "1") return true;

  // OpenMPI
  const char* ompi = std::getenv("OMPI_MCA_opal_cuda_support");
  if (ompi && std::string(ompi) == "true") return true;

  // Cray MPI
  const char* cray = std::getenv("CRAY_CUDA_MPS");
  if (cray && std::string(cray) == "1") return true;

  // Conservative default: assume not available
  return false;
#endif
#else
  return false;  // No CUDA support compiled in
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::exchange_ghosts_cpu()
{
  // Multi-pass exchange to fill faces, edges, and corners:
  // Pass 0: Fill face ghosts (1D neighbors)
  // Pass 1: Fill edge ghosts (2D neighbors, using filled faces)
  // Pass 2: Fill corner/vertex ghosts (3D neighbors, using filled edges)
  // For N-D arrays, need N passes to reach diagonal corners

  int ndims = static_cast<int>(dims.size());
  int num_passes = ndims;  // N dimensions needs N passes for full corner propagation

  for (int pass = 0; pass < num_passes; pass++) {
    std::vector<MPI_Request> requests;
    std::vector<std::vector<T>> send_buffers(dist_->neighbors_.size());
    std::vector<std::vector<T>> recv_buffers(dist_->neighbors_.size());

    // Calculate buffer sizes for this pass
    std::vector<size_t> buffer_sizes(dist_->neighbors_.size());
    for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
      buffer_sizes[i] = calculate_buffer_size(i, pass);
    }

    // Post receives for all neighbors
    for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
      recv_buffers[i].resize(buffer_sizes[i]);

      MPI_Request req;
      int tag = dist_->neighbors_[i].direction * 10 + pass;  // Different tag per pass
      MPI_Irecv(recv_buffers[i].data(),
                static_cast<int>(buffer_sizes[i]),
                mpi_datatype(),
                dist_->neighbors_[i].rank,
                tag,
                dist_->comm,
                &req);
      requests.push_back(req);
    }

    // Pack and send boundary data to all neighbors
    for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
      send_buffers[i].resize(buffer_sizes[i]);
      pack_boundary_data(i, send_buffers[i], pass);

      // Reverse the direction for the tag (what we receive from left, they send from right)
      int tag = (dist_->neighbors_[i].direction ^ 1) * 10 + pass;  // Different tag per pass

      MPI_Send(send_buffers[i].data(),
               static_cast<int>(buffer_sizes[i]),
               mpi_datatype(),
               dist_->neighbors_[i].rank,
               tag,
               dist_->comm);
    }

    // Wait for all receives to complete
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received ghost data
    for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
      unpack_ghost_data(i, recv_buffers[i], pass);
    }
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::exchange_ghosts_gpu_staged()
{
  // Fallback: Stage through host memory
  // Strategy: GPU → host → MPI exchange → GPU

  // 1. Copy full array from device to host
  copy_from_device();

  // 2. Perform exchange on host
  exchange_ghosts_cpu();

  // 3. Copy back to device (use same device type and ID as before)
  copy_to_device(device_type, device_id);
}

#if NDARRAY_HAVE_CUDA && defined(__CUDACC__)
template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::exchange_ghosts_gpu_direct()
{
  // GPU-aware MPI: Pass device pointers directly to MPI
  // Use CUDA kernels for pack/unpack operations

  // Allocate device buffers for send/recv
  std::vector<T*> d_send_buffers(dist_->neighbors_.size());
  std::vector<T*> d_recv_buffers(dist_->neighbors_.size());

  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    CUDA_CHECK(cudaMalloc(&d_send_buffers[i],
                          dist_->neighbors_[i].send_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_recv_buffers[i],
                          dist_->neighbors_[i].recv_count * sizeof(T)));
  }

  T* d_array = static_cast<T*>(devptr_.get());

  // Pack boundary data on device using CUDA kernels
  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    const auto& neighbor = dist_->neighbors_[i];
    int dim = neighbor.direction / 2;
    bool is_high = (neighbor.direction % 2) == 1;
    size_t ghost_width = dist_->ghost_widths_[dim];

    if (dims.size() == 1) {
      // 1D case
      launch_pack_boundary_1d(
          d_send_buffers[i],
          d_array,
          static_cast<int>(dims[0]),
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(dist_->local_core_.size(0)));
    } else if (dims.size() == 2 && dim < 2) {
      // 2D case
      launch_pack_boundary_2d(
          d_send_buffers[i],
          d_array,
          static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          dim,
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(dist_->local_core_.size(0)),
          static_cast<int>(dist_->local_core_.size(1)));
    } else if (dims.size() == 3 && dim < 3) {
      // 3D case
      launch_pack_boundary_3d(
          d_send_buffers[i],
          d_array,
          static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          static_cast<int>(dims[2]),
          dim,
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(dist_->local_core_.size(0)),
          static_cast<int>(dist_->local_core_.size(1)),
          static_cast<int>(dist_->local_core_.size(2)));
    }
  }

  // Synchronize before MPI operations
  CUDA_CHECK(cudaDeviceSynchronize());

  // Post non-blocking receives with device pointers
  std::vector<MPI_Request> requests;
  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    MPI_Request req;
    int tag = dist_->neighbors_[i].direction;
    MPI_Irecv(d_recv_buffers[i],
              static_cast<int>(dist_->neighbors_[i].recv_count),
              mpi_datatype(),
              dist_->neighbors_[i].rank,
              tag,
              dist_->comm,
              &req);
    requests.push_back(req);
  }

  // Send boundary data with device pointers
  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    int tag = dist_->neighbors_[i].direction ^ 1;
    MPI_Send(d_send_buffers[i],
             static_cast<int>(dist_->neighbors_[i].send_count),
             mpi_datatype(),
             dist_->neighbors_[i].rank,
             tag,
             dist_->comm);
  }

  // Wait for all receives to complete
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Unpack ghost data on device using CUDA kernels
  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    const auto& neighbor = dist_->neighbors_[i];
    int dim = neighbor.direction / 2;
    bool is_high = (neighbor.direction % 2) == 1;
    size_t ghost_width = dist_->ghost_widths_[dim];

    // Calculate ghost offsets
    size_t ghost_low = dist_->local_core_.start(dim) - dist_->local_extent_.start(dim);
    size_t ghost_high = (dist_->local_extent_.start(dim) + dist_->local_extent_.size(dim)) -
                        (dist_->local_core_.start(dim) + dist_->local_core_.size(dim));

    if (dims.size() == 1) {
      // 1D case
      launch_unpack_ghost_1d(
          d_array,
          d_recv_buffers[i],
          static_cast<int>(dims[0]),
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(ghost_low),
          static_cast<int>(ghost_high),
          static_cast<int>(dist_->local_core_.size(0)));
    } else if (dims.size() == 2 && dim < 2) {
      // 2D case
      launch_unpack_ghost_2d(
          d_array,
          d_recv_buffers[i],
          static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          dim,
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(ghost_low),
          static_cast<int>(ghost_high),
          static_cast<int>(dist_->local_core_.size(0)),
          static_cast<int>(dist_->local_core_.size(1)));
    } else if (dims.size() == 3 && dim < 3) {
      // 3D case
      launch_unpack_ghost_3d(
          d_array,
          d_recv_buffers[i],
          static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          static_cast<int>(dims[2]),
          dim,
          is_high,
          static_cast<int>(ghost_width),
          static_cast<int>(ghost_low),
          static_cast<int>(ghost_high),
          static_cast<int>(dist_->local_core_.size(0)),
          static_cast<int>(dist_->local_core_.size(1)),
          static_cast<int>(dist_->local_core_.size(2)));
    }
  }

  // Synchronize after unpacking
  CUDA_CHECK(cudaDeviceSynchronize());

  // Free device buffers
  for (size_t i = 0; i < dist_->neighbors_.size(); i++) {
    CUDA_CHECK(cudaFree(d_send_buffers[i]));
    CUDA_CHECK(cudaFree(d_recv_buffers[i]));
  }
}
#elif NDARRAY_HAVE_CUDA
// Stub implementation when CUDA is available but not compiling with nvcc
template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::exchange_ghosts_gpu_direct()
{
  // This should never be called since the call site is guarded with __CUDACC__
  // But we need a definition to satisfy the linker
  fatal("exchange_ghosts_gpu_direct requires compilation with nvcc");
}
#endif

template <typename T, typename StoragePolicy>
size_t ndarray<T, StoragePolicy>::calculate_buffer_size(int neighbor_idx, int pass)
{
  const auto& neighbor = dist_->neighbors_[neighbor_idx];
  int dim = neighbor.direction / 2;
  const int ndims = static_cast<int>(dims.size());

  size_t ghost_width = dist_->ghost_widths_[dim];

  size_t buffer_size = ghost_width;
  for (int d = 0; d < ndims; d++) {
    if (d != dim) {
      // In pass 0: use core size (faces only)
      // In pass 1: use extent size (includes corners from filled ghosts)
      if (pass == 0) {
        buffer_size *= dist_->local_core_.size(d);
      } else {
        buffer_size *= dist_->local_extent_.size(d);
      }
    }
  }

  return buffer_size;
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::pack_boundary_data(int neighbor_idx, std::vector<T>& buffer, int pass)
{
  const auto& neighbor = dist_->neighbors_[neighbor_idx];
  int dim = neighbor.direction / 2;  // Which dimension: 0, 1, 2, ...
  bool is_high = (neighbor.direction % 2) == 1;  // true = right/up, false = left/down

  size_t ghost_width = dist_->ghost_widths_[dim];

  // Calculate ghost offset
  size_t ghost_offset_0 = dist_->local_core_.start(0) - dist_->local_extent_.start(0);

  if (dim == 0 && dims.size() >= 1) {
    // Boundary in dimension 0
    size_t start_idx = is_high ? (dist_->local_core_.size(0) - ghost_width) : 0;
    start_idx += ghost_offset_0;  // Account for ghost offset!
    size_t buffer_idx = 0;

    if (dims.size() == 1) {
      // 1D case
      for (size_t i = 0; i < ghost_width; i++) {
        buffer[buffer_idx++] = f(start_idx + i);
      }
    } else if (dims.size() == 2) {
      // 2D case
      size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
      // Pass 0: use core size (faces only), Pass 1: use extent size (includes corners)
      size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
      size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;

      for (size_t i = 0; i < ghost_width; i++) {
        for (size_t j = 0; j < dim1_size; j++) {
          buffer[buffer_idx++] = f(start_idx + i, dim1_start + j);
        }
      }
    } else if (dims.size() >= 3) {
      // 3D case
      size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
      size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
      size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
      size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;
      size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
      size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

      for (size_t i = 0; i < ghost_width; i++) {
        for (size_t j = 0; j < dim1_size; j++) {
          for (size_t k = 0; k < dim2_size; k++) {
            buffer[buffer_idx++] = f(start_idx + i, dim1_start + j, dim2_start + k);
          }
        }
      }
    }
  } else if (dim == 1 && dims.size() >= 2) {
    // Boundary in dimension 1
    size_t ghost_offset_0 = dist_->local_core_.start(0) - dist_->local_extent_.start(0);
    size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
    size_t start_idx = is_high ? (dist_->local_core_.size(1) - ghost_width) : 0;
    start_idx += ghost_offset_1;  // Account for ghost offset!
    size_t buffer_idx = 0;

    // Pass 0: use core size (faces only), Pass 1: use extent size (includes corners)
    size_t dim0_size = (pass == 0) ? dist_->local_core_.size(0) : dist_->local_extent_.size(0);
    size_t dim0_start = (pass == 0) ? ghost_offset_0 : 0;

    if (dims.size() == 2) {
      // 2D case
      for (size_t i = 0; i < dim0_size; i++) {
        for (size_t j = 0; j < ghost_width; j++) {
          buffer[buffer_idx++] = f(dim0_start + i, start_idx + j);
        }
      }
    } else if (dims.size() >= 3) {
      // 3D case
      size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
      size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
      size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

      for (size_t i = 0; i < dim0_size; i++) {
        for (size_t j = 0; j < ghost_width; j++) {
          for (size_t k = 0; k < dim2_size; k++) {
            buffer[buffer_idx++] = f(dim0_start + i, start_idx + j, dim2_start + k);
          }
        }
      }
    }
  } else if (dim == 2 && dims.size() >= 3) {
    // Boundary in dimension 2 (3D arrays)
    size_t ghost_offset_0 = dist_->local_core_.start(0) - dist_->local_extent_.start(0);
    size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
    size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
    size_t start_idx = is_high ? (dist_->local_core_.size(2) - ghost_width) : 0;
    start_idx += ghost_offset_2;  // Account for ghost offset!
    size_t buffer_idx = 0;

    // Pass 0: use core size (faces only), Pass 1: use extent size (includes edges/corners)
    size_t dim0_size = (pass == 0) ? dist_->local_core_.size(0) : dist_->local_extent_.size(0);
    size_t dim0_start = (pass == 0) ? ghost_offset_0 : 0;
    size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
    size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;

    for (size_t i = 0; i < dim0_size; i++) {
      for (size_t j = 0; j < dim1_size; j++) {
        for (size_t k = 0; k < ghost_width; k++) {
          buffer[buffer_idx++] = f(dim0_start + i, dim1_start + j, start_idx + k);
        }
      }
    }
  }
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::unpack_ghost_data(int neighbor_idx, const std::vector<T>& buffer, int pass)
{
  const auto& neighbor = dist_->neighbors_[neighbor_idx];
  int dim = neighbor.direction / 2;
  bool is_high = (neighbor.direction % 2) == 1;

  size_t ghost_width = dist_->ghost_widths_[dim];

  // Calculate ghost offsets
  size_t ghost_low = dist_->local_core_.start(dim) - dist_->local_extent_.start(dim);
  size_t ghost_high = (dist_->local_extent_.start(dim) + dist_->local_extent_.size(dim)) -
                      (dist_->local_core_.start(dim) + dist_->local_core_.size(dim));

  if (dim == 0 && dims.size() >= 1) {
    size_t start_idx = is_high ? (dist_->local_core_.size(0) + ghost_low) : 0;
    size_t buffer_idx = 0;

    if (!is_high && ghost_low > 0) {
      // Unpack into left ghost
      if (dims.size() == 1) {
        for (size_t i = 0; i < ghost_width && i < ghost_low; i++) {
          f(start_idx + i) = buffer[buffer_idx++];
        }
      } else if (dims.size() == 2) {
        size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
        // Pass 0: unpack core-sized face, Pass 1: unpack extent-sized face (with corners)
        size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
        size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;

        for (size_t i = 0; i < ghost_width && i < ghost_low; i++) {
          for (size_t j = 0; j < dim1_size; j++) {
            f(start_idx + i, dim1_start + j) = buffer[buffer_idx++];
          }
        }
      } else if (dims.size() >= 3) {
        size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
        size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
        size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
        size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;
        size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
        size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

        for (size_t i = 0; i < ghost_width && i < ghost_low; i++) {
          for (size_t j = 0; j < dim1_size; j++) {
            for (size_t k = 0; k < dim2_size; k++) {
              f(start_idx + i, dim1_start + j, dim2_start + k) = buffer[buffer_idx++];
            }
          }
        }
      }
    } else if (is_high && ghost_high > 0) {
      // Unpack into right ghost
      if (dims.size() == 1) {
        for (size_t i = 0; i < ghost_width && i < ghost_high; i++) {
          f(start_idx + i) = buffer[buffer_idx++];
        }
      } else if (dims.size() == 2) {
        size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
        // Pass 0: unpack core-sized face, Pass 1: unpack extent-sized face (with corners)
        size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
        size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;

        for (size_t i = 0; i < ghost_width && i < ghost_high; i++) {
          for (size_t j = 0; j < dim1_size; j++) {
            f(start_idx + i, dim1_start + j) = buffer[buffer_idx++];
          }
        }
      } else if (dims.size() >= 3) {
        size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
        size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
        size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
        size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;
        size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
        size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

        for (size_t i = 0; i < ghost_width && i < ghost_high; i++) {
          for (size_t j = 0; j < dim1_size; j++) {
            for (size_t k = 0; k < dim2_size; k++) {
              f(start_idx + i, dim1_start + j, dim2_start + k) = buffer[buffer_idx++];
            }
          }
        }
      }
    }
  } else if (dim == 1 && dims.size() >= 2) {
    size_t ghost_offset_0 = dist_->local_core_.start(0) - dist_->local_extent_.start(0);
    size_t start_idx = is_high ? (dist_->local_core_.size(1) + ghost_low) : 0;
    size_t buffer_idx = 0;

    // Pass 0: use core size (faces only), Pass 1: use extent size (includes corners)
    size_t dim0_size = (pass == 0) ? dist_->local_core_.size(0) : dist_->local_extent_.size(0);
    size_t dim0_start = (pass == 0) ? ghost_offset_0 : 0;

    if (!is_high && ghost_low > 0) {
      if (dims.size() == 2) {
        for (size_t i = 0; i < dim0_size; i++) {
          for (size_t j = 0; j < ghost_width && j < ghost_low; j++) {
            f(dim0_start + i, start_idx + j) = buffer[buffer_idx++];
          }
        }
      } else if (dims.size() >= 3) {
        size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
        size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
        size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

        for (size_t i = 0; i < dim0_size; i++) {
          for (size_t j = 0; j < ghost_width && j < ghost_low; j++) {
            for (size_t k = 0; k < dim2_size; k++) {
              f(dim0_start + i, start_idx + j, dim2_start + k) = buffer[buffer_idx++];
            }
          }
        }
      }
    } else if (is_high && ghost_high > 0) {
      if (dims.size() == 2) {
        for (size_t i = 0; i < dim0_size; i++) {
          for (size_t j = 0; j < ghost_width && j < ghost_high; j++) {
            f(dim0_start + i, start_idx + j) = buffer[buffer_idx++];
          }
        }
      } else if (dims.size() >= 3) {
        size_t ghost_offset_2 = dist_->local_core_.start(2) - dist_->local_extent_.start(2);
        size_t dim2_size = (pass == 0) ? dist_->local_core_.size(2) : dist_->local_extent_.size(2);
        size_t dim2_start = (pass == 0) ? ghost_offset_2 : 0;

        for (size_t i = 0; i < dim0_size; i++) {
          for (size_t j = 0; j < ghost_width && j < ghost_high; j++) {
            for (size_t k = 0; k < dim2_size; k++) {
              f(dim0_start + i, start_idx + j, dim2_start + k) = buffer[buffer_idx++];
            }
          }
        }
      }
    }
  } else if (dim == 2 && dims.size() >= 3) {
    size_t ghost_offset_0 = dist_->local_core_.start(0) - dist_->local_extent_.start(0);
    size_t ghost_offset_1 = dist_->local_core_.start(1) - dist_->local_extent_.start(1);
    size_t start_idx = is_high ? (dist_->local_core_.size(2) + ghost_low) : 0;
    size_t buffer_idx = 0;

    // Pass 0: use core size (faces only), Pass 1: use extent size (includes edges/corners)
    size_t dim0_size = (pass == 0) ? dist_->local_core_.size(0) : dist_->local_extent_.size(0);
    size_t dim0_start = (pass == 0) ? ghost_offset_0 : 0;
    size_t dim1_size = (pass == 0) ? dist_->local_core_.size(1) : dist_->local_extent_.size(1);
    size_t dim1_start = (pass == 0) ? ghost_offset_1 : 0;

    if (!is_high && ghost_low > 0) {
      for (size_t i = 0; i < dim0_size; i++) {
        for (size_t j = 0; j < dim1_size; j++) {
          for (size_t k = 0; k < ghost_width && k < ghost_low; k++) {
            f(dim0_start + i, dim1_start + j, start_idx + k) = buffer[buffer_idx++];
          }
        }
      }
    } else if (is_high && ghost_high > 0) {
      for (size_t i = 0; i < dim0_size; i++) {
        for (size_t j = 0; j < dim1_size; j++) {
          for (size_t k = 0; k < ghost_width && k < ghost_high; k++) {
            f(dim0_start + i, dim1_start + j, start_idx + k) = buffer[buffer_idx++];
          }
        }
      }
    }
  }
}

template <typename T, typename StoragePolicy>
MPI_Datatype ndarray<T, StoragePolicy>::mpi_datatype() const
{
  return mpi_dtype();  // Use existing static method
}

#endif // NDARRAY_HAVE_MPI

//////////////////////////////////
// Distribution-aware I/O implementations
//////////////////////////////////

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_NETCDF

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_netcdf_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_NETCDF
  // Internal helper to check for time dimension
  auto check_time = [&](int ncid, int varid) {
    int unlimid;
    nc_inq_unlimdim(ncid, &unlimid);
    if (unlimid >= 0) {
      int ndims, dimids[NC_MAX_VAR_DIMS];
      nc_inq_varndims(ncid, varid, &ndims);
      nc_inq_vardimid(ncid, varid, dimids);
      for (int d = 0; d < ndims; d++) {
        if (dimids[d] == unlimid) return true;
      }
    }
    return false;
  };
#endif

#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel read using local core region
    const auto& core = dist_->local_core_;
    std::vector<size_t> starts(core.nd());
    std::vector<size_t> sizes(core.nd());

    for (size_t d = 0; d < core.nd(); d++) {
      starts[d] = core.start(d);
      sizes[d] = core.size(d);
    }

    // Use base class parallel read with start/size
    this->read_netcdf(filename, varname, starts.data(), sizes.data(), dist_->comm);

#if NDARRAY_HAVE_NETCDF
    // Detection after read (simplified - in practice, we might need to open the file again or trust read_netcdf)
    // For now, let's just use the base read
#endif

  } else if (should_use_replicated_io()) {
    // Replicated mode: Rank 0 reads, broadcast to others
    if (dist_->rank == 0) {
      // Rank 0 reads full array
      this->read_netcdf(filename, varname, MPI_COMM_SELF);
    }

    // Broadcast size from rank 0
    size_t total_size = this->size();
    MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, dist_->comm);

    // Other ranks allocate
    if (dist_->rank != 0) {
      this->reshapef(this->dims);  // Dims should already be set
    }

    // Broadcast data
    MPI_Bcast(this->data(), static_cast<int>(total_size), mpi_datatype(), 0, dist_->comm);
    
    // Propagate flags
    size_t flags[2] = {this->n_component_dims, (size_t)this->is_time_varying};
    MPI_Bcast(flags, 2, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) {
      this->n_component_dims = flags[0];
      this->is_time_varying = (bool)flags[1];
    }

  } else {
    // Serial mode: Regular read
    this->read_netcdf(filename, varname, MPI_COMM_SELF);
  }
#else
  // MPI not available, only serial mode
  this->read_netcdf(filename, varname, MPI_COMM_SELF);
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::write_netcdf_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel write using local core region
    const auto& core = dist_->local_core_;
    std::vector<size_t> starts(core.nd());
    std::vector<size_t> sizes(core.nd());

    for (size_t d = 0; d < core.nd(); d++) {
      starts[d] = core.start(d);
      sizes[d] = core.size(d);
    }

    // Create file on rank 0
    if (dist_->rank == 0) {
      int ncid;
      NC_SAFE_CALL(nc_create(filename.c_str(), NC_NETCDF4, &ncid));

      // Define dimensions
      std::vector<int> dimids(core.nd());
      for (size_t d = 0; d < core.nd(); d++) {
        std::string dimname = "dim" + std::to_string(d);
        NC_SAFE_CALL(nc_def_dim(ncid, dimname.c_str(), dist_->global_lattice_.size(d), &dimids[d]));
      }

      // Define variable
      int varid;
      NC_SAFE_CALL(nc_def_var(ncid, varname.c_str(), this->nc_dtype(), core.nd(), dimids.data(), &varid));
      NC_SAFE_CALL(nc_enddef(ncid));
      NC_SAFE_CALL(nc_close(ncid));
    }

    MPI_Barrier(dist_->comm);

    // All ranks write their portion
    int ncid;
#if NDARRAY_HAVE_NETCDF_PARALLEL
    NC_SAFE_CALL(nc_open_par(filename.c_str(), NC_WRITE, dist_->comm, MPI_INFO_NULL, &ncid));

    int varid;
    NC_SAFE_CALL(nc_inq_varid(ncid, varname.c_str(), &varid));

    this->to_netcdf(ncid, varid, starts.data(), sizes.data());

    NC_SAFE_CALL(nc_close(ncid));
#else
    fatal("Parallel NetCDF write requires NetCDF built with MPI support");
#endif

  } else if (should_use_replicated_io()) {
    // Replicated mode: Only rank 0 writes
    if (dist_->rank == 0) {
      int ncid;
      NC_SAFE_CALL(nc_create(filename.c_str(), NC_NETCDF4, &ncid));

      // Define dimensions
      std::vector<int> dimids(this->nd());
      for (size_t d = 0; d < this->nd(); d++) {
        std::string dimname = "dim" + std::to_string(d);
        NC_SAFE_CALL(nc_def_dim(ncid, dimname.c_str(), this->dimf(d), &dimids[d]));
      }

      // Define variable
      int varid;
      NC_SAFE_CALL(nc_def_var(ncid, varname.c_str(), this->nc_dtype(), this->nd(), dimids.data(), &varid));
      NC_SAFE_CALL(nc_enddef(ncid));

      // Write data
      this->to_netcdf(ncid, varid);

      NC_SAFE_CALL(nc_close(ncid));
    }
    MPI_Barrier(dist_->comm);

  } else {
    // Serial mode: Regular write
    int ncid;
    NC_SAFE_CALL(nc_create(filename.c_str(), NC_NETCDF4, &ncid));

    // Define dimensions
    std::vector<int> dimids(this->nd());
    for (size_t d = 0; d < this->nd(); d++) {
      std::string dimname = "dim" + std::to_string(d);
      NC_SAFE_CALL(nc_def_dim(ncid, dimname.c_str(), this->dimf(d), &dimids[d]));
    }

    // Define variable
    int varid;
    NC_SAFE_CALL(nc_def_var(ncid, varname.c_str(), this->nc_dtype(), this->nd(), dimids.data(), &varid));
    NC_SAFE_CALL(nc_enddef(ncid));

    // Write data
    this->to_netcdf(ncid, varid);

    NC_SAFE_CALL(nc_close(ncid));
  }
#else
  // MPI not available, only serial mode (to be implemented if needed, currently falls back to fatal error via virtual interface)
  // Actually we should probably implement serial NetCDF write here too for completeness
  int ncid;
  NC_SAFE_CALL(nc_create(filename.c_str(), NC_NETCDF4, &ncid));
  std::vector<int> dimids(this->nd());
  for (size_t d = 0; d < this->nd(); d++) {
    std::string dimname = "dim" + std::to_string(d);
    NC_SAFE_CALL(nc_def_dim(ncid, dimname.c_str(), this->dimf(d), &dimids[d]));
  }
  int varid;
  NC_SAFE_CALL(nc_def_var(ncid, varname.c_str(), this->nc_dtype(), this->nd(), dimids.data(), &varid));
  NC_SAFE_CALL(nc_enddef(ncid));
  this->to_netcdf(ncid, varid);
  NC_SAFE_CALL(nc_close(ncid));
#endif
}

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_NETCDF

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_PNETCDF

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_pnetcdf_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel PNetCDF read
    const auto& core = dist_->local_core_;
    const auto& extent = dist_->local_extent_;
    const size_t nd = core.nd();
    std::vector<MPI_Offset> starts(nd);
    std::vector<MPI_Offset> sizes(nd);

    // NetCDF uses C-order (last dim fastest), ndarray uses Fortran-order (first dim fastest)
    // Reverse indices for PNetCDF calls
    for (size_t d = 0; d < nd; d++) {
      starts[nd - 1 - d] = static_cast<MPI_Offset>(core.start(d));
      sizes[nd - 1 - d] = static_cast<MPI_Offset>(core.size(d));
    }

    int ncid, varid;
    PNC_SAFE_CALL(ncmpi_open(dist_->comm, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid));
    PNC_SAFE_CALL(ncmpi_inq_varid(ncid, varname.c_str(), &varid));

    // Calculate offset in local storage (core relative to extent)
    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = (nd >= 2) ? (core.start(1) - extent.start(1)) : 0;
    size_t off_k = (nd >= 3) ? (core.start(2) - extent.start(2)) : 0;

    if (nd == 1) {
      T* data_ptr = &this->f(off_i);
      if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_get_vara_float_all(ncid, varid, starts.data(), sizes.data(), data_ptr));
      else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_get_vara_double_all(ncid, varid, starts.data(), sizes.data(), data_ptr));
      else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_get_vara_int_all(ncid, varid, starts.data(), sizes.data(), (int*)data_ptr));
    } else if (nd == 2) {
      for (size_t j = 0; j < core.size(1); j++) {
        // In NetCDF (y, x), we iterate over y (dim 1 of ndarray)
        // starts[0] is y_start, starts[1] is x_start
        MPI_Offset st[2] = {starts[0] + (MPI_Offset)j, starts[1]};
        MPI_Offset sz[2] = {1, sizes[1]};
        T* col_ptr = &this->f(off_i, off_j + j);
        if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_get_vara_float_all(ncid, varid, st, sz, col_ptr));
        else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_get_vara_double_all(ncid, varid, st, sz, col_ptr));
        else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_get_vara_int_all(ncid, varid, st, sz, (int*)col_ptr));
      }
    } else if (nd == 3) {
      for (size_t k = 0; k < core.size(2); k++) {
        for (size_t j = 0; j < core.size(1); j++) {
          // NetCDF (z, y, x)
          MPI_Offset st[3] = {starts[0] + (MPI_Offset)k, starts[1] + (MPI_Offset)j, starts[2]};
          MPI_Offset sz[3] = {1, 1, sizes[2]};
          T* ptr = &this->f(off_i, off_j + j, off_k + k);
          if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_get_vara_float_all(ncid, varid, st, sz, ptr));
          else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_get_vara_double_all(ncid, varid, st, sz, ptr));
          else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_get_vara_int_all(ncid, varid, st, sz, (int*)ptr));
        }
      }
    } else {
      fatal("Parallel read only implemented for 1D, 2D, 3D");
    }

    PNC_SAFE_CALL(ncmpi_close(ncid));

  } else if (should_use_replicated_io()) {
    // Replicated mode: Rank 0 reads, broadcast
    if (dist_->rank == 0) {
      int ncid, varid;
      PNC_SAFE_CALL(ncmpi_open(MPI_COMM_SELF, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid));
      PNC_SAFE_CALL(ncmpi_inq_varid(ncid, varname.c_str(), &varid));

      int ndims;
      PNC_SAFE_CALL(ncmpi_inq_varndims(ncid, varid, &ndims));
      std::vector<MPI_Offset> starts(ndims, 0);
      std::vector<MPI_Offset> sizes(ndims);
      std::vector<int> dimids(ndims);
      PNC_SAFE_CALL(ncmpi_inq_vardimid(ncid, varid, dimids.data()));
      for (int d = 0; d < ndims; d++) PNC_SAFE_CALL(ncmpi_inq_dimlen(ncid, dimids[d], &sizes[d]));

      this->read_pnetcdf_all(ncid, varid, starts.data(), sizes.data());
      PNC_SAFE_CALL(ncmpi_close(ncid));
    }

    size_t total_size = this->size();
    MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) this->reshapef(this->dims);
    MPI_Bcast(this->data(), static_cast<int>(total_size), mpi_datatype(), 0, dist_->comm);

    // Propagate flags
    size_t flags[2] = {this->n_component_dims, (size_t)this->is_time_varying};
    MPI_Bcast(flags, 2, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) {
      this->n_component_dims = flags[0];
      this->is_time_varying = (bool)flags[1];
    }

  } else {
    // Serial mode
    int ncid, varid;
    PNC_SAFE_CALL(ncmpi_open(MPI_COMM_SELF, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid));
    PNC_SAFE_CALL(ncmpi_inq_varid(ncid, varname.c_str(), &varid));

    int ndims;
    PNC_SAFE_CALL(ncmpi_inq_varndims(ncid, varid, &ndims));
    std::vector<MPI_Offset> starts(ndims, 0);
    std::vector<MPI_Offset> sizes(ndims);
    std::vector<int> dimids(ndims);
    PNC_SAFE_CALL(ncmpi_inq_vardimid(ncid, varid, dimids.data()));
    for (int d = 0; d < ndims; d++) PNC_SAFE_CALL(ncmpi_inq_dimlen(ncid, dimids[d], &sizes[d]));

    this->read_pnetcdf_all(ncid, varid, starts.data(), sizes.data());
    PNC_SAFE_CALL(ncmpi_close(ncid));
  }
#else
  // MPI not available, only serial mode
  int ncid, varid;
  PNC_SAFE_CALL(ncmpi_open(MPI_COMM_SELF, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid));
  PNC_SAFE_CALL(ncmpi_inq_varid(ncid, varname.c_str(), &varid));
  int ndims;
  PNC_SAFE_CALL(ncmpi_inq_varndims(ncid, varid, &ndims));
  std::vector<MPI_Offset> starts(ndims, 0);
  std::vector<MPI_Offset> sizes(ndims);
  std::vector<int> dimids(ndims);
  PNC_SAFE_CALL(ncmpi_inq_vardimid(ncid, varid, dimids.data()));
  for (int d = 0; d < ndims; d++) PNC_SAFE_CALL(ncmpi_inq_dimlen(ncid, dimids[d], &sizes[d]));
  this->read_pnetcdf_all(ncid, varid, starts.data(), sizes.data());
  PNC_SAFE_CALL(ncmpi_close(ncid));
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::write_pnetcdf_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel PNetCDF write
    const auto& core = dist_->local_core_;
    const auto& extent = dist_->local_extent_;
    const size_t nd = core.nd();
    std::vector<MPI_Offset> starts(nd);
    std::vector<MPI_Offset> sizes(nd);

    for (size_t d = 0; d < nd; d++) {
      starts[nd - 1 - d] = static_cast<MPI_Offset>(core.start(d));
      sizes[nd - 1 - d] = static_cast<MPI_Offset>(core.size(d));
    }

    int ncid, varid;
    PNC_SAFE_CALL(ncmpi_create(dist_->comm, filename.c_str(), NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, &ncid));

    std::vector<int> dimids(nd);
    for (size_t d = 0; d < nd; d++) {
      std::string dimname = "dim" + std::to_string(d);
      // Define dims in C-order (reverse of ndarray dims)
      PNC_SAFE_CALL(ncmpi_def_dim(ncid, dimname.c_str(), dist_->global_lattice_.size(nd - 1 - d), &dimids[d]));
    }

    PNC_SAFE_CALL(ncmpi_def_var(ncid, varname.c_str(), this->pnc_dtype(), nd, dimids.data(), &varid));
    PNC_SAFE_CALL(ncmpi_enddef(ncid));

    // Handle non-contiguous layout
    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = (nd >= 2) ? (core.start(1) - extent.start(1)) : 0;
    size_t off_k = (nd >= 3) ? (core.start(2) - extent.start(2)) : 0;

    if (nd == 1) {
      const T* ptr = &this->f(off_i);
      if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_put_vara_float_all(ncid, varid, starts.data(), sizes.data(), ptr));
      else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_put_vara_double_all(ncid, varid, starts.data(), sizes.data(), ptr));
      else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_put_vara_int_all(ncid, varid, starts.data(), sizes.data(), (const int*)ptr));
    } else if (nd == 2) {
      for (size_t j = 0; j < core.size(1); j++) {
        MPI_Offset st[2] = {starts[0] + (MPI_Offset)j, starts[1]};
        MPI_Offset sz[2] = {1, sizes[1]};
        const T* ptr = &this->f(off_i, off_j + j);
        if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_put_vara_float_all(ncid, varid, st, sz, ptr));
        else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_put_vara_double_all(ncid, varid, st, sz, ptr));
        else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_put_vara_int_all(ncid, varid, st, sz, (const int*)ptr));
      }
    } else if (nd == 3) {
      for (size_t k = 0; k < core.size(2); k++) {
        for (size_t j = 0; j < core.size(1); j++) {
          MPI_Offset st[3] = {starts[0] + (MPI_Offset)k, starts[1] + (MPI_Offset)j, starts[2]};
          MPI_Offset sz[3] = {1, 1, sizes[2]};
          const T* ptr = &this->f(off_i, off_j + j, off_k + k);
          if constexpr (std::is_same_v<T, float>) PNC_SAFE_CALL(ncmpi_put_vara_float_all(ncid, varid, st, sz, ptr));
          else if constexpr (std::is_same_v<T, double>) PNC_SAFE_CALL(ncmpi_put_vara_double_all(ncid, varid, st, sz, ptr));
          else if constexpr (std::is_same_v<T, int>) PNC_SAFE_CALL(ncmpi_put_vara_int_all(ncid, varid, st, sz, (const int*)ptr));
        }
      }
    } else {
      fatal("Parallel write only implemented for 1D, 2D, 3D");
    }

    PNC_SAFE_CALL(ncmpi_close(ncid));

  } else if (should_use_replicated_io()) {
    // Replicated mode: Rank 0 writes
    if (dist_->rank == 0) {
      int ncid, varid;
      PNC_SAFE_CALL(ncmpi_create(MPI_COMM_SELF, filename.c_str(), NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, &ncid));

      std::vector<int> dimids(this->nd());
      for (size_t d = 0; d < this->nd(); d++) {
        std::string dimname = "dim" + std::to_string(d);
        PNC_SAFE_CALL(ncmpi_def_dim(ncid, dimname.c_str(), this->dimf(this->nd() - 1 - d), &dimids[d]));
      }

      PNC_SAFE_CALL(ncmpi_def_var(ncid, varname.c_str(), this->pnc_dtype(), this->nd(), dimids.data(), &varid));
      PNC_SAFE_CALL(ncmpi_enddef(ncid));

      std::vector<MPI_Offset> starts(this->nd(), 0);
      std::vector<MPI_Offset> sizes(this->nd());
      for (size_t d = 0; d < this->nd(); d++) sizes[d] = this->dimf(this->nd() - 1 - d);

      this->write_pnetcdf_all(ncid, varid, starts.data(), sizes.data());
      PNC_SAFE_CALL(ncmpi_close(ncid));
    }
    MPI_Barrier(dist_->comm);

  } else {
    // Serial mode
    int ncid, varid;
    PNC_SAFE_CALL(ncmpi_create(MPI_COMM_SELF, filename.c_str(), NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, &ncid));

    std::vector<int> dimids(this->nd());
    for (size_t d = 0; d < this->nd(); d++) {
      std::string dimname = "dim" + std::to_string(d);
      PNC_SAFE_CALL(ncmpi_def_dim(ncid, dimname.c_str(), this->dimf(this->nd() - 1 - d), &dimids[d]));
    }

    PNC_SAFE_CALL(ncmpi_def_var(ncid, varname.c_str(), this->pnc_dtype(), this->nd(), dimids.data(), &varid));
    PNC_SAFE_CALL(ncmpi_enddef(ncid));

    std::vector<MPI_Offset> starts(this->nd(), 0);
    std::vector<MPI_Offset> sizes(this->nd());
    for (size_t d = 0; d < this->nd(); d++) sizes[d] = this->dimf(this->nd() - 1 - d);

    this->write_pnetcdf_all(ncid, varid, starts.data(), sizes.data());
    PNC_SAFE_CALL(ncmpi_close(ncid));
  }
#endif
}

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_PNETCDF

#if NDARRAY_HAVE_HDF5

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_hdf5_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel HDF5 read
    // Note: Requires HDF5 built with parallel support (--enable-parallel)
#ifdef H5_HAVE_PARALLEL
    const auto& core = dist_->local_core_;
    const auto& extent = dist_->local_extent_;
    const size_t nd = core.nd();

    // Open file with MPI-IO
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, dist_->comm, MPI_INFO_NULL);
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    if (file_id < 0) throw std::runtime_error("Failed to open HDF5 file: " + filename);

    hid_t dataset_id = H5Dopen(file_id, varname.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
      H5Fclose(file_id);
      throw std::runtime_error("Failed to open HDF5 dataset: " + varname);
    }

    // Set up hyperslab (C-order, so reverse ndarray dimensions)
    std::vector<hsize_t> starts(nd);
    std::vector<hsize_t> counts(nd);
    for (size_t d = 0; d < nd; d++) {
      starts[nd - 1 - d] = core.start(d);
      counts[nd - 1 - d] = core.size(d);
    }

    hid_t file_space = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, starts.data(), NULL, counts.data(), NULL);

    // Memory space and transfer property
    hid_t mem_space = H5Screate_simple(nd, counts.data(), NULL);
    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

    // Handle non-contiguous local storage
    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = (nd >= 2) ? (core.start(1) - extent.start(1)) : 0;
    size_t off_k = (nd >= 3) ? (core.start(2) - extent.start(2)) : 0;

    if (nd == 1) {
      H5Dread(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i));
    } else if (nd == 2) {
      // Hyperslab for a single column in memory
      hsize_t m_counts[2] = {1, counts[1]}; // One row in C-order (one column in Fortran)
      H5Sclose(mem_space);
      mem_space = H5Screate_simple(2, m_counts, NULL);
      
      for (size_t j = 0; j < core.size(1); j++) {
        hsize_t st[2] = {starts[0] + j, starts[1]};
        hsize_t sz[2] = {1, counts[1]};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, st, NULL, sz, NULL);
        H5Dread(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i, off_j + j));
      }
    } else if (nd == 3) {
      hsize_t m_counts[3] = {1, 1, counts[2]};
      H5Sclose(mem_space);
      mem_space = H5Screate_simple(3, m_counts, NULL);

      for (size_t k = 0; k < core.size(2); k++) {
        for (size_t j = 0; j < core.size(1); j++) {
          hsize_t st[3] = {starts[0] + k, starts[1] + j, starts[2]};
          hsize_t sz[3] = {1, 1, counts[2]};
          H5Sselect_hyperslab(file_space, H5S_SELECT_SET, st, NULL, sz, NULL);
          H5Dread(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i, off_j + j, off_k + k));
        }
      }
    } else {
      fatal("Parallel HDF5 read only implemented for 1D, 2D, 3D");
    }

    H5Pclose(xfer_plist);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
#else
    fatal(ERR_HDF5_NOT_PARALLEL);
#endif

  } else if (should_use_replicated_io()) {
    // Replicated mode: Rank 0 reads + broadcast
    if (dist_->rank == 0) this->read_h5(filename, varname);
    size_t total_size = this->size();
    MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) this->reshapef(this->dims);
    MPI_Bcast(this->data(), static_cast<int>(total_size), mpi_datatype(), 0, dist_->comm);

    // Propagate flags
    size_t flags[2] = {this->n_component_dims, (size_t)this->is_time_varying};
    MPI_Bcast(flags, 2, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) {
      this->n_component_dims = flags[0];
      this->is_time_varying = (bool)flags[1];
    }
  } else {
    // Serial mode fallback
    this->read_h5(filename, varname);
  }
#else
  // MPI not available, only serial mode
  this->read_h5(filename, varname);
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::write_hdf5_auto(
  const std::string& filename, const std::string& varname)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: Parallel HDF5 write
#ifdef H5_HAVE_PARALLEL
    const auto& core = dist_->local_core_;
    const auto& extent = dist_->local_extent_;
    const size_t nd = core.nd();

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, dist_->comm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    // Create dataspace for global array (C-order)
    std::vector<hsize_t> global_dims(nd);
    for (size_t d = 0; d < nd; d++) global_dims[nd - 1 - d] = dist_->global_lattice_.size(d);
    hid_t file_space = H5Screate_simple(nd, global_dims.data(), NULL);

    hid_t dataset_id = H5Dcreate(file_id, varname.c_str(), h5_mem_type_id(), file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Select hyperslab
    std::vector<hsize_t> starts(nd), counts(nd);
    for (size_t d = 0; d < nd; d++) {
      starts[nd - 1 - d] = core.start(d);
      counts[nd - 1 - d] = core.size(d);
    }

    // Memory space and transfer property
    hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = (nd >= 2) ? (core.start(1) - extent.start(1)) : 0;
    size_t off_k = (nd >= 3) ? (core.start(2) - extent.start(2)) : 0;

    if (nd == 1) {
      hid_t mem_space = H5Screate_simple(1, &counts[0], NULL);
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, starts.data(), NULL, counts.data(), NULL);
      H5Dwrite(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i));
      H5Sclose(mem_space);
    } else if (nd == 2) {
      hsize_t m_counts[2] = {1, counts[1]};
      hid_t mem_space = H5Screate_simple(2, m_counts, NULL);
      for (size_t j = 0; j < core.size(1); j++) {
        hsize_t st[2] = {starts[0] + j, starts[1]};
        hsize_t sz[2] = {1, counts[1]};
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, st, NULL, sz, NULL);
        H5Dwrite(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i, off_j + j));
      }
      H5Sclose(mem_space);
    } else if (nd == 3) {
      hsize_t m_counts[3] = {1, 1, counts[2]};
      hid_t mem_space = H5Screate_simple(3, m_counts, NULL);
      for (size_t k = 0; k < core.size(2); k++) {
        for (size_t j = 0; j < core.size(1); j++) {
          hsize_t st[3] = {starts[0] + k, starts[1] + j, starts[2]};
          hsize_t sz[3] = {1, 1, counts[2]};
          H5Sselect_hyperslab(file_space, H5S_SELECT_SET, st, NULL, sz, NULL);
          H5Dwrite(dataset_id, h5_mem_type_id(), mem_space, file_space, xfer_plist, &this->f(off_i, off_j + j, off_k + k));
        }
      }
      H5Sclose(mem_space);
    } else {
      fatal("Parallel HDF5 write only implemented for 1D, 2D, 3D");
    }

    H5Pclose(xfer_plist);
    H5Sclose(file_space);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
#else
    fatal(ERR_HDF5_NOT_PARALLEL);
#endif

  } else if (should_use_replicated_io()) {
    // Replicated mode: Only rank 0 writes
    if (dist_->rank == 0) this->to_h5(filename, varname);
    MPI_Barrier(dist_->comm);
  } else {
    // Serial mode fallback
    this->to_h5(filename, varname);
  }
#else
  // MPI not available, only serial mode
  this->to_h5(filename, varname);
#endif
}

#endif // NDARRAY_HAVE_HDF5

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::read_binary_auto(const std::string& filename)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);

  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: MPI-IO parallel read
    // Read only the local core region using column-major (Fortran) order
    const auto& core = dist_->local_core_;
    const auto& extent = dist_->local_extent_;
    const size_t nd = this->nd();

    // Check if we have ghost layers
    bool has_ghosts = false;
    for (size_t d = 0; d < nd; d++) {
      if (core.size(d) != extent.size(d)) {
        has_ghosts = true;
        break;
      }
    }

    MPI_File fh;
    int err = MPI_File_open(dist_->comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
      throw std::runtime_error("Failed to open file for parallel read: " + filename);
    }

    // For simplicity, read column-by-column for 2D
    if (nd == 2) {
      size_t core_size0 = core.size(0);
      size_t core_size1 = core.size(1);
      size_t global_size0 = dist_->global_lattice_.size(0);

      // Calculate ghost offsets if needed
      size_t ghost_offset_0 = has_ghosts ? (core.start(0) - extent.start(0)) : 0;
      size_t ghost_offset_1 = has_ghosts ? (core.start(1) - extent.start(1)) : 0;

      for (size_t j = 0; j < core_size1; j++) {
        size_t global_j = core.start(1) + j;
        size_t global_i = core.start(0);
        MPI_Offset col_offset = (global_i + global_j * global_size0) * sizeof(T);

        // Read into core region (accounting for ghosts if present)
        T* col_ptr = &this->f(ghost_offset_0, ghost_offset_1 + j);
        MPI_Status status;
        // Use non-collective read since different ranks have different core_size1
        // (different number of loop iterations). Collective operations must be
        // called the same number of times by all ranks.
        err = MPI_File_read_at(fh, col_offset, col_ptr, static_cast<int>(core_size0),
                               mpi_datatype(), &status);
        if (err != MPI_SUCCESS) {
          MPI_File_close(&fh);
          throw std::runtime_error("MPI_File_read_at failed for column " + std::to_string(j));
        }
      }
    } else {
      // 1D or higher-D: simple contiguous read
      // Calculate offset for column-major order
      MPI_Offset offset = 0;
      MPI_Offset stride = sizeof(T);

      for (size_t d = 0; d < nd; d++) {
        offset += static_cast<MPI_Offset>(core.start(d)) * stride;
        stride *= dist_->global_lattice_.size(d);
      }

      MPI_Status status;
      err = MPI_File_read_at_all(fh, offset, this->data(), static_cast<int>(core.n()),
                                 mpi_datatype(), &status);
      if (err != MPI_SUCCESS) {
        MPI_File_close(&fh);
        throw std::runtime_error("MPI_File_read_at_all failed");
      }
    }

    MPI_File_close(&fh);

  } else if (should_use_replicated_io()) {
    // Replicated mode: Rank 0 reads + broadcast
    if (dist_->rank == 0) {
      this->read_binary_file(filename);
    }

    size_t total_size = this->size();
    MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG, 0, dist_->comm);

    if (dist_->rank != 0) {
      this->reshapef(this->dims);
    }

    MPI_Bcast(this->data(), static_cast<int>(total_size), mpi_datatype(), 0, dist_->comm);

    // Propagate flags
    size_t flags[2] = {this->n_component_dims, (size_t)this->is_time_varying};
    MPI_Bcast(flags, 2, MPI_UNSIGNED_LONG, 0, dist_->comm);
    if (dist_->rank != 0) {
      this->n_component_dims = flags[0];
      this->is_time_varying = (bool)flags[1];
    }

  } else {
    // Serial mode fallback
    this->read_binary_file(filename);
  }
#else
  // MPI not available, only serial mode
  this->read_binary_file(filename);
#endif
}

template <typename T, typename StoragePolicy>
void ndarray<T, StoragePolicy>::write_binary_auto(const std::string& filename)
{
#if NDARRAY_HAVE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized && should_use_parallel_io()) {
    // Distributed mode: MPI-IO parallel write
    const auto& core = dist_->local_core_;
    const size_t nd = this->nd();

    MPI_File fh;
    MPI_File_open(dist_->comm, filename.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // Calculate offset for column-major order
    MPI_Offset offset = 0;
    MPI_Offset stride = sizeof(T);

    for (size_t d = 0; d < nd; d++) {
      offset += static_cast<MPI_Offset>(core.start(d)) * stride;
      stride *= dist_->global_lattice_.size(d);
    }

    // For 2D, write column-by-column
    if (nd == 2) {
      size_t core_size0 = core.size(0);
      size_t core_size1 = core.size(1);
      size_t global_size0 = dist_->global_lattice_.size(0);

      for (size_t j = 0; j < core_size1; j++) {
        size_t global_j = core.start(1) + j;
        size_t global_i = core.start(0);
        MPI_Offset col_offset = (global_i + global_j * global_size0) * sizeof(T);

        const T* col_ptr = &this->f(0, j);
        MPI_File_write_at_all(fh, col_offset, const_cast<T*>(col_ptr),
                              static_cast<int>(core_size0),
                              mpi_datatype(), MPI_STATUS_IGNORE);
      }
    } else {
      // 1D or higher-D: simple contiguous write
      MPI_File_write_at_all(fh, offset, this->data(), static_cast<int>(core.n()),
                            mpi_datatype(), MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);

  } else if (should_use_replicated_io()) {
    // Replicated mode: Only rank 0 writes
    if (dist_->rank == 0) {
      this->to_binary_file(filename);
    }
    MPI_Barrier(dist_->comm);

  } else {
    // Serial mode fallback
    this->to_binary_file(filename);
  }
#else
  // MPI not available, only serial mode
  this->to_binary_file(filename);
#endif
}

// Type aliases for convenience
template <typename T>
using ndarray_native = ndarray<T, native_storage>;

#if NDARRAY_HAVE_XTENSOR
template <typename T>
using ndarray_xtensor = ndarray<T, xtensor_storage>;
#endif

#if NDARRAY_HAVE_EIGEN
template <typename T>
using ndarray_eigen = ndarray<T, eigen_storage>;
#endif

//////////////////////////////////
// Factory methods (moved here to ensure ndarray<T> is fully defined)
//////////////////////////////////

inline std::shared_ptr<ndarray_base> ndarray_base::new_by_dtype(int type)
{
  std::shared_ptr<ndarray_base> p;

  if (type == NDARRAY_DTYPE_INT)
    p.reset(new ndarray<int>);
  else if (type == NDARRAY_DTYPE_FLOAT)
    p.reset(new ndarray<float>);
  else if (type == NDARRAY_DTYPE_DOUBLE)
    p.reset(new ndarray<double>);
  else if (type == NDARRAY_DTYPE_UNSIGNED_INT)
    p.reset(new ndarray<unsigned int>);
  else if (type == NDARRAY_DTYPE_UNSIGNED_CHAR)
    p.reset(new ndarray<unsigned char>);
  else if (type == NDARRAY_DTYPE_CHAR)
    p.reset(new ndarray<char>);
  else
    fatal(ERR_NOT_IMPLEMENTED);

  return p;
}

inline std::shared_ptr<ndarray_base> ndarray_base::new_by_vtk_dtype(int type)
{
  std::shared_ptr<ndarray_base> p;

#if NDARRAY_HAVE_VTK
  if (type == VTK_INT)
    p.reset(new ndarray<int>);
  else if (type == VTK_FLOAT)
    p.reset(new ndarray<float>);
  else if (type == VTK_DOUBLE)
    p.reset(new ndarray<double>);
  else if (type == VTK_UNSIGNED_INT)
    p.reset(new ndarray<unsigned int>);
  else if (type == VTK_UNSIGNED_CHAR)
    p.reset(new ndarray<unsigned char>);
  else
    throw not_implemented("VTK rectilinear grid output not yet implemented");
#else
  throw feature_not_available(ERR_NOT_BUILT_WITH_VTK, "VTK support not enabled in this build");
#endif

  return p;
}

inline std::shared_ptr<ndarray_base> ndarray_base::new_by_nc_dtype(int typep)
{
  std::shared_ptr<ndarray_base> p;

#if NDARRAY_HAVE_NETCDF
  if (typep == NC_INT)
    p.reset(new ndarray<int>);
  else if (typep == NC_FLOAT)
    p.reset(new ndarray<float>);
  else if (typep == NC_DOUBLE)
    p.reset(new ndarray<double>);
  else if (typep == NC_UINT)
    p.reset(new ndarray<unsigned int>);
  else if (typep == NC_CHAR)
    p.reset(new ndarray<char>);
  else
    fatal(ERR_NOT_IMPLEMENTED);
#else
  fatal(ERR_NOT_BUILT_WITH_NETCDF);
#endif

  return p;
}

inline std::shared_ptr<ndarray_base> ndarray_base::new_by_adios2_dtype(const std::string type)
{
  std::shared_ptr<ndarray_base> p;
#if NDARRAY_HAVE_ADIOS2
  if (type == adios2::GetType<int>())
    p.reset(new ndarray<int>);
  else if (type == adios2::GetType<float>())
    p.reset(new ndarray<float>);
  else if (type == adios2::GetType<double>())
    p.reset(new ndarray<double>);
  else if (type == adios2::GetType<unsigned int>())
    p.reset(new ndarray<unsigned int>);
  else if (type == adios2::GetType<unsigned long>())
    p.reset(new ndarray<unsigned long>);
  else if (type == adios2::GetType<unsigned char>())
    p.reset(new ndarray<unsigned char>);
  else if (type == adios2::GetType<char>())
    p.reset(new ndarray<char>);
  else
    throw not_implemented("Unsupported ADIOS2 data type");
#else
  throw feature_not_available(ERR_NOT_BUILT_WITH_ADIOS2, "ADIOS2 support not enabled in this build");
#endif
}

#if NDARRAY_HAVE_HDF5
inline std::shared_ptr<ndarray_base> ndarray_base::new_by_h5_dtype(hid_t type)
{
  std::shared_ptr<ndarray_base> p;

  if (H5Tequal(type, H5T_NATIVE_INT) > 0)
    p.reset(new ndarray<int>);
  else if (H5Tequal(type, H5T_NATIVE_FLOAT) > 0)
    p.reset(new ndarray<float>);
  else if (H5Tequal(type, H5T_NATIVE_DOUBLE) > 0)
    p.reset(new ndarray<double>);
  else if (H5Tequal(type, H5T_NATIVE_UINT) > 0)
    p.reset(new ndarray<unsigned int>);
  else if (H5Tequal(type, H5T_NATIVE_ULONG) > 0)
    p.reset(new ndarray<unsigned long>);
  else if (H5Tequal(type, H5T_NATIVE_UCHAR) > 0)
    p.reset(new ndarray<unsigned char>);
  else if (H5Tequal(type, H5T_NATIVE_CHAR) > 0)
    p.reset(new ndarray<char>);
  else
    fatal(ERR_NOT_IMPLEMENTED);

  return p;
}
#endif

} // namespace ftk

#endif // _NDARRAY_NDARRAY_HH
