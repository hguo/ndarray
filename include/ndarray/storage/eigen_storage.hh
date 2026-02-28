#ifndef _FTK_EIGEN_STORAGE_HH
#define _FTK_EIGEN_STORAGE_HH

#include "storage_policy.hh"

#if NDARRAY_HAVE_EIGEN
#include <Eigen/Dense>
#include <vector>

namespace ftk {

// Eigen storage policy using Eigen::Matrix
// Provides optimized linear algebra operations
struct eigen_storage {
  template <typename T>
  struct container_type {
    // Use Eigen::Matrix with dynamic size, column-major (matches Fortran/NumPy order)
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> data_;
    std::vector<size_t> shape_;  // Track shape separately for N-D arrays

    size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    void resize(size_t n) {
      data_.resize(n, 1);
      shape_ = {n};
    }

    void reshape(const std::vector<size_t>& shape) {
      shape_ = shape;
      size_t total = 1;
      for (auto s : shape) total *= s;

      if (shape.size() == 2) {
        // 2D array - use native Eigen matrix layout
        data_.resize(shape[1], shape[0]);
      } else {
        // 1D or N-D array - flatten to column vector
        data_.resize(total, 1);
      }
    }

    T& operator[](size_t i) { return data_.data()[i]; }
    const T& operator[](size_t i) const { return data_.data()[i]; }

    void fill(T value) { data_.fill(value); }

    // Access underlying Eigen matrix for linear algebra operations
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& get_matrix() { return data_; }
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& get_matrix() const { return data_; }
  };
};

} // namespace ftk

#endif // NDARRAY_HAVE_EIGEN

#endif // _FTK_EIGEN_STORAGE_HH
