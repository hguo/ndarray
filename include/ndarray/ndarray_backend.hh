#ifndef _NDARRAY_BACKEND_HH
#define _NDARRAY_BACKEND_HH

#include <ndarray/config.hh>
#include <vector>
#include <memory>

#if NDARRAY_HAVE_EIGEN
#include <Eigen/Dense>
#endif

#if NDARRAY_HAVE_XTENSOR
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#endif

namespace ftk {

/**
 * @brief Backend storage policies for ndarray
 *
 * Backends determine how array data is stored and accessed.
 * Available backends:
 * - native_backend: std::vector (default, always available)
 * - eigen_backend: Eigen::Matrix/Vector (requires NDARRAY_HAVE_EIGEN)
 * - xtensor_backend: xt::xarray (requires NDARRAY_HAVE_XTENSOR)
 */

///////////
// Native Backend (std::vector) - Always available
///////////

template <typename T>
struct native_backend {
  using value_type = T;
  using storage_type = std::vector<T>;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  storage_type data_;

  native_backend() = default;

  void resize(size_t n) {
    data_.resize(n);
  }

  void clear() {
    data_.clear();
  }

  size_t size() const {
    return data_.size();
  }

  bool empty() const {
    return data_.empty();
  }

  pointer data() {
    return data_.data();
  }

  const_pointer data() const {
    return data_.data();
  }

  reference operator[](size_t i) {
    return data_[i];
  }

  const_reference operator[](size_t i) const {
    return data_[i];
  }

  void fill(const T& value) {
    std::fill(data_.begin(), data_.end(), value);
  }

  // Iterator support
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  // Compatibility with std::vector interface
  std::vector<T>& std_vector() { return data_; }
  const std::vector<T>& std_vector() const { return data_; }
};

///////////
// Eigen Backend
///////////

#if NDARRAY_HAVE_EIGEN

template <typename T>
struct eigen_backend {
  using value_type = T;
  using storage_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;  // Column vector for flat storage
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  storage_type data_;

  eigen_backend() = default;

  void resize(size_t n) {
    data_.resize(n);
  }

  void clear() {
    data_.resize(0);
  }

  size_t size() const {
    return data_.size();
  }

  bool empty() const {
    return data_.size() == 0;
  }

  pointer data() {
    return data_.data();
  }

  const_pointer data() const {
    return data_.data();
  }

  reference operator[](size_t i) {
    return data_(i);
  }

  const_reference operator[](size_t i) const {
    return data_(i);
  }

  void fill(const T& value) {
    data_.setConstant(value);
  }

  // Iterator support
  auto begin() { return data_.data(); }
  auto end() { return data_.data() + data_.size(); }
  auto begin() const { return data_.data(); }
  auto end() const { return data_.data() + data_.size(); }

  // Convert to std::vector (copy)
  std::vector<T> std_vector() const {
    return std::vector<T>(data_.data(), data_.data() + data_.size());
  }

  // Access underlying Eigen storage
  storage_type& eigen_vector() { return data_; }
  const storage_type& eigen_vector() const { return data_; }
};

#endif // NDARRAY_HAVE_EIGEN

///////////
// xtensor Backend
///////////

#if NDARRAY_HAVE_XTENSOR

template <typename T>
struct xtensor_backend {
  using value_type = T;
  using storage_type = xt::xarray<T>;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  storage_type data_;

  xtensor_backend() = default;

  void resize(size_t n) {
    data_.resize({n});
  }

  void clear() {
    data_.resize({0});
  }

  size_t size() const {
    return data_.size();
  }

  bool empty() const {
    return data_.size() == 0;
  }

  pointer data() {
    return data_.data();
  }

  const_pointer data() const {
    return data_.data();
  }

  reference operator[](size_t i) {
    return data_.data()[i];  // Flat access
  }

  const_reference operator[](size_t i) const {
    return data_.data()[i];
  }

  void fill(const T& value) {
    data_.fill(value);
  }

  // Iterator support
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  // Convert to std::vector (copy)
  std::vector<T> std_vector() const {
    return std::vector<T>(data_.begin(), data_.end());
  }

  // Access underlying xtensor storage
  storage_type& xtensor_array() { return data_; }
  const storage_type& xtensor_array() const { return data_; }
};

#endif // NDARRAY_HAVE_XTENSOR

} // namespace ftk

#endif // _NDARRAY_BACKEND_HH
