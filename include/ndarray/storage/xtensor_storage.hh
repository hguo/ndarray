#ifndef _FTK_XTENSOR_STORAGE_HH
#define _FTK_XTENSOR_STORAGE_HH

#include "storage_policy.hh"

#if NDARRAY_HAVE_XTENSOR
#include <xtensor/containers/xarray.hpp>
#include <vector>

namespace ftk {

// xtensor storage policy using xt::xarray
// Provides SIMD optimization and expression templates
struct xtensor_storage {
  template <typename T>
  struct container_type {
    xt::xarray<T> data_;

    size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    void resize(size_t n) {
      data_.resize({n});
    }

    void reshape(const std::vector<size_t>& shape) {
      data_.reshape(shape);
    }

    T& operator[](size_t i) { return data_.data()[i]; }
    const T& operator[](size_t i) const { return data_.data()[i]; }

    void fill(T value) { data_.fill(value); }

    // Access underlying xtensor array for advanced operations
    xt::xarray<T>& get_xarray() { return data_; }
    const xt::xarray<T>& get_xarray() const { return data_; }
  };
};

} // namespace ftk

#endif // NDARRAY_HAVE_XTENSOR

#endif // _FTK_XTENSOR_STORAGE_HH
