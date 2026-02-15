#ifndef _FTK_NATIVE_STORAGE_HH
#define _FTK_NATIVE_STORAGE_HH

#include <vector>
#include <algorithm>

namespace ftk {

// Native storage policy using std::vector
// This is the default storage backend, providing backward compatibility
struct native_storage {
  template <typename T>
  struct container_type {
    std::vector<T> data_;

    size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    void resize(size_t n) { data_.resize(n); }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

    // Iterator support for compatibility
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }
  };
};

} // namespace ftk

#endif // _FTK_NATIVE_STORAGE_HH
