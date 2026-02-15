#ifndef _FTK_STORAGE_POLICY_HH
#define _FTK_STORAGE_POLICY_HH

#include <type_traits>

namespace ftk {

// Storage policy concept (C++17 compatible)
// All storage policies must provide a container_type<T> with these operations:
// - size_t size() const
// - T* data()
// - const T* data() const
// - void resize(size_t)
// - T& operator[](size_t)
// - const T& operator[](size_t) const
// - void fill(T value)

// Optional operations:
// - void reshape(const std::vector<size_t>&)  // For multi-dimensional backends

// Type trait to check if a type has reshape method
template <typename T, typename = void>
struct has_reshape : std::false_type {};

template <typename T>
struct has_reshape<T, std::void_t<decltype(std::declval<T>().reshape(std::declval<std::vector<size_t>>()))>>
  : std::true_type {};

} // namespace ftk

#endif // _FTK_STORAGE_POLICY_HH
