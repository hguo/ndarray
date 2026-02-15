#ifndef _NDARRAY_NDARRAY_GROUP_HH
#define _NDARRAY_NDARRAY_GROUP_HH

#include <ndarray/ndarray.hh>
#include <map>
#include <stdexcept>

namespace ftk {

template <typename StoragePolicy = native_storage>
struct ndarray_group : public std::map<std::string, std::shared_ptr<ndarray_base>> {
  ndarray_group() {}

  bool has(const std::string key) const { return this->find(key) != this->end(); }

  std::shared_ptr<ndarray_base> get(const std::string key) const {
    auto it = this->find(key);
    if (it != this->end())
      return it->second;
    else
      return nullptr;
  }

  void set(const std::string key, std::shared_ptr<ndarray_base> ptr) { this->emplace(key, ptr); }
  template <typename T> void set(const std::string key, const ndarray<T, StoragePolicy> &arr);
  template <typename T> void set(const std::string key, ndarray<T, StoragePolicy> &&arr);  // Move version

  template <typename T> std::shared_ptr<ndarray<T, StoragePolicy>> get_ptr(const std::string key) {
    if (has(key)) return std::dynamic_pointer_cast<ndarray<T, StoragePolicy>>(at(key));
    else return nullptr;
  }

  // Zero-copy access: returns reference to internal array
  // WARNING: Reference is only valid while ndarray_group exists
  template <typename T> const ndarray<T, StoragePolicy>& get_ref(const std::string key) const;
  template <typename T> ndarray<T, StoragePolicy>& get_ref(const std::string key);

  void remove(const std::string key);

  template <typename T> ndarray<T, StoragePolicy> get_arr(const std::string key) {
    if (has(key)) return *get_ptr<T>(key);
    else return ndarray<T, StoragePolicy>();
  }

  // template <typename ... Args> ndarray_group(Args&&... args);

  void print_info(std::ostream& os) const;

public:
  std::string name;
};

template <typename StoragePolicy>
template <typename T>
void ndarray_group<StoragePolicy>::set(const std::string key, const ndarray<T, StoragePolicy> &arr)
{
  std::shared_ptr<ndarray_base> parr(new ndarray<T, StoragePolicy>(arr));  // Copy construct
  this->set(key, parr);
}

// Move version - avoids copy when caller uses std::move()
template <typename StoragePolicy>
template <typename T>
void ndarray_group<StoragePolicy>::set(const std::string key, ndarray<T, StoragePolicy> &&arr)
{
  std::shared_ptr<ndarray_base> parr(new ndarray<T, StoragePolicy>(std::move(arr)));  // Move construct
  this->set(key, parr);
}

// Zero-copy read access - returns const reference
template <typename StoragePolicy>
template <typename T>
const ndarray<T, StoragePolicy>& ndarray_group<StoragePolicy>::get_ref(const std::string key) const
{
  auto ptr = get_ptr<T>(key);
  if (!ptr) {
    throw std::runtime_error("ndarray_group::get_ref: key '" + key + "' not found");
  }
  return *ptr;
}

// Zero-copy write access - returns mutable reference
template <typename StoragePolicy>
template <typename T>
ndarray<T, StoragePolicy>& ndarray_group<StoragePolicy>::get_ref(const std::string key)
{
  auto ptr = get_ptr<T>(key);
  if (!ptr) {
    throw std::runtime_error("ndarray_group::get_ref: key '" + key + "' not found");
  }
  return *ptr;
}

template <typename StoragePolicy>
inline void ndarray_group<StoragePolicy>::remove(const std::string key)
{
  auto p = this->find(key);
  if (p != this->end()) {
    this->erase(p);
  }
}

template <typename StoragePolicy>
inline void ndarray_group<StoragePolicy>::print_info(std::ostream& os) const
{
  os << "array_group "
     << (this->name.empty() ? "(no name)" : name)
     << (this->empty() ? " (empty)" : " ")
     << std::endl;

  for (const auto &kv : *this) {
    os << " - name: " << kv.first.c_str() << ", ";
    os << "type: " << ndarray_base::dtype2str( kv.second->type() ) << ", ";
    kv.second->print_shapef(os);
    os << std::endl;
  }
}

// Type aliases for convenience
using ndarray_group_native = ndarray_group<native_storage>;

#if NDARRAY_HAVE_XTENSOR
using ndarray_group_xtensor = ndarray_group<xtensor_storage>;
#endif

#if NDARRAY_HAVE_EIGEN
using ndarray_group_eigen = ndarray_group<eigen_storage>;
#endif

}

#endif
