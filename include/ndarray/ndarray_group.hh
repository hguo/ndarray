#ifndef _NDARRAY_NDARRAY_GROUP_HH
#define _NDARRAY_NDARRAY_GROUP_HH

#include <ndarray/ndarray.hh>
#include <unordered_map>

namespace ndarray {

struct ndarray_group : public std::unordered_map<std::string, std::shared_ptr<ndarray_base>> {
  ndarray_group() {}

  bool has(const std::string key) const { return this->find(key) != this->end(); }

  void set(const std::string key, std::shared_ptr<ndarray_base> ptr) { this->emplace(key, ptr); }
  template <typename T> void set(const std::string key, const ndarray<T> &arr);

  template <typename T> std::shared_ptr<ndarray<T>> get_ptr(const std::string key) {
    if (has(key)) return std::dynamic_pointer_cast<ndarray<T>>(at(key)); 
    else return nullptr;
  }

  void remove(const std::string key);

  template <typename T> ndarray<T> get(const std::string key) { return *get_ptr<T>(key); }

  // template <typename ... Args> ndarray_group(Args&&... args);
};

template <typename T>
void ndarray_group::set(const std::string key, const ndarray<T> &arr)
{
  // std::cerr << arr << std::endl;
  std::shared_ptr<ndarray_base> parr(new ndarray<T>);
  *(std::dynamic_pointer_cast<ndarray<T>>(parr)) = arr;
  // std::cerr << *(std::dynamic_pointer_cast<ndarray<T>>(parr)) << std::endl;
  this->set(key, parr);
}

inline void ndarray_group::remove(const std::string key)
{
  auto p = this->find(key);
  if (p != this->end()) {
    this->erase(p);
  }
}

}

#endif
