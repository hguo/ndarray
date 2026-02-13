#ifndef _NDARRAY_NDARRAY_GROUP_HH
#define _NDARRAY_NDARRAY_GROUP_HH

#include <ndarray/ndarray.hh>
#include <map>
#include <stdexcept>

namespace ftk {

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
  template <typename T> void set(const std::string key, const ndarray<T> &arr);
  template <typename T> void set(const std::string key, ndarray<T> &&arr);  // Move version

  template <typename T> std::shared_ptr<ndarray<T>> get_ptr(const std::string key) {
    if (has(key)) return std::dynamic_pointer_cast<ndarray<T>>(at(key));
    else return nullptr;
  }

  // Zero-copy access: returns reference to internal array
  // WARNING: Reference is only valid while ndarray_group exists
  template <typename T> const ndarray<T>& get_ref(const std::string key) const;
  template <typename T> ndarray<T>& get_ref(const std::string key);

  void remove(const std::string key);

  template <typename T> ndarray<T> get_arr(const std::string key) {
    if (has(key)) return *get_ptr<T>(key);
    else return ndarray<T>();
  }

  // template <typename ... Args> ndarray_group(Args&&... args);

  void print_info(std::ostream& os) const;

public:
  std::string name;
};

template <typename T>
void ndarray_group::set(const std::string key, const ndarray<T> &arr)
{
  std::shared_ptr<ndarray_base> parr(new ndarray<T>(arr));  // Copy construct
  this->set(key, parr);
}

// Move version - avoids copy when caller uses std::move()
template <typename T>
void ndarray_group::set(const std::string key, ndarray<T> &&arr)
{
  std::shared_ptr<ndarray_base> parr(new ndarray<T>(std::move(arr)));  // Move construct
  this->set(key, parr);
}

// Zero-copy read access - returns const reference
template <typename T>
const ndarray<T>& ndarray_group::get_ref(const std::string key) const
{
  auto ptr = get_ptr<T>(key);
  if (!ptr) {
    throw std::runtime_error("ndarray_group::get_ref: key '" + key + "' not found");
  }
  return *ptr;
}

// Zero-copy write access - returns mutable reference
template <typename T>
ndarray<T>& ndarray_group::get_ref(const std::string key)
{
  auto ptr = get_ptr<T>(key);
  if (!ptr) {
    throw std::runtime_error("ndarray_group::get_ref: key '" + key + "' not found");
  }
  return *ptr;
}

inline void ndarray_group::remove(const std::string key)
{
  auto p = this->find(key);
  if (p != this->end()) {
    this->erase(p);
  }
}

inline void ndarray_group::print_info(std::ostream& os) const
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

}

#endif
