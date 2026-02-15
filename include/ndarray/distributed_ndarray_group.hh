#ifndef _DISTRIBUTED_NDARRAY_GROUP_HH
#define _DISTRIBUTED_NDARRAY_GROUP_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_MPI

#include <ndarray/distributed_ndarray.hh>
#include <map>
#include <string>
#include <memory>
#include <mpi.h>

namespace ftk {

/**
 * @brief Container for multiple distributed ndarrays
 *
 * Similar to ndarray_group but contains distributed_ndarray instances
 * instead of regular ndarray instances. All arrays share the same
 * domain decomposition.
 *
 * Used by distributed_stream to return multiple variables at once.
 *
 * @code
 * distributed_ndarray_group<float> group(MPI_COMM_WORLD);
 * group.add("temperature", temp_array);
 * group.add("pressure", pres_array);
 *
 * auto& temp = group["temperature"];
 * temp.exchange_ghosts();
 * @endcode
 */
template <typename T = float, typename StoragePolicy = native_storage>
class distributed_ndarray_group {
public:
  using distributed_array_type = distributed_ndarray<T, StoragePolicy>;
  using map_type = std::map<std::string, distributed_array_type>;

  /**
   * @brief Constructor
   * @param comm MPI communicator
   */
  distributed_ndarray_group(MPI_Comm comm = MPI_COMM_WORLD)
    : comm_(comm)
  {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);
  }

  /**
   * @brief Add distributed array to group
   * @param name Variable name
   * @param array Distributed array (moved)
   */
  void add(const std::string& name, distributed_array_type&& array)
  {
    arrays_[name] = std::move(array);
  }

  /**
   * @brief Add distributed array to group (copy)
   * @param name Variable name
   * @param array Distributed array (copied)
   */
  void add(const std::string& name, const distributed_array_type& array)
  {
    arrays_[name] = array;
  }

  /**
   * @brief Access distributed array by name
   * @param name Variable name
   * @return Reference to distributed array
   */
  distributed_array_type& operator[](const std::string& name)
  {
    return arrays_.at(name);
  }

  /**
   * @brief Access distributed array by name (const)
   * @param name Variable name
   * @return Const reference to distributed array
   */
  const distributed_array_type& operator[](const std::string& name) const
  {
    return arrays_.at(name);
  }

  /**
   * @brief Get distributed array by name
   * @param name Variable name
   * @return Reference to distributed array
   */
  distributed_array_type& get(const std::string& name)
  {
    return arrays_.at(name);
  }

  /**
   * @brief Get distributed array by name (const)
   * @param name Variable name
   * @return Const reference to distributed array
   */
  const distributed_array_type& get(const std::string& name) const
  {
    return arrays_.at(name);
  }

  /**
   * @brief Check if array exists in group
   * @param name Variable name
   * @return true if exists
   */
  bool has(const std::string& name) const
  {
    return arrays_.find(name) != arrays_.end();
  }

  /**
   * @brief Get number of arrays in group
   * @return Number of arrays
   */
  size_t size() const
  {
    return arrays_.size();
  }

  /**
   * @brief Check if group is empty
   * @return true if empty
   */
  bool empty() const
  {
    return arrays_.empty();
  }

  /**
   * @brief Exchange ghosts for all arrays in group
   */
  void exchange_ghosts_all()
  {
    for (auto& pair : arrays_) {
      pair.second.exchange_ghosts();
    }
  }

  /**
   * @brief Get iterator to beginning of arrays
   */
  typename map_type::iterator begin()
  {
    return arrays_.begin();
  }

  /**
   * @brief Get iterator to end of arrays
   */
  typename map_type::iterator end()
  {
    return arrays_.end();
  }

  /**
   * @brief Get const iterator to beginning of arrays
   */
  typename map_type::const_iterator begin() const
  {
    return arrays_.begin();
  }

  /**
   * @brief Get const iterator to end of arrays
   */
  typename map_type::const_iterator end() const
  {
    return arrays_.end();
  }

  /**
   * @brief Get names of all arrays in group
   * @return Vector of variable names
   */
  std::vector<std::string> names() const
  {
    std::vector<std::string> result;
    for (const auto& pair : arrays_) {
      result.push_back(pair.first);
    }
    return result;
  }

  // MPI accessors
  int rank() const { return rank_; }
  int nprocs() const { return nprocs_; }
  MPI_Comm comm() const { return comm_; }

private:
  MPI_Comm comm_;
  int rank_;
  int nprocs_;
  map_type arrays_;
};

} // namespace ftk

#endif // NDARRAY_HAVE_MPI

#endif // _DISTRIBUTED_NDARRAY_GROUP_HH
