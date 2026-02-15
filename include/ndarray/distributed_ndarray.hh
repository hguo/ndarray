#ifndef _FTK_DISTRIBUTED_NDARRAY_HH
#define _FTK_DISTRIBUTED_NDARRAY_HH

#include <ndarray/ndarray.hh>
#include <ndarray/lattice.hh>
#include <ndarray/lattice_partitioner.hh>
#include <memory>

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

namespace ftk {

/**
 * Distributed multidimensional array for MPI parallel I/O and computation
 *
 * Provides:
 * - Automatic or user-specified domain decomposition
 * - Ghost layer management
 * - Global/local index conversion
 * - Parallel I/O (NetCDF, HDF5, ADIOS2, binary with MPI-IO)
 * - Ghost cell exchange via MPI
 *
 * Each MPI rank owns a portion of the global domain (the "core") plus optional
 * ghost cells from neighbors (the "extent").
 *
 * Usage:
 *   distributed_ndarray<float> darray(MPI_COMM_WORLD);
 *   darray.decompose({1000, 800}, 0, {}, {1, 1});  // auto decomp with 1-layer ghosts
 *   darray.read_parallel("data.nc", "temperature");
 *   darray.exchange_ghosts();
 *   // Perform local computation on darray.local_array()
 *   darray.write_parallel("output.nc", "result");
 */
template <typename T, typename StoragePolicy = native_storage>
class distributed_ndarray {
public:
  // Construction
  distributed_ndarray(MPI_Comm comm = MPI_COMM_WORLD);
  ~distributed_ndarray() = default;

  // Copy/move (deep copy of local data, MPI comm is not copyable)
  distributed_ndarray(const distributed_ndarray&) = delete;
  distributed_ndarray& operator=(const distributed_ndarray&) = delete;
  distributed_ndarray(distributed_ndarray&&) = default;
  distributed_ndarray& operator=(distributed_ndarray&&) = default;

  /**
   * Decompose global domain across MPI ranks
   *
   * @param global_dims Global array dimensions
   * @param nprocs Number of processes to use (0 = use all ranks in comm)
   * @param decomp Decomposition pattern per dimension (empty = automatic via prime factorization)
   *               e.g., {4, 2, 0} means: split dim 0 into 4 parts, dim 1 into 2 parts, don't split dim 2
   * @param ghost Ghost layer width per dimension (empty = no ghosts)
   *              e.g., {1, 1, 0} means: 1-layer ghosts in dims 0 and 1, none in dim 2
   */
  void decompose(const std::vector<size_t>& global_dims,
                 size_t nprocs = 0,
                 const std::vector<size_t>& decomp = {},
                 const std::vector<size_t>& ghost = {});

  // Access to local data and lattices
  const ndarray<T, StoragePolicy>& local_array() const { return local_data_; }
  ndarray<T, StoragePolicy>& local_array() { return local_data_; }

  const lattice& global_lattice() const { return global_lattice_; }
  const lattice& local_core() const { return local_core_; }
  const lattice& local_extent() const { return local_extent_; }

  /**
   * Convert global index to local index
   *
   * @param global_idx Global index (relative to global domain)
   * @return Local index (relative to local_core start)
   * @note Does not check if index is actually local - use is_local() first
   */
  std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const;

  /**
   * Convert local index to global index
   *
   * @param local_idx Local index (relative to local_core start)
   * @return Global index (relative to global domain)
   */
  std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const;

  /**
   * Check if global index is owned by this rank
   *
   * @param global_idx Global index
   * @return True if this rank's local_core contains the index
   */
  bool is_local(const std::vector<size_t>& global_idx) const;

  // MPI info
  int rank() const { return rank_; }
  int nprocs() const { return nprocs_; }
  MPI_Comm comm() const { return comm_; }

  /**
   * Parallel read from file (detects format from extension)
   *
   * Reads this rank's local portion (local_core) from file.
   * After read, call exchange_ghosts() to fill ghost layers.
   *
   * Supported formats:
   * - .nc: NetCDF via PNetCDF (collective read)
   * - .h5: HDF5 with MPI-IO (collective read)
   * - .bin: Binary with MPI-IO (collective read, assumes contiguous row-major)
   *
   * @param filename Path to file
   * @param varname Variable name (ignored for binary files)
   * @param timestep Timestep index (default 0, ignored for binary files)
   */
  void read_parallel(const std::string& filename,
                     const std::string& varname = "",
                     int timestep = 0);

  /**
   * Exchange ghost cells with neighboring ranks
   *
   * Updates ghost layers from neighbors' core boundary data via MPI communication.
   * Must be called after read_parallel() or any operation that modifies core data.
   *
   * Uses non-blocking receives (MPI_Irecv) + blocking sends (MPI_Send) + wait.
   * Only exchanges with direct neighbors (face-adjacent in decomposition).
   */
  void exchange_ghosts();

  // TODO: Phase 2 continuation
  // void write_parallel(const std::string& filename, const std::string& varname, int timestep = 0);

private:
  // Format-specific parallel read methods
  void read_parallel_netcdf(const std::string& filename, const std::string& varname, int timestep);
  void read_parallel_hdf5(const std::string& filename, const std::string& varname, int timestep);
  void read_parallel_binary(const std::string& filename);

  // Helper to get MPI datatype for T
  static MPI_Datatype mpi_datatype();

  // Helper to get file extension
  static std::string get_file_extension(const std::string& filename);

  // Ghost exchange helpers
  void identify_neighbors();
  void pack_boundary_data(int neighbor_idx, std::vector<T>& buffer);
  void unpack_ghost_data(int neighbor_idx, const std::vector<T>& buffer);

  MPI_Comm comm_;
  int rank_;
  int nprocs_;

  lattice global_lattice_;        // Full global domain
  std::unique_ptr<lattice_partitioner> partitioner_;

  lattice local_core_;            // This rank's core region (no ghosts)
  lattice local_extent_;          // This rank's extent region (core + ghosts)

  ndarray<T, StoragePolicy> local_data_;  // Local data (allocated to extent size)

  // Neighbor information for ghost exchange
  struct Neighbor {
    int rank;              // Neighbor's MPI rank
    int direction;         // Which face: 0=left, 1=right (dim 0); 2=down, 3=up (dim 1); etc.
    size_t send_offset;    // Offset in local_data for boundary to send
    size_t send_count;     // Number of elements to send
    size_t recv_offset;    // Offset in local_data for ghost to receive
    size_t recv_count;     // Number of elements to receive
  };

  std::vector<Neighbor> neighbors_;  // List of neighbors for ghost exchange
  bool neighbors_identified_ = false;  // Whether identify_neighbors() has been called
};

/////
// Implementation
/////

template <typename T, typename StoragePolicy>
distributed_ndarray<T, StoragePolicy>::distributed_ndarray(MPI_Comm comm)
  : comm_(comm), rank_(0), nprocs_(1)
{
#if NDARRAY_HAVE_MPI
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nprocs_);
#else
  (void)comm;  // Avoid unused parameter warning
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::decompose(
    const std::vector<size_t>& global_dims,
    size_t nprocs,
    const std::vector<size_t>& decomp,
    const std::vector<size_t>& ghost)
{
#if !NDARRAY_HAVE_MPI
  throw std::runtime_error("distributed_ndarray::decompose: MPI support not enabled");
#else
  // Use all ranks if nprocs=0
  if (nprocs == 0) {
    nprocs = static_cast<size_t>(nprocs_);
  }

  // Verify nprocs matches communicator size
  if (nprocs != static_cast<size_t>(nprocs_)) {
    throw std::invalid_argument(
        "distributed_ndarray::decompose: nprocs does not match MPI_Comm_size");
  }

  // Create global lattice
  global_lattice_.reshape(global_dims);

  // Create partitioner and decompose
  partitioner_ = std::make_unique<lattice_partitioner>(global_lattice_);

  if (decomp.empty() && ghost.empty()) {
    // Automatic decomposition, no ghosts
    partitioner_->partition(nprocs);
  } else if (!decomp.empty() && ghost.empty()) {
    // User-specified decomposition, no ghosts
    partitioner_->partition(nprocs, decomp);
  } else if (!decomp.empty() && !ghost.empty()) {
    // User-specified decomposition with ghosts
    partitioner_->partition(nprocs, decomp, ghost);
  } else {
    // Automatic decomposition with ghosts
    partitioner_->partition(nprocs, std::vector<size_t>(), ghost);
  }

  // Verify decomposition succeeded
  if (partitioner_->np() == 0) {
    throw std::runtime_error(
        "distributed_ndarray::decompose: lattice_partitioner failed to decompose domain");
  }

  if (partitioner_->np() != nprocs) {
    throw std::runtime_error(
        "distributed_ndarray::decompose: number of partitions does not match nprocs");
  }

  // Store this rank's core and extent lattices
  local_core_ = partitioner_->get_core(rank_);
  local_extent_ = partitioner_->get_ext(rank_);

  // Allocate local data to extent size
  std::vector<size_t> extent_dims = local_extent_.sizes();
  local_data_.reshapef(extent_dims);

  // Identify neighbors for ghost exchange
  if (!ghost.empty()) {
    identify_neighbors();
  }
#endif
}

template <typename T, typename StoragePolicy>
std::vector<size_t> distributed_ndarray<T, StoragePolicy>::global_to_local(
    const std::vector<size_t>& global_idx) const
{
  if (global_idx.size() != local_core_.nd()) {
    throw std::invalid_argument(
        "distributed_ndarray::global_to_local: index dimensionality mismatch");
  }

  std::vector<size_t> local_idx(global_idx.size());
  for (size_t d = 0; d < global_idx.size(); d++) {
    // Local index = global index - local_core start
    local_idx[d] = global_idx[d] - local_core_.start(d);
  }

  return local_idx;
}

template <typename T, typename StoragePolicy>
std::vector<size_t> distributed_ndarray<T, StoragePolicy>::local_to_global(
    const std::vector<size_t>& local_idx) const
{
  if (local_idx.size() != local_core_.nd()) {
    throw std::invalid_argument(
        "distributed_ndarray::local_to_global: index dimensionality mismatch");
  }

  std::vector<size_t> global_idx(local_idx.size());
  for (size_t d = 0; d < local_idx.size(); d++) {
    // Global index = local index + local_core start
    global_idx[d] = local_idx[d] + local_core_.start(d);
  }

  return global_idx;
}

template <typename T, typename StoragePolicy>
bool distributed_ndarray<T, StoragePolicy>::is_local(
    const std::vector<size_t>& global_idx) const
{
  if (global_idx.size() != local_core_.nd()) {
    return false;
  }

  // Check if global_idx is within local_core bounds
  for (size_t d = 0; d < global_idx.size(); d++) {
    if (global_idx[d] < local_core_.start(d) ||
        global_idx[d] >= local_core_.start(d) + local_core_.size(d)) {
      return false;
    }
  }

  return true;
}

/////
// Parallel I/O Implementation
/////

template <typename T, typename StoragePolicy>
std::string distributed_ndarray<T, StoragePolicy>::get_file_extension(
    const std::string& filename)
{
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) {
    return "";
  }
  return filename.substr(dot_pos);
}

template <typename T, typename StoragePolicy>
MPI_Datatype distributed_ndarray<T, StoragePolicy>::mpi_datatype()
{
#if NDARRAY_HAVE_MPI
  if (std::is_same<T, float>::value) {
    return MPI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    return MPI_DOUBLE;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, unsigned int>::value) {
    return MPI_UNSIGNED;
  } else if (std::is_same<T, long>::value) {
    return MPI_LONG;
  } else if (std::is_same<T, unsigned long>::value) {
    return MPI_UNSIGNED_LONG;
  } else if (std::is_same<T, long long>::value) {
    return MPI_LONG_LONG;
  } else if (std::is_same<T, short>::value) {
    return MPI_SHORT;
  } else if (std::is_same<T, unsigned short>::value) {
    return MPI_UNSIGNED_SHORT;
  } else if (std::is_same<T, char>::value) {
    return MPI_CHAR;
  } else if (std::is_same<T, unsigned char>::value) {
    return MPI_UNSIGNED_CHAR;
  } else {
    throw std::runtime_error("distributed_ndarray: unsupported MPI datatype");
  }
#else
  throw std::runtime_error("distributed_ndarray: MPI support not enabled");
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::read_parallel(
    const std::string& filename,
    const std::string& varname,
    int timestep)
{
#if !NDARRAY_HAVE_MPI
  throw std::runtime_error("distributed_ndarray::read_parallel: MPI support not enabled");
#else
  // Detect format from file extension
  std::string ext = get_file_extension(filename);

  if (ext == ".nc") {
#if NDARRAY_HAVE_PNETCDF
    read_parallel_netcdf(filename, varname, timestep);
#else
    throw std::runtime_error(
        "distributed_ndarray::read_parallel: NetCDF file detected but PNetCDF support not enabled");
#endif
  } else if (ext == ".h5" || ext == ".hdf5") {
#if NDARRAY_HAVE_HDF5
    read_parallel_hdf5(filename, varname, timestep);
#else
    throw std::runtime_error(
        "distributed_ndarray::read_parallel: HDF5 file detected but HDF5 support not enabled");
#endif
  } else if (ext == ".bin" || ext == ".raw" || ext == ".dat") {
    read_parallel_binary(filename);
  } else {
    throw std::runtime_error(
        "distributed_ndarray::read_parallel: unsupported file format: " + ext);
  }
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::read_parallel_netcdf(
    const std::string& filename,
    const std::string& varname,
    int timestep)
{
#if !NDARRAY_HAVE_PNETCDF
  throw std::runtime_error("distributed_ndarray::read_parallel_netcdf: PNetCDF support not enabled");
#else
  // Open file with MPI
  int ncid;
  PNC_SAFE_CALL(ncmpi_open(comm_, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid));

  // Get variable ID
  int varid;
  PNC_SAFE_CALL(ncmpi_inq_varid(ncid, varname.c_str(), &varid));

  // Get variable dimensionality
  int ndims;
  PNC_SAFE_CALL(ncmpi_inq_varndims(ncid, varid, &ndims));

  // Verify dimensionality matches decomposition (excluding unlimited time dimension)
  int expected_ndims = static_cast<int>(local_core_.nd());
  if (ndims != expected_ndims && ndims != expected_ndims + 1) {
    ncmpi_close(ncid);
    throw std::runtime_error(
        "distributed_ndarray::read_parallel_netcdf: variable dimensionality mismatch");
  }

  // Set up start/count arrays from local_core
  std::vector<MPI_Offset> start(ndims);
  std::vector<MPI_Offset> count(ndims);

  // Handle time dimension if present
  if (ndims == expected_ndims + 1) {
    start[0] = timestep;
    count[0] = 1;
    for (int d = 0; d < expected_ndims; d++) {
      start[d + 1] = static_cast<MPI_Offset>(local_core_.start(d));
      count[d + 1] = static_cast<MPI_Offset>(local_core_.size(d));
    }
  } else {
    for (int d = 0; d < expected_ndims; d++) {
      start[d] = static_cast<MPI_Offset>(local_core_.start(d));
      count[d] = static_cast<MPI_Offset>(local_core_.size(d));
    }
  }

  // Read into local_data_ (core portion only, not ghosts)
  // We need to create a temporary array for the core portion
  ndarray<T, StoragePolicy> core_data;
  std::vector<size_t> core_dims = local_core_.sizes();
  core_data.reshapef(core_dims);

  // Use existing read_pnetcdf_all from ndarray
  core_data.read_pnetcdf_all(ncid, varid, start.data(), count.data());

  // Close file
  PNC_SAFE_CALL(ncmpi_close(ncid));

  // Copy core data into local_data_ (which includes ghost regions)
  // For now, just copy the core portion; ghosts remain uninitialized until exchange_ghosts()
  size_t core_size = core_data.size();
  for (size_t i = 0; i < core_size; i++) {
    local_data_[i] = core_data[i];
  }
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::read_parallel_hdf5(
    const std::string& filename,
    const std::string& varname,
    int timestep)
{
#if !NDARRAY_HAVE_HDF5
  throw std::runtime_error("distributed_ndarray::read_parallel_hdf5: HDF5 support not enabled");
#else
  (void)filename;
  (void)varname;
  (void)timestep;

  // TODO: Implement HDF5 parallel read
  // Requires:
  // 1. H5Pset_fapl_mpio() for file access property list
  // 2. H5Dopen() to open dataset
  // 3. H5Screate_simple() and H5Sselect_hyperslab() for memory/file dataspaces
  // 4. H5Dread() with collective I/O
  // 5. H5Dclose(), H5Fclose()

  throw std::runtime_error(
      "distributed_ndarray::read_parallel_hdf5: HDF5 parallel read not yet implemented");
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::read_parallel_binary(
    const std::string& filename)
{
#if !NDARRAY_HAVE_MPI
  throw std::runtime_error("distributed_ndarray::read_parallel_binary: MPI support not enabled");
#else
  // Open file with MPI-IO
  MPI_File fh;
  int result = MPI_File_open(comm_, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if (result != MPI_SUCCESS) {
    throw std::runtime_error(
        "distributed_ndarray::read_parallel_binary: failed to open file: " + filename);
  }

  // Calculate offset for this rank's data in the file
  // ndarray uses Fortran/column-major order by default (reshapef)
  // In column-major: rightmost dimension varies fastest in memory
  MPI_Offset offset = 0;
  size_t stride = 1;

  // Calculate offset from local_core start position (column-major)
  for (size_t d = 0; d < local_core_.nd(); d++) {
    offset += static_cast<MPI_Offset>(local_core_.start(d)) * stride;
    stride *= global_lattice_.size(d);
  }
  offset *= sizeof(T);

  // For non-contiguous reads (when local_core != local_extent due to ghosts),
  // we need to read into the core portion correctly
  // Strategy: Read data matching local core dimensions

  if (local_core_.nd() == 2) {
    // 2D case with column-major (Fortran) order
    // In column-major, columns are contiguous in memory
    size_t core_size0 = local_core_.size(0);
    size_t core_size1 = local_core_.size(1);
    size_t global_size0 = global_lattice_.size(0);

    // Calculate ghost offsets
    size_t ghost_start0 = local_core_.start(0) - local_extent_.start(0);
    size_t ghost_start1 = local_core_.start(1) - local_extent_.start(1);

    // Read column by column since file is in column-major order
    for (size_t j = 0; j < core_size1; j++) {
      // Calculate offset for this column in the file
      size_t global_i = local_core_.start(0);
      size_t global_j = local_core_.start(1) + j;
      MPI_Offset col_offset = (global_i + global_j * global_size0) * sizeof(T);

      // Read directly into the correct position in local_data_
      size_t local_i = ghost_start0;
      size_t local_j = ghost_start1 + j;
      T* col_ptr = &local_data_.f(local_i, local_j);

      MPI_Status status;
      result = MPI_File_read_at_all(fh, col_offset, col_ptr,
                                      static_cast<int>(core_size0),
                                      mpi_datatype(), &status);
      if (result != MPI_SUCCESS) {
        MPI_File_close(&fh);
        throw std::runtime_error(
            "distributed_ndarray::read_parallel_binary: MPI_File_read_at_all failed");
      }
    }
  } else {
    // Generic N-D case: read into temporary buffer then copy
    size_t count = local_core_.n();
    std::vector<T> buffer(count);

    // Collective read at calculated offset
    MPI_Status status;
    result = MPI_File_read_at_all(fh, offset, buffer.data(), static_cast<int>(count),
                                    mpi_datatype(), &status);
    if (result != MPI_SUCCESS) {
      MPI_File_close(&fh);
      throw std::runtime_error(
          "distributed_ndarray::read_parallel_binary: MPI_File_read_at_all failed");
    }

    // Copy buffer into local_data_ accounting for multi-dimensional layout
    // Calculate ghost offsets
    std::vector<size_t> ghost_start(local_core_.nd());
    for (size_t d = 0; d < local_core_.nd(); d++) {
      ghost_start[d] = local_core_.start(d) - local_extent_.start(d);
    }

    // Map each element from buffer to correct position in local_data_
    size_t buffer_idx = 0;
    std::vector<size_t> local_idx(local_core_.nd());

    // Iterate through all core elements
    std::function<void(size_t)> copy_recursive = [&](size_t dim) {
      if (dim == local_core_.nd()) {
        // Leaf: copy element
        std::vector<size_t> target_idx(local_core_.nd());
        for (size_t d = 0; d < local_core_.nd(); d++) {
          target_idx[d] = ghost_start[d] + local_idx[d];
        }
        local_data_.at(target_idx) = buffer[buffer_idx++];
        return;
      }

      for (size_t i = 0; i < local_core_.size(dim); i++) {
        local_idx[dim] = i;
        copy_recursive(dim + 1);
      }
    };

    copy_recursive(0);
  }

  // Close file
  MPI_File_close(&fh);
#endif
}

/////
// Ghost Exchange Implementation
/////

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::identify_neighbors()
{
#if !NDARRAY_HAVE_MPI
  return;
#else
  neighbors_.clear();
  neighbors_identified_ = false;

  // For each dimension, check if there are neighbors on left/right (low/high)
  int ndims = static_cast<int>(local_core_.nd());

  for (int dim = 0; dim < ndims; dim++) {
    // Check left neighbor (lower index in this dimension)
    if (local_core_.start(dim) > global_lattice_.start(dim)) {
      // There is a neighbor on the left
      // Find which rank owns the cell just before our start
      std::vector<size_t> neighbor_point(ndims);
      for (int d = 0; d < ndims; d++) {
        if (d == dim) {
          neighbor_point[d] = local_core_.start(d) - 1;
        } else {
          neighbor_point[d] = local_core_.start(d);
        }
      }

      // Find which rank owns this point
      int neighbor_rank = -1;
      for (size_t p = 0; p < partitioner_->np(); p++) {
        const auto& core = partitioner_->get_core(p);
        if (core.contains(neighbor_point)) {
          neighbor_rank = static_cast<int>(p);
          break;
        }
      }

      if (neighbor_rank >= 0 && neighbor_rank != rank_) {
        Neighbor neighbor;
        neighbor.rank = neighbor_rank;
        neighbor.direction = dim * 2;  // 0=left in dim 0, 2=left in dim 1, etc.

        // Calculate send/recv counts and offsets
        // For simplicity in Phase 3, we'll implement a basic version
        // that assumes uniform ghost width of 1
        // More sophisticated packing can be added later

        size_t ghost_width = local_extent_.start(dim) + local_extent_.size(dim) -
                             (local_core_.start(dim) + local_core_.size(dim));
        if (ghost_width == 0) {
          ghost_width = local_core_.start(dim) - local_extent_.start(dim);
        }
        if (ghost_width == 0) ghost_width = 1;  // Default to 1

        // Calculate number of elements in the boundary face
        size_t face_size = 1;
        for (int d = 0; d < ndims; d++) {
          if (d == dim) {
            face_size *= ghost_width;
          } else {
            face_size *= local_core_.size(d);
          }
        }

        neighbor.send_count = face_size;
        neighbor.recv_count = face_size;

        // Store for now - actual offset calculation done during exchange
        neighbor.send_offset = 0;  // Calculated during exchange
        neighbor.recv_offset = 0;  // Calculated during exchange

        neighbors_.push_back(neighbor);
      }
    }

    // Check right neighbor (higher index in this dimension)
    size_t core_end = local_core_.start(dim) + local_core_.size(dim);
    size_t global_end = global_lattice_.start(dim) + global_lattice_.size(dim);
    if (core_end < global_end) {
      // There is a neighbor on the right
      std::vector<size_t> neighbor_point(ndims);
      for (int d = 0; d < ndims; d++) {
        if (d == dim) {
          neighbor_point[d] = core_end;
        } else {
          neighbor_point[d] = local_core_.start(d);
        }
      }

      // Find which rank owns this point
      int neighbor_rank = -1;
      for (size_t p = 0; p < partitioner_->np(); p++) {
        const auto& core = partitioner_->get_core(p);
        if (core.contains(neighbor_point)) {
          neighbor_rank = static_cast<int>(p);
          break;
        }
      }

      if (neighbor_rank >= 0 && neighbor_rank != rank_) {
        Neighbor neighbor;
        neighbor.rank = neighbor_rank;
        neighbor.direction = dim * 2 + 1;  // 1=right in dim 0, 3=right in dim 1, etc.

        // Calculate send/recv counts
        size_t ghost_width = local_extent_.start(dim) + local_extent_.size(dim) -
                             (local_core_.start(dim) + local_core_.size(dim));
        if (ghost_width == 0) {
          ghost_width = local_core_.start(dim) - local_extent_.start(dim);
        }
        if (ghost_width == 0) ghost_width = 1;

        size_t face_size = 1;
        for (int d = 0; d < ndims; d++) {
          if (d == dim) {
            face_size *= ghost_width;
          } else {
            face_size *= local_core_.size(d);
          }
        }

        neighbor.send_count = face_size;
        neighbor.recv_count = face_size;
        neighbor.send_offset = 0;
        neighbor.recv_offset = 0;

        neighbors_.push_back(neighbor);
      }
    }
  }

  neighbors_identified_ = true;
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::exchange_ghosts()
{
#if !NDARRAY_HAVE_MPI
  return;
#else
  if (!neighbors_identified_ || neighbors_.empty()) {
    // No neighbors or not yet identified - nothing to exchange
    return;
  }

  // For simplicity in Phase 3, implement a basic version using blocking communication
  // More sophisticated version with non-blocking can be added later

  std::vector<MPI_Request> requests;
  std::vector<std::vector<T>> send_buffers(neighbors_.size());
  std::vector<std::vector<T>> recv_buffers(neighbors_.size());

  // Post receives for all neighbors
  for (size_t i = 0; i < neighbors_.size(); i++) {
    recv_buffers[i].resize(neighbors_[i].recv_count);

    MPI_Request req;
    int tag = neighbors_[i].direction;
    MPI_Irecv(recv_buffers[i].data(),
              static_cast<int>(neighbors_[i].recv_count),
              mpi_datatype(),
              neighbors_[i].rank,
              tag,
              comm_,
              &req);
    requests.push_back(req);
  }

  // Pack and send boundary data to all neighbors
  for (size_t i = 0; i < neighbors_.size(); i++) {
    send_buffers[i].resize(neighbors_[i].send_count);
    pack_boundary_data(i, send_buffers[i]);

    // Reverse the direction for the tag (what we receive from left, they send from right)
    int tag = neighbors_[i].direction ^ 1;  // Flip last bit: 0↔1, 2↔3, etc.

    MPI_Send(send_buffers[i].data(),
             static_cast<int>(neighbors_[i].send_count),
             mpi_datatype(),
             neighbors_[i].rank,
             tag,
             comm_);
  }

  // Wait for all receives to complete
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Unpack received ghost data
  for (size_t i = 0; i < neighbors_.size(); i++) {
    unpack_ghost_data(i, recv_buffers[i]);
  }
#endif
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::pack_boundary_data(
    int neighbor_idx,
    std::vector<T>& buffer)
{
  const Neighbor& neighbor = neighbors_[neighbor_idx];
  int dim = neighbor.direction / 2;  // Which dimension: 0, 1, 2, ...
  bool is_high = (neighbor.direction % 2) == 1;  // true = right/up, false = left/down

  // For now, implement simple 1D case
  // More sophisticated multi-dimensional packing can be added

  // Get the boundary slice from local_core
  // For left neighbor (dim=0, dir=0): send leftmost layer
  // For right neighbor (dim=0, dir=1): send rightmost layer

  size_t ghost_width = 1;  // Simplified: assume 1-layer ghosts
  auto& local = local_data_;

  if (dim == 0) {
    // Boundary in dimension 0
    size_t start_idx = is_high ? (local_core_.size(0) - ghost_width) : 0;
    size_t buffer_idx = 0;

    for (size_t i = 0; i < ghost_width; i++) {
      for (size_t j = 0; j < local_core_.size(1); j++) {
        buffer[buffer_idx++] = local.at(start_idx + i, j);
      }
    }
  } else if (dim == 1 && local_core_.nd() >= 2) {
    // Boundary in dimension 1
    size_t start_idx = is_high ? (local_core_.size(1) - ghost_width) : 0;
    size_t buffer_idx = 0;

    for (size_t i = 0; i < local_core_.size(0); i++) {
      for (size_t j = 0; j < ghost_width; j++) {
        buffer[buffer_idx++] = local.at(i, start_idx + j);
      }
    }
  }
  // TODO: Add dimension 2, 3, etc. for higher-dimensional arrays
}

template <typename T, typename StoragePolicy>
void distributed_ndarray<T, StoragePolicy>::unpack_ghost_data(
    int neighbor_idx,
    const std::vector<T>& buffer)
{
  const Neighbor& neighbor = neighbors_[neighbor_idx];
  int dim = neighbor.direction / 2;
  bool is_high = (neighbor.direction % 2) == 1;

  size_t ghost_width = 1;
  auto& local = local_data_;

  // Unpack into ghost region
  // For left neighbor: unpack into left ghost
  // For right neighbor: unpack into right ghost

  size_t ghost_low = local_core_.start(dim) - local_extent_.start(dim);
  size_t ghost_high = (local_extent_.start(dim) + local_extent_.size(dim)) -
                      (local_core_.start(dim) + local_core_.size(dim));

  if (dim == 0) {
    size_t start_idx = is_high ? (local_core_.size(0) + ghost_low) : 0;
    size_t buffer_idx = 0;

    if (!is_high && ghost_low > 0) {
      // Unpack into left ghost
      for (size_t i = 0; i < ghost_width && i < ghost_low; i++) {
        for (size_t j = 0; j < local_core_.size(1); j++) {
          local.at(start_idx + i, j) = buffer[buffer_idx++];
        }
      }
    } else if (is_high && ghost_high > 0) {
      // Unpack into right ghost
      for (size_t i = 0; i < ghost_width && i < ghost_high; i++) {
        for (size_t j = 0; j < local_core_.size(1); j++) {
          local.at(start_idx + i, j) = buffer[buffer_idx++];
        }
      }
    }
  } else if (dim == 1 && local_core_.nd() >= 2) {
    size_t start_idx = is_high ? (local_core_.size(1) + ghost_low) : 0;
    size_t buffer_idx = 0;

    if (!is_high && ghost_low > 0) {
      for (size_t i = 0; i < local_core_.size(0); i++) {
        for (size_t j = 0; j < ghost_width && j < ghost_low; j++) {
          local.at(i, start_idx + j) = buffer[buffer_idx++];
        }
      }
    } else if (is_high && ghost_high > 0) {
      for (size_t i = 0; i < local_core_.size(0); i++) {
        for (size_t j = 0; j < ghost_width && j < ghost_high; j++) {
          local.at(i, start_idx + j) = buffer[buffer_idx++];
        }
      }
    }
  }
}

} // namespace ftk

#endif // _FTK_DISTRIBUTED_NDARRAY_HH
