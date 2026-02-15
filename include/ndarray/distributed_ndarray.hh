#ifndef _FTK_DISTRIBUTED_NDARRAY_HH
#define _FTK_DISTRIBUTED_NDARRAY_HH

#include <ndarray/ndarray.hh>
#include <ndarray/lattice.hh>
#include <ndarray/lattice_partitioner.hh>

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

  // TODO: Phase 2 continuation
  // void write_parallel(const std::string& filename, const std::string& varname, int timestep = 0);

  // TODO: Ghost exchange (Phase 3)
  // void exchange_ghosts();

private:
  // Format-specific parallel read methods
  void read_parallel_netcdf(const std::string& filename, const std::string& varname, int timestep);
  void read_parallel_hdf5(const std::string& filename, const std::string& varname, int timestep);
  void read_parallel_binary(const std::string& filename);

  // Helper to get MPI datatype for T
  static MPI_Datatype mpi_datatype();

  // Helper to get file extension
  static std::string get_file_extension(const std::string& filename);

  MPI_Comm comm_;
  int rank_;
  int nprocs_;

  lattice global_lattice_;        // Full global domain
  lattice_partitioner partitioner_;

  lattice local_core_;            // This rank's core region (no ghosts)
  lattice local_extent_;          // This rank's extent region (core + ghosts)

  ndarray<T, StoragePolicy> local_data_;  // Local data (allocated to extent size)

  // TODO: Neighbor information for ghost exchange (Phase 3)
  // std::vector<int> neighbor_ranks_;
  // std::vector<lattice> send_regions_;
  // std::vector<lattice> recv_regions_;
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
  partitioner_ = lattice_partitioner(global_lattice_);

  if (decomp.empty() && ghost.empty()) {
    // Automatic decomposition, no ghosts
    partitioner_.partition(nprocs);
  } else if (!decomp.empty() && ghost.empty()) {
    // User-specified decomposition, no ghosts
    partitioner_.partition(nprocs, decomp);
  } else if (!decomp.empty() && !ghost.empty()) {
    // User-specified decomposition with ghosts
    partitioner_.partition(nprocs, decomp, ghost);
  } else {
    // Automatic decomposition with ghosts
    partitioner_.partition(nprocs, std::vector<size_t>(), ghost);
  }

  // Verify decomposition succeeded
  if (partitioner_.np() == 0) {
    throw std::runtime_error(
        "distributed_ndarray::decompose: lattice_partitioner failed to decompose domain");
  }

  if (partitioner_.np() != nprocs) {
    throw std::runtime_error(
        "distributed_ndarray::decompose: number of partitions does not match nprocs");
  }

  // Store this rank's core and extent lattices
  local_core_ = partitioner_.get_core(rank_);
  local_extent_ = partitioner_.get_ext(rank_);

  // Allocate local data to extent size
  std::vector<size_t> extent_dims = local_extent_.sizes();
  local_data_.reshapef(extent_dims);

  // TODO Phase 3: Identify neighbors for ghost exchange
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
  // Assumes global array is stored contiguously in row-major (C) order
  MPI_Offset offset = 0;
  size_t stride = 1;

  // Calculate offset from local_core start position
  for (int d = static_cast<int>(local_core_.nd()) - 1; d >= 0; d--) {
    offset += static_cast<MPI_Offset>(local_core_.start(d)) * stride;
    stride *= global_lattice_.size(d);
  }
  offset *= sizeof(T);

  // Calculate number of elements to read (core only)
  size_t count = local_core_.n();

  // Create temporary buffer for core data
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

  // Copy buffer into local_data_ (core portion)
  for (size_t i = 0; i < count; i++) {
    local_data_[i] = buffer[i];
  }

  // Close file
  MPI_File_close(&fh);
#endif
}

} // namespace ftk

#endif // _FTK_DISTRIBUTED_NDARRAY_HH
