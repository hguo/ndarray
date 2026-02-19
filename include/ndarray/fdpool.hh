#ifndef _NDARRAY_FDPOOL_H
#define _NDARRAY_FDPOOL_H

#include <ndarray/config.hh>
#include <ndarray/error.hh>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif
#if NDARRAY_HAVE_NETCDF
#include <netcdf.h>
#endif

#include <map>

namespace ftk {

/**
 * @brief NetCDF file descriptor pool (singleton)
 *
 * fdpool_nc manages a pool of open NetCDF file descriptors to avoid
 * opening the same file multiple times simultaneously. This is critical
 * because:
 *
 * 1. NetCDF library behavior: Opening the same file multiple times can lead to:
 *    - File corruption (especially with parallel NetCDF)
 *    - Race conditions on file metadata
 *    - Resource exhaustion (file descriptor limits)
 *    - Undefined behavior with concurrent writes
 *
 * 2. Stream usage pattern: When using ndarray_group_stream, the same NetCDF
 *    file may be accessed multiple times within a single program:
 *    - Reading different variables from the same file
 *    - Reading different timesteps from the same file
 *    - Multiple substreams referencing the same file
 *
 * 3. Performance: Reusing file descriptors avoids repeated open/close overhead
 *
 * Implementation:
 * - Singleton pattern ensures one global pool per process
 * - Maps filename -> ncid (NetCDF file identifier)
 * - First open() call opens the file and caches ncid
 * - Subsequent open() calls return cached ncid
 * - close_all() closes all files in pool (called by ndarray_finalize())
 *
 * Thread safety: NOT thread-safe. Assumes single-threaded or externally
 * synchronized access. For multi-threaded applications, add mutex protection.
 *
 * MPI behavior: Each MPI rank maintains its own pool. For parallel NetCDF,
 * open() uses nc_open_par() if available, allowing collective I/O.
 *
 * Example usage:
 * @code
 * // Typical usage (internal to ndarray_group_stream):
 * auto& pool = fdpool_nc::get_instance();
 * int ncid = pool.open("data.nc");  // Opens file (or returns cached ncid)
 *
 * // Read variables using ncid...
 * nc_get_var_float(ncid, varid, data);
 *
 * // No need to close - fdpool manages lifecycle
 *
 * // At program exit:
 * ftk::ndarray_finalize();  // Closes all files in pool
 * @endcode
 *
 * @see ndarray_finalize() in util.hh for cleanup
 * @see ndarray_group_stream for usage context
 */
struct fdpool_nc {
  // Singleton pattern - prevent copying
  fdpool_nc(fdpool_nc const&) = delete;
  void operator=(fdpool_nc const&) = delete;

  /**
   * @brief Get the singleton instance
   * @return Reference to the global fdpool_nc instance
   */
  static fdpool_nc& get_instance() {
    static fdpool_nc instance;
    return instance;
  }

  /**
   * @brief Open NetCDF file or return cached file descriptor
   *
   * Opens the NetCDF file on first call for a given filename. Subsequent
   * calls with the same filename return the cached ncid without reopening.
   *
   * @param filename Path to NetCDF file (absolute or relative)
   * @param comm MPI communicator for parallel NetCDF (default: MPI_COMM_WORLD)
   * @return NetCDF file identifier (ncid) for use with NetCDF API
   *
   * @note If NDARRAY_HAVE_NETCDF_PARALLEL is enabled, attempts parallel open first,
   *       falls back to serial open if parallel open fails.
   * @note Throws/exits on open failure via NC_SAFE_CALL macro
   *
   * @warning Do NOT call nc_close() on returned ncid - use close_all() instead
   */
  int open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);

  /**
   * @brief Close all NetCDF files in the pool
   *
   * Closes all cached NetCDF file descriptors and clears the pool.
   * Should be called at program exit via ndarray_finalize().
   *
   * @note Safe to call multiple times (idempotent)
   * @note Automatically called by ndarray_finalize() in util.hh
   */
  void close_all();

private:
  fdpool_nc() {}  // Private constructor for singleton

  /**
   * @brief Pool of open file descriptors
   * Key: filename (string)
   * Value: ncid (NetCDF file identifier)
   */
  std::map<std::string, int> pool;
};

////
// Implementation
////

inline int fdpool_nc::open(const std::string& f, MPI_Comm comm)
{
  // Check if file already opened
  auto it = pool.find(f);
  if (it == pool.end()) {
    // File not in pool - open it for the first time
    int ncid, rtn;

#if NDARRAY_HAVE_NETCDF
#if NDARRAY_HAVE_NETCDF_PARALLEL
    // Try parallel open first (requires NetCDF built with parallel support)
    rtn = nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
    if (rtn != NC_NOERR) {
      // Parallel open failed - fall back to serial open
      // This can happen if file not accessible in parallel mode or
      // NetCDF-4 features conflict with parallel access
      NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
    }
#else
    // Parallel NetCDF not available - use serial open
    NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#endif
#endif

    // Debug output (commented out by default)
    // fprintf(stderr, "[fdpool] opened netcdf file %s, ncid=%d.\n", f.c_str(), ncid);

    // Cache the file descriptor
    pool[f] = ncid;
    return ncid;
  } else {
    // File already in pool - return cached ncid
    // This avoids double-opening the same file
    return it->second;
  }
}

inline void fdpool_nc::close_all()
{
  // Close all cached file descriptors
  for (const auto &kv : pool) {
#if NDARRAY_HAVE_NETCDF
    // Debug output (commented out by default)
    // fprintf(stderr, "[fdpool] closing netcdf file %s, ncid=%d.\n", kv.first.c_str(), kv.second);

    // Close the NetCDF file
    NC_SAFE_CALL( nc_close(kv.second) );
#endif
  }

  // Clear the pool after closing all files
  // This allows open() to work again if called after close_all()
  pool.clear();
}

} // namespace

#endif
