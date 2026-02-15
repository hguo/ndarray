#ifndef _DISTRIBUTED_NDARRAY_GROUP_STREAM_HH
#define _DISTRIBUTED_NDARRAY_GROUP_STREAM_HH

/**
 * @file distributed_ndarray_group_stream.hh
 * @brief Unified header for distributed memory time-varying scientific data
 *
 * This is a convenience header that includes all distributed ndarray functionality
 * for YAML-based stream processing with MPI domain decomposition.
 *
 * Includes:
 * - distributed_ndarray: MPI-aware arrays with decomposition and ghost exchange
 * - distributed_ndarray_group: Container for multiple distributed arrays
 * - distributed_stream: YAML-configured stream with parallel I/O
 *
 * Example Usage:
 * @code
 * #include <ndarray/distributed_ndarray_group_stream.hh>
 *
 * ftk::distributed_stream<> stream(MPI_COMM_WORLD);
 * stream.parse_yaml("config.yaml");
 *
 * for (int t = 0; t < stream.n_timesteps(); t++) {
 *   auto group = stream.read(t);
 *   group->exchange_ghosts_all();
 *   // process data...
 * }
 * @endcode
 *
 * YAML Configuration Example:
 * @code{.yaml}
 * decomposition:
 *   global_dims: [1000, 800, 600]
 *   ghost: [1, 1, 1]
 *
 * streams:
 *   - name: simulation
 *     format: netcdf
 *     filenames: data_*.nc
 *     vars:
 *       - name: temperature
 *       - name: pressure
 *       - name: velocity
 * @endcode
 */

#include <ndarray/config.hh>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

// Core distributed array functionality
#include <ndarray/distributed_ndarray.hh>

// Group and stream for YAML-based workflows
#include <ndarray/distributed_ndarray_group.hh>
#include <ndarray/distributed_ndarray_stream.hh>

// Underlying stream infrastructure (needed for substream support)
#include <ndarray/ndarray_group_stream.hh>

namespace ftk {

/**
 * @brief Type alias for distributed stream with default float type
 *
 * Common type alias for convenience, equivalent to:
 * ftk::distributed_stream<float, native_storage>
 */
template <typename StoragePolicy = native_storage>
using default_distributed_stream = distributed_stream<float, StoragePolicy>;

/**
 * @brief Type alias for distributed group with default float type
 *
 * Common type alias for convenience, equivalent to:
 * ftk::distributed_ndarray_group<float, native_storage>
 */
template <typename StoragePolicy = native_storage>
using default_distributed_group = distributed_ndarray_group<float, StoragePolicy>;

} // namespace ftk

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#endif // _DISTRIBUTED_NDARRAY_GROUP_STREAM_HH
