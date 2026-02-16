/**
 * Reference Ghost Exchange Implementation
 *
 * Serial, obviously-correct implementation for validating distributed ghost exchange.
 * This is intentionally slow but straightforward for verification purposes.
 *
 * NOT for production use - for testing only!
 */

#include <ndarray/ndarray.hh>
#include <ndarray/lattice.hh>
#include <ndarray/lattice_partitioner.hh>
#include <vector>
#include <iostream>

namespace ftk {
namespace reference {

/**
 * Reference ghost exchange for 2D arrays
 *
 * Takes a serial (replicated) array and simulates ghost exchange by:
 * 1. Partitioning the array into subdomains
 * 2. For each subdomain's ghost cells, copying from the owning subdomain's core
 *
 * This is the "obviously correct" way to do ghost exchange - no MPI, no parallelism,
 * just straightforward copying. Use to validate the actual distributed implementation.
 *
 * @param data - Flattened 2D array in Fortran order (column-major)
 * @param global_dims - Global array dimensions [nx, ny]
 * @param nprocs - Number of processes to simulate
 * @param decomp - Decomposition pattern (empty = auto)
 * @param ghost - Ghost width per dimension
 * @return Vector of local arrays (one per rank) with ghosts populated
 */
template <typename T>
std::vector<ftk::ndarray<T>> reference_ghost_exchange_2d(
    const ftk::ndarray<T>& global_data,
    const std::vector<size_t>& global_dims,
    size_t nprocs,
    const std::vector<size_t>& decomp = {},
    const std::vector<size_t>& ghost = {1, 1})
{
  if (global_dims.size() != 2) {
    throw std::invalid_argument("reference_ghost_exchange_2d: requires 2D array");
  }

  // Create global lattice
  lattice global_lattice(global_dims);

  // Create partitioner
  lattice_partitioner partitioner(global_lattice);
  partitioner.partition(nprocs, decomp, ghost);

  // Result: one local array per rank
  std::vector<ftk::ndarray<T>> local_arrays(nprocs);

  // For each rank, create local array with extent (core + ghosts)
  for (size_t rank = 0; rank < nprocs; rank++) {
    const auto& core = partitioner.get_core(rank);
    const auto& extent = partitioner.get_ext(rank);

    // Allocate local array with extent dimensions
    auto& local = local_arrays[rank];
    local.reshapef(extent.size(0), extent.size(1));
    local.fill(static_cast<T>(0));  // Initialize with zeros

    // Step 1: Copy core data from global array
    for (size_t i = 0; i < core.size(0); i++) {
      for (size_t j = 0; j < core.size(1); j++) {
        size_t global_i = core.start(0) + i;
        size_t global_j = core.start(1) + j;

        // Local index = global index - extent start
        size_t local_i = global_i - extent.start(0);
        size_t local_j = global_j - extent.start(1);

        // Copy from global to local
        local.f(local_i, local_j) = global_data.f(global_i, global_j);
      }
    }

    // Step 2: Populate ghost cells
    // For each cell in extent that's NOT in core (i.e., ghost cells)
    for (size_t local_i = 0; local_i < extent.size(0); local_i++) {
      for (size_t local_j = 0; local_j < extent.size(1); local_j++) {
        size_t global_i = extent.start(0) + local_i;
        size_t global_j = extent.start(1) + local_j;

        // Check if this is a ghost cell (not in core)
        bool in_core = (global_i >= core.start(0) && global_i < core.start(0) + core.size(0) &&
                        global_j >= core.start(1) && global_j < core.start(1) + core.size(1));

        if (!in_core) {
          // This is a ghost cell - find which rank owns this global point
          // and copy from that rank's core

          // Check if global point is within global domain
          if (global_i < global_dims[0] && global_j < global_dims[1]) {
            // Find owning rank by checking all ranks' cores
            for (size_t owner_rank = 0; owner_rank < nprocs; owner_rank++) {
              const auto& owner_core = partitioner.get_core(owner_rank);

              bool is_owned = (global_i >= owner_core.start(0) &&
                               global_i < owner_core.start(0) + owner_core.size(0) &&
                               global_j >= owner_core.start(1) &&
                               global_j < owner_core.start(1) + owner_core.size(1));

              if (is_owned) {
                // Copy from global data (which has the correct values)
                local.f(local_i, local_j) = global_data.f(global_i, global_j);
                break;
              }
            }
          }
          // If global point is outside domain, leave as zero (boundary ghost)
        }
      }
    }
  }

  return local_arrays;
}

/**
 * Verify that distributed ghost exchange matches reference implementation
 *
 * @param distributed_arrays - Arrays after distributed ghost exchange (from MPI ranks)
 * @param reference_arrays - Arrays from reference implementation
 * @param tolerance - Floating-point comparison tolerance
 * @return true if all arrays match within tolerance
 */
template <typename T>
bool verify_ghost_exchange(
    const std::vector<ftk::ndarray<T>>& distributed_arrays,
    const std::vector<ftk::ndarray<T>>& reference_arrays,
    T tolerance = static_cast<T>(1e-6))
{
  if (distributed_arrays.size() != reference_arrays.size()) {
    std::cerr << "verify_ghost_exchange: array count mismatch ("
              << distributed_arrays.size() << " vs " << reference_arrays.size() << ")" << std::endl;
    return false;
  }

  bool all_match = true;

  for (size_t rank = 0; rank < distributed_arrays.size(); rank++) {
    const auto& dist = distributed_arrays[rank];
    const auto& ref = reference_arrays[rank];

    // Check dimensions match
    if (dist.nd() != ref.nd() || dist.size() != ref.size()) {
      std::cerr << "Rank " << rank << ": dimension mismatch" << std::endl;
      all_match = false;
      continue;
    }

    // Check all values match
    for (size_t i = 0; i < dist.size(); i++) {
      T dist_val = dist[i];
      T ref_val = ref[i];

      if (std::abs(dist_val - ref_val) > tolerance) {
        if (all_match) {  // Only print first few mismatches
          std::cerr << "Rank " << rank << " index " << i << ": "
                    << "distributed=" << dist_val << ", reference=" << ref_val
                    << ", diff=" << (dist_val - ref_val) << std::endl;
        }
        all_match = false;
      }
    }
  }

  return all_match;
}

/**
 * Create a test global array with known pattern
 * Pattern: value = i * multiplier + j
 */
template <typename T>
ftk::ndarray<T> create_test_array(size_t nx, size_t ny, T multiplier = 100)
{
  ftk::ndarray<T> arr;
  arr.reshapef(nx, ny);

  for (size_t i = 0; i < nx; i++) {
    for (size_t j = 0; j < ny; j++) {
      arr.f(i, j) = static_cast<T>(i) * multiplier + static_cast<T>(j);
    }
  }

  return arr;
}

} // namespace reference
} // namespace ftk

// Example usage and tests
#ifdef REFERENCE_GHOST_EXCHANGE_MAIN

#include <iostream>

int main() {
  std::cout << "Reference Ghost Exchange Test" << std::endl;
  std::cout << "=============================" << std::endl;

  // Create test array
  const size_t nx = 10, ny = 8;
  auto global_data = ftk::reference::create_test_array<float>(nx, ny);

  std::cout << "\nGlobal array: " << nx << " × " << ny << std::endl;
  std::cout << "Sample values:" << std::endl;
  for (size_t i = 0; i < std::min(size_t(3), nx); i++) {
    for (size_t j = 0; j < std::min(size_t(3), ny); j++) {
      std::cout << "  [" << i << "," << j << "] = " << global_data.f(i, j) << std::endl;
    }
  }

  // Test with 2 ranks
  std::cout << "\n--- Test with 2 ranks ---" << std::endl;
  auto local_2ranks = ftk::reference::reference_ghost_exchange_2d(
      global_data, {nx, ny}, 2, {2, 0}, {1, 1});

  for (size_t rank = 0; rank < local_2ranks.size(); rank++) {
    const auto& local = local_2ranks[rank];
    std::cout << "Rank " << rank << ": "
              << local.dimf(0) << " × " << local.dimf(1)
              << " (size=" << local.size() << ")" << std::endl;

    // Print first few values
    std::cout << "  First 3×3 values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), local.dimf(0)); i++) {
      std::cout << "    ";
      for (size_t j = 0; j < std::min(size_t(3), local.dimf(1)); j++) {
        std::cout << local.f(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }

  // Test with 4 ranks
  std::cout << "\n--- Test with 4 ranks (2×2 decomposition) ---" << std::endl;
  auto local_4ranks = ftk::reference::reference_ghost_exchange_2d(
      global_data, {nx, ny}, 4, {2, 2}, {1, 1});

  for (size_t rank = 0; rank < local_4ranks.size(); rank++) {
    const auto& local = local_4ranks[rank];
    std::cout << "Rank " << rank << ": "
              << local.dimf(0) << " × " << local.dimf(1) << std::endl;
  }

  std::cout << "\n✓ Reference implementation completed successfully" << std::endl;
  return 0;
}

#endif // REFERENCE_GHOST_EXCHANGE_MAIN
