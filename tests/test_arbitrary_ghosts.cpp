/**
 * Arbitrary Ghost Widths Tests
 *
 * Tests ghost cell exchange with non-uniform ghost widths per dimension.
 *
 * Tests verify:
 * - Ghost exchange with variable widths (e.g., 2 in dim 0, 1 in dim 1)
 * - Corner propagation across multiple passes
 * - Data correctness in multi-layer ghost regions
 *
 * Run with: mpirun -np 4 ./test_arbitrary_ghosts
 */

#include <ndarray/config.hh>
#include <iostream>

#if NDARRAY_HAVE_MPI

#include <ndarray/ndarray.hh>
#include <mpi.h>
#include <cassert>
#include <vector>
#include <numeric>

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "[Rank " << rank << "] FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while (0)

#define TEST_SECTION(name) \
  if (rank == 0) std::cout << "  " << name << std::endl

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test: Arbitrary Ghost Widths and Corner Propagation ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI ranks" << std::endl;
  }

  // We need at least 4 ranks for a 2x2 decomposition to test corners
  if (nprocs < 4) {
    if (rank == 0) std::cout << "Skipping corner test (requires >= 4 ranks)" << std::endl;
    // But we can still test variable widths in 1D if nprocs >= 2
  }

  TEST_SECTION("Create 2D decomposition with non-uniform ghosts (2 in dim 0, 1 in dim 1)");
  ftk::ndarray<int> darray;
  
  // Use a 2x2 split if possible, else 1D
  std::vector<size_t> decomp;
  if (nprocs >= 4) {
    decomp = {2, 2};
  } else {
    decomp = {static_cast<size_t>(nprocs), 1};
  }

  // Set ghost widths: 2 layers in dim 0, 1 layer in dim 1
  std::vector<size_t> ghost_widths = {2, 1};
  
  darray.decompose(MPI_COMM_WORLD, 
                   {20, 20}, 
                   static_cast<size_t>(nprocs), 
                   decomp, 
                   ghost_widths);

  TEST_SECTION("Initialize core with global indices");
  // Fill core with value = global_i * 100 + global_j
  const auto& core = darray.local_core();
  const auto& extent = darray.local_extent();
  
  // Local dimensions (including ghosts)
  size_t loc_ni = extent.size(0);
  size_t loc_nj = extent.size(1);
  
  // Initialize everything to -1
  darray.fill(-1);

  // Calculate local-to-extent offsets
  size_t off_i = core.start(0) - extent.start(0);
  size_t off_j = core.start(1) - extent.start(1);

  for (size_t i = 0; i < core.size(0); i++) {
    for (size_t j = 0; j < core.size(1); j++) {
      size_t global_i = core.start(0) + i;
      size_t global_j = core.start(1) + j;
      darray.f(off_i + i, off_j + j) = static_cast<int>(global_i * 100 + global_j);
    }
  }

  TEST_SECTION("Exchange ghosts (multi-pass)");
  darray.exchange_ghosts();

  TEST_SECTION("Verify ghost regions");
  bool success = true;
  
  // For each point in the local extent (including ghosts)
  for (size_t i = 0; i < loc_ni; i++) {
    for (size_t j = 0; j < loc_nj; j++) {
      size_t global_i = extent.start(0) + i;
      size_t global_j = extent.start(1) + j;
      
      // If the point is within global boundaries
      if (global_i < 20 && global_j < 20) {
        int expected = static_cast<int>(global_i * 100 + global_j);
        int actual = darray.f(i, j);
        
        if (actual != expected) {
          // If it's a ghost cell, it should have been filled
          // (Assuming the point belongs to some rank's core)
          std::cerr << "[Rank " << rank << "] Mismatch at local(" << i << "," << j << "), "
                    << "global(" << global_i << "," << global_j << "): "
                    << "expected " << expected << ", got " << actual << std::endl;
          success = false;
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (success) {
    if (rank == 0) std::cout << "  ✓ All ghost cells (including corners and multi-layer) correctly filled" << std::endl;
  } else {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "Ghost exchange tests require MPI support" << std::endl;
  return 0;
}

#endif // NDARRAY_HAVE_MPI
