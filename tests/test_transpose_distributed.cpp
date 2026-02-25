/**
 * @file test_transpose_distributed.cpp
 * @brief Comprehensive tests for distributed array transpose functionality
 *
 * Tests transpose operations on MPI-distributed arrays including:
 * - 2D and 3D arrays with various decompositions
 * - Arrays with component dimensions
 * - Time-varying arrays
 * - Ghost layers
 * - Error conditions
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace ftk;

// Test utilities
#define TEST_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "[FAILED] " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while(0)

#define TEST_SUCCESS(msg) \
  do { \
    if (rank == 0) std::cout << "[PASSED] " << msg << std::endl; \
  } while(0)

/**
 * @brief Initialize array with pattern: value = sum of global indices
 */
template <typename T>
void init_array_with_pattern(ndarray<T>& arr) {
  const size_t nd = arr.nd();
  const lattice& local_core = arr.local_core();

  std::vector<size_t> idx(nd);
  for (size_t i = 0; i < arr.nelem(); i++) {
    // Get local index
    size_t tmp = i;
    for (size_t d = 0; d < nd; d++) {
      idx[d] = tmp % arr.dimf(d);
      tmp /= arr.dimf(d);
    }

    // Convert to global index
    auto global_idx = arr.local_to_global(idx);

    // Compute value as sum of global indices
    T value = 0;
    for (size_t d = 0; d < nd; d++) {
      value += global_idx[d];
    }

    arr[i] = value;
  }
}

/**
 * @brief Verify transposed array has correct values
 */
template <typename T>
bool verify_transposed_pattern(const ndarray<T>& arr, const std::vector<size_t>& axes) {
  const size_t nd = arr.nd();

  for (size_t i = 0; i < arr.nelem(); i++) {
    // Get local index in transposed array
    std::vector<size_t> transposed_local_idx(nd);
    size_t tmp = i;
    for (size_t d = 0; d < nd; d++) {
      transposed_local_idx[d] = tmp % arr.dimf(d);
      tmp /= arr.dimf(d);
    }

    // Convert to global index (in transposed coordinates)
    auto transposed_global_idx = arr.local_to_global(transposed_local_idx);

    // Apply inverse permutation to get original global index
    std::vector<size_t> original_global_idx(nd);
    for (size_t d = 0; d < nd; d++) {
      original_global_idx[axes[d]] = transposed_global_idx[d];
    }

    // Expected value is sum of original global indices
    T expected = 0;
    for (size_t d = 0; d < nd; d++) {
      expected += original_global_idx[d];
    }

    if (std::abs(arr[i] - expected) > 1e-10) {
      return false;
    }
  }

  return true;
}

/**
 * Test 1: 2D transpose with 1D decomposition
 */
void test_2d_transpose_1d_decomp(MPI_Comm comm, int rank, int nprocs) {
  // Create 100x80 array, decompose along dimension 0
  ndarray<double> arr;
  arr.decompose(comm, {100, 80}, nprocs, {static_cast<size_t>(nprocs), 0}, {0, 0});

  init_array_with_pattern(arr);

  // Transpose (swap dimensions)
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify dimensions
  TEST_ASSERT(transposed.is_distributed(), "Result should be distributed");
  TEST_ASSERT(transposed.global_lattice().size(0) == 80, "New dim 0 should be 80");
  TEST_ASSERT(transposed.global_lattice().size(1) == 100, "New dim 1 should be 100");

  // Verify decomposition pattern is permuted
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[0] == 0, "New dim 0 should not be decomposed");
  TEST_ASSERT(new_decomp[1] == static_cast<size_t>(nprocs), "New dim 1 should be decomposed");

  // Verify data correctness
  TEST_ASSERT(verify_transposed_pattern(transposed, {1, 0}), "Transposed data should be correct");

  TEST_SUCCESS("2D transpose with 1D decomposition");
}

/**
 * Test 2: 2D transpose with 2D decomposition
 */
void test_2d_transpose_2d_decomp(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs != 4) {
    if (rank == 0) std::cout << "[SKIPPED] 2D transpose with 2D decomposition (requires 4 ranks)" << std::endl;
    return;
  }

  // Create 120x80 array, decompose 2x2
  ndarray<double> arr;
  arr.decompose(comm, {120, 80}, nprocs, {2, 2}, {0, 0});

  init_array_with_pattern(arr);

  // Transpose
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify dimensions
  TEST_ASSERT(transposed.global_lattice().size(0) == 80, "New dim 0 should be 80");
  TEST_ASSERT(transposed.global_lattice().size(1) == 120, "New dim 1 should be 120");

  // Verify decomposition is swapped
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[0] == 2, "New dim 0 decomp should be 2");
  TEST_ASSERT(new_decomp[1] == 2, "New dim 1 decomp should be 2");

  // Verify data
  TEST_ASSERT(verify_transposed_pattern(transposed, {1, 0}), "Data should be correct");

  TEST_SUCCESS("2D transpose with 2D decomposition");
}

/**
 * Test 3: 3D transpose with 2D decomposition
 */
void test_3d_transpose_2d_decomp(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] 3D transpose (requires at least 2 ranks)" << std::endl;
    return;
  }

  // Create 50x60x70 array, decompose along first 2 dims
  size_t nx = 50, ny = 60, nz = 70;
  std::vector<size_t> decomp = {2, static_cast<size_t>(nprocs)/2, 0};  // 2 x (nprocs/2) x 1
  if (nprocs == 1) decomp = {0, 0, 0};
  if (nprocs == 2) decomp = {2, 0, 0};

  ndarray<double> arr;
  arr.decompose(comm, {nx, ny, nz}, nprocs, decomp, {0, 0, 0});

  init_array_with_pattern(arr);

  // Transpose: swap first two dimensions {1, 0, 2}
  auto transposed = ftk::transpose(arr, {1, 0, 2});

  // Verify dimensions
  TEST_ASSERT(transposed.global_lattice().size(0) == ny, "New dim 0 should be ny");
  TEST_ASSERT(transposed.global_lattice().size(1) == nx, "New dim 1 should be nx");
  TEST_ASSERT(transposed.global_lattice().size(2) == nz, "New dim 2 should be nz");

  // Verify decomposition is permuted
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[0] == decomp[1], "New dim 0 decomp should match old dim 1");
  TEST_ASSERT(new_decomp[1] == decomp[0], "New dim 1 decomp should match old dim 0");
  TEST_ASSERT(new_decomp[2] == decomp[2], "New dim 2 decomp should match old dim 2");

  // Verify data
  TEST_ASSERT(verify_transposed_pattern(transposed, {1, 0, 2}), "Data should be correct");

  TEST_SUCCESS("3D transpose with 2D decomposition");
}

/**
 * Test 4: Vector field transpose (component dimensions)
 */
void test_vector_field_transpose(MPI_Comm comm, int rank, int nprocs) {
  // Create velocity field: [3, 100, 80] (3 components, 100x80 grid)
  ndarray<double> velocity;
  velocity.decompose(comm, {3, 100, 80}, nprocs, {0, static_cast<size_t>(nprocs), 0}, {0, 0, 0});
  velocity.set_multicomponents(1);  // First dim is component

  init_array_with_pattern(velocity);

  // Transpose spatial dimensions only: {0, 2, 1}
  // Component dim stays first, swap spatial dims
  auto transposed = ftk::transpose(velocity, {0, 2, 1});

  // Verify dimensions
  TEST_ASSERT(transposed.global_lattice().size(0) == 3, "Component dim should stay 3");
  TEST_ASSERT(transposed.global_lattice().size(1) == 80, "New spatial dim 0 should be 80");
  TEST_ASSERT(transposed.global_lattice().size(2) == 100, "New spatial dim 1 should be 100");

  // Verify component metadata preserved
  TEST_ASSERT(transposed.multicomponents() == 1, "Multicomponents should be preserved");

  // Verify decomposition
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[0] == 0, "Component dim should not be decomposed");

  // Verify data
  TEST_ASSERT(verify_transposed_pattern(transposed, {0, 2, 1}), "Data should be correct");

  TEST_SUCCESS("Vector field transpose (component dimensions)");
}

/**
 * Test 5: Time-varying array transpose
 */
void test_time_varying_transpose(MPI_Comm comm, int rank, int nprocs) {
  // Create time series: [100, 80, 50] (spatial 100x80, 50 timesteps)
  ndarray<double> timeseries;
  timeseries.decompose(comm, {100, 80, 50}, nprocs, {static_cast<size_t>(nprocs), 0, 0}, {0, 0, 0});
  timeseries.set_has_time(true);  // Last dim is time

  init_array_with_pattern(timeseries);

  // Transpose spatial dimensions only: {1, 0, 2}
  // Time dim stays last
  auto transposed = ftk::transpose(timeseries, {1, 0, 2});

  // Verify dimensions
  TEST_ASSERT(transposed.global_lattice().size(0) == 80, "New spatial dim 0 should be 80");
  TEST_ASSERT(transposed.global_lattice().size(1) == 100, "New spatial dim 1 should be 100");
  TEST_ASSERT(transposed.global_lattice().size(2) == 50, "Time dim should stay 50");

  // Verify time metadata preserved
  TEST_ASSERT(transposed.has_time(), "has_time should be preserved");

  // Verify decomposition
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[2] == 0, "Time dim should not be decomposed");

  // Verify data
  TEST_ASSERT(verify_transposed_pattern(transposed, {1, 0, 2}), "Data should be correct");

  TEST_SUCCESS("Time-varying array transpose");
}

/**
 * Test 6: Transpose with ghost layers
 */
void test_transpose_with_ghosts(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] Transpose with ghosts (requires at least 2 ranks)" << std::endl;
    return;
  }

  // Create array with ghost layers
  ndarray<double> arr;
  arr.decompose(comm, {100, 80}, nprocs, {static_cast<size_t>(nprocs), 0}, {1, 1});

  init_array_with_pattern(arr);

  // Exchange ghosts before transpose
  arr.exchange_ghosts();

  // Transpose
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify ghost widths are permuted
  auto new_ghosts = transposed.ghost_widths();
  TEST_ASSERT(new_ghosts[0] == 1, "New dim 0 ghost should be 1");
  TEST_ASSERT(new_ghosts[1] == 1, "New dim 1 ghost should be 1");

  // Verify data in core region
  TEST_ASSERT(verify_transposed_pattern(transposed, {1, 0}), "Core data should be correct");

  // Ghost exchange should still work after transpose
  transposed.exchange_ghosts();

  TEST_SUCCESS("Transpose with ghost layers");
}

/**
 * Test 7: Error case - trying to move component dimension
 */
void test_error_move_component_dim(MPI_Comm comm, int rank, int nprocs) {
  // Create vector field
  ndarray<double> velocity;
  velocity.decompose(comm, {3, 100, 80}, nprocs, {0, static_cast<size_t>(nprocs), 0}, {0, 0, 0});
  velocity.set_multicomponents(1);

  bool caught_error = false;
  try {
    // Try to move component dim to position 1 (INVALID)
    auto transposed = ftk::transpose(velocity, {1, 0, 2});
  } catch (const invalid_operation& e) {
    caught_error = true;
  }

  TEST_ASSERT(caught_error, "Should throw error when moving component dimension");

  TEST_SUCCESS("Error case: moving component dimension");
}

/**
 * Test 8: Error case - trying to move time dimension
 */
void test_error_move_time_dim(MPI_Comm comm, int rank, int nprocs) {
  // Create time-varying array
  ndarray<double> timeseries;
  timeseries.decompose(comm, {100, 80, 50}, nprocs, {static_cast<size_t>(nprocs), 0, 0}, {0, 0, 0});
  timeseries.set_has_time(true);

  bool caught_error = false;
  try {
    // Try to move time dim to position 0 (INVALID)
    auto transposed = ftk::transpose(timeseries, {2, 0, 1});
  } catch (const invalid_operation& e) {
    caught_error = true;
  }

  TEST_ASSERT(caught_error, "Should throw error when moving time dimension");

  TEST_SUCCESS("Error case: moving time dimension");
}

/**
 * Test 9: Identity transpose
 */
void test_identity_transpose(MPI_Comm comm, int rank, int nprocs) {
  ndarray<double> arr;
  arr.decompose(comm, {100, 80}, nprocs, {static_cast<size_t>(nprocs), 0}, {0, 0});

  init_array_with_pattern(arr);

  // Identity transpose
  auto transposed = ftk::transpose(arr, {0, 1});

  // Should produce identical distribution
  TEST_ASSERT(transposed.is_distributed(), "Result should be distributed");
  TEST_ASSERT(transposed.global_lattice().size(0) == 100, "Dim 0 unchanged");
  TEST_ASSERT(transposed.global_lattice().size(1) == 80, "Dim 1 unchanged");

  // Verify data unchanged
  TEST_ASSERT(verify_transposed_pattern(transposed, {0, 1}), "Data should be unchanged");

  TEST_SUCCESS("Identity transpose");
}

/**
 * Test 10: Multiple transposes in sequence
 */
void test_multiple_transposes(MPI_Comm comm, int rank, int nprocs) {
  ndarray<double> arr;
  arr.decompose(comm, {50, 60, 70}, nprocs, {static_cast<size_t>(nprocs), 0, 0}, {0, 0, 0});

  init_array_with_pattern(arr);

  // First transpose: {1, 0, 2}
  auto t1 = ftk::transpose(arr, {1, 0, 2});

  // Second transpose: {2, 1, 0}
  auto t2 = ftk::transpose(t1, {2, 1, 0});

  // Third transpose: back to original {1, 2, 0}
  auto t3 = ftk::transpose(t2, {1, 2, 0});

  // Should be back to original dimensions
  TEST_ASSERT(t3.global_lattice().size(0) == 50, "Back to original dim 0");
  TEST_ASSERT(t3.global_lattice().size(1) == 60, "Back to original dim 1");
  TEST_ASSERT(t3.global_lattice().size(2) == 70, "Back to original dim 2");

  TEST_SUCCESS("Multiple transposes in sequence");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Distributed Transpose Tests" << std::endl;
    std::cout << "Running with " << nprocs << " MPI ranks" << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  try {
    test_2d_transpose_1d_decomp(MPI_COMM_WORLD, rank, nprocs);
    test_2d_transpose_2d_decomp(MPI_COMM_WORLD, rank, nprocs);
    test_3d_transpose_2d_decomp(MPI_COMM_WORLD, rank, nprocs);
    test_vector_field_transpose(MPI_COMM_WORLD, rank, nprocs);
    test_time_varying_transpose(MPI_COMM_WORLD, rank, nprocs);
    test_transpose_with_ghosts(MPI_COMM_WORLD, rank, nprocs);
    test_error_move_component_dim(MPI_COMM_WORLD, rank, nprocs);
    test_error_move_time_dim(MPI_COMM_WORLD, rank, nprocs);
    test_identity_transpose(MPI_COMM_WORLD, rank, nprocs);
    test_multiple_transposes(MPI_COMM_WORLD, rank, nprocs);

    if (rank == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "All tests passed!" << std::endl;
      std::cout << "========================================\n" << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "[Rank " << rank << "] Exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
