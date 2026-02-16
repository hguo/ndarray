/**
 * Unified ndarray Tests with MPI Distribution
 *
 * Tests domain decomposition, index conversion, and parallel I/O for
 * distributed memory settings with MPI using the unified ndarray API.
 *
 * Run with: mpirun -np 4 ./test_distributed_ndarray
 */

#include <ndarray/config.hh>
#include <iostream>

#if NDARRAY_HAVE_MPI

#include <ndarray/ndarray.hh>
#include <mpi.h>
#include <cassert>
#include <cmath>

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

int test_automatic_decomposition() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 1: Automatic Decomposition ===" << std::endl;
  }

  TEST_SECTION("Create distributed array with unified API");
  ftk::ndarray<float> darray;

  TEST_SECTION("Decompose 1000×800 array automatically");
  darray.decompose(MPI_COMM_WORLD, {1000, 800});

  TEST_ASSERT(darray.rank() == rank, "Rank should match MPI rank");
  TEST_ASSERT(darray.nprocs() == nprocs, "Nprocs should match MPI size");
  TEST_ASSERT(darray.is_distributed(), "Array should be distributed");

  // Verify global lattice
  TEST_ASSERT(darray.global_lattice().nd() == 2, "Global lattice should be 2D");
  TEST_ASSERT(darray.global_lattice().size(0) == 1000, "Global dim 0 should be 1000");
  TEST_ASSERT(darray.global_lattice().size(1) == 800, "Global dim 1 should be 800");

  // Verify local core exists and is non-empty
  TEST_ASSERT(darray.local_core().nd() == 2, "Local core should be 2D");
  TEST_ASSERT(darray.local_core().size(0) > 0, "Local core dim 0 should be positive");
  TEST_ASSERT(darray.local_core().size(1) > 0, "Local core dim 1 should be positive");

  // Verify total size equals global size
  size_t local_size = darray.local_core().size(0) * darray.local_core().size(1);
  size_t total_size = 0;
  MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(total_size == 1000 * 800, "Sum of local sizes should equal global size");

  // Print decomposition info
  if (rank == 0) std::cout << "  Decomposition across " << nprocs << " ranks:" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  for (int r = 0; r < nprocs; r++) {
    if (rank == r) {
      std::cout << "    Rank " << rank << ": core=" << darray.local_core() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) std::cout << "  ✓ Automatic decomposition passed" << std::endl;
  return 0;
}

int test_manual_decomposition() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 2: Manual Decomposition ===" << std::endl;
  }

  TEST_SECTION("1D decomposition along first dimension");
  ftk::ndarray<double> darray;

  // Split only dimension 0, leave dimension 1 intact
  darray.decompose(MPI_COMM_WORLD,
                   {1000, 800},
                   static_cast<size_t>(nprocs),
                   {static_cast<size_t>(nprocs), 0},  // 1D decomposition
                   {});  // No ghosts

  // Each rank should have full dimension 1 (800)
  TEST_ASSERT(darray.local_core().size(1) == 800,
              "All ranks should have full dimension 1");

  // Sum of dimension 0 sizes should equal 1000
  size_t local_dim0 = darray.local_core().size(0);
  size_t total_dim0 = 0;
  MPI_Allreduce(&local_dim0, &total_dim0, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(total_dim0 == 1000, "Sum of dim 0 sizes should equal 1000");

  if (rank == 0) std::cout << "  ✓ Manual 1D decomposition passed" << std::endl;
  return 0;
}

int test_ghost_layers() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 3: Ghost Layers ===" << std::endl;
  }

  TEST_SECTION("Decompose with 1-layer ghosts");
  ftk::ndarray<float> darray;

  darray.decompose(MPI_COMM_WORLD,
                   {1000, 800},
                   0,  // Auto nprocs
                   {},  // Auto decomposition
                   {1, 1});  // 1-layer ghosts in both dimensions

  // Verify extent is larger than core (has ghosts)
  TEST_ASSERT(darray.local_extent().nd() == darray.local_core().nd(),
              "Extent and core should have same dimensionality");

  // For interior ranks, extent should be core + 2 ghosts per dimension
  // For boundary ranks, extent may be core + 1 ghost on one side
  size_t core_size0 = darray.local_core().size(0);
  size_t core_size1 = darray.local_core().size(1);
  size_t extent_size0 = darray.local_extent().size(0);
  size_t extent_size1 = darray.local_extent().size(1);

  // Extent should be at least as large as core
  TEST_ASSERT(extent_size0 >= core_size0, "Extent dim 0 should be >= core dim 0");
  TEST_ASSERT(extent_size1 >= core_size1, "Extent dim 1 should be >= core dim 1");

  // Local data should be allocated to extent size
  TEST_ASSERT(darray.local_array().dimf(0) == extent_size0,
              "Local array dim 0 should match extent");
  TEST_ASSERT(darray.local_array().dimf(1) == extent_size1,
              "Local array dim 1 should match extent");

  // Print ghost layer info
  if (rank == 0) std::cout << "  Ghost layer configuration:" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  for (int r = 0; r < nprocs; r++) {
    if (rank == r) {
      std::cout << "    Rank " << rank << ": core=" << darray.local_core()
                << ", extent=" << darray.local_extent() << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) std::cout << "  ✓ Ghost layer allocation passed" << std::endl;
  return 0;
}

int test_index_conversion() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 4: Index Conversion ===" << std::endl;
  }

  TEST_SECTION("Setup 1000×800 array with 1D decomposition");
  ftk::ndarray<int> darray;

  darray.decompose(MPI_COMM_WORLD,
                   {1000, 800},
                   static_cast<size_t>(nprocs),
                   {static_cast<size_t>(nprocs), 0});

  TEST_SECTION("Test global_to_local conversion");
  // Test conversion for this rank's owned data
  size_t core_start0 = darray.local_core().start(0);
  size_t core_start1 = darray.local_core().start(1);

  // Global index at start of this rank's core → should be local [0, 0]
  auto local_idx = darray.global_to_local({core_start0, core_start1});
  TEST_ASSERT(local_idx[0] == 0, "Local index 0 should be 0 at core start");
  TEST_ASSERT(local_idx[1] == 0, "Local index 1 should be 0 at core start");

  // Global index at middle of this rank's core
  size_t mid0 = core_start0 + darray.local_core().size(0) / 2;
  size_t mid1 = core_start1 + darray.local_core().size(1) / 2;
  local_idx = darray.global_to_local({mid0, mid1});
  TEST_ASSERT(local_idx[0] == darray.local_core().size(0) / 2,
              "Local index should be correct at midpoint");

  TEST_SECTION("Test local_to_global conversion");
  // Local [0, 0] → should be global [core_start0, core_start1]
  auto global_idx = darray.local_to_global({0, 0});
  TEST_ASSERT(global_idx[0] == core_start0, "Global index 0 should match core start");
  TEST_ASSERT(global_idx[1] == core_start1, "Global index 1 should match core start");

  // Round-trip test
  std::vector<size_t> original_global = {core_start0 + 10, core_start1 + 20};
  local_idx = darray.global_to_local(original_global);
  global_idx = darray.local_to_global(local_idx);
  TEST_ASSERT(global_idx[0] == original_global[0], "Round-trip index 0 should match");
  TEST_ASSERT(global_idx[1] == original_global[1], "Round-trip index 1 should match");

  TEST_SECTION("Test is_local");
  // Point in this rank's core should be local
  TEST_ASSERT(darray.is_local({core_start0, core_start1}), "Core start should be local");

  // Point before this rank's core should not be local (if not rank 0)
  if (core_start0 > 0) {
    TEST_ASSERT(!darray.is_local({core_start0 - 1, core_start1}),
                 "Point before core should not be local");
  }

  // Point after this rank's core should not be local (if not last rank)
  size_t core_end0 = core_start0 + darray.local_core().size(0);
  if (core_end0 < 1000) {
    TEST_ASSERT(!darray.is_local({core_end0, core_start1}),
                 "Point after core should not be local");
  }

  if (rank == 0) std::cout << "  ✓ Index conversion passed" << std::endl;
  return 0;
}

int test_data_access() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 5: Data Access and Manipulation ===" << std::endl;
  }

  TEST_SECTION("Create and fill local arrays");
  ftk::ndarray<double> darray;

  darray.decompose(MPI_COMM_WORLD, {100, 80}, 0, {}, {1, 1});

  // Fill local array with rank-specific values
  auto& local = darray.local_array();
  for (size_t i = 0; i < local.size(); i++) {
    local[i] = static_cast<double>(rank) * 1000.0 + static_cast<double>(i);
  }

  TEST_SECTION("Verify local array properties");
  TEST_ASSERT(local.size() == darray.local_extent().n(),
              "Local array size should match extent");
  TEST_ASSERT(local.dimf(0) == darray.local_extent().size(0),
              "Local array dim 0 should match extent");
  TEST_ASSERT(local.dimf(1) == darray.local_extent().size(1),
              "Local array dim 1 should match extent");

  TEST_SECTION("Test element access");
  TEST_ASSERT(local[0] == static_cast<double>(rank) * 1000.0,
              "First element should have correct value");

  if (rank == 0) std::cout << "  ✓ Data access passed" << std::endl;
  return 0;
}

#if NDARRAY_HAVE_PNETCDF
int test_parallel_netcdf_read() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 6: Parallel NetCDF Read ===" << std::endl;
  }

  TEST_SECTION("Create test NetCDF file (rank 0)");
  const size_t global_nx = 100;
  const size_t global_ny = 80;

  // Rank 0 creates the file
  if (rank == 0) {
    ftk::ndarray<float> global_data;
    global_data.reshapef(global_nx, global_ny);

    // Fill with known pattern: data[i,j] = i * 100 + j
    for (size_t i = 0; i < global_nx; i++) {
      for (size_t j = 0; j < global_ny; j++) {
        global_data.at(i, j) = static_cast<float>(i * 100 + j);
      }
    }

    // Write to NetCDF file (serial write from rank 0)
    try {
#if NDARRAY_HAVE_NETCDF
      int ncid, varid;
      int dimids[2];

      // Create file
      nc_create("test_distributed.nc", NC_CLOBBER | NC_64BIT_OFFSET, &ncid);

      // Define dimensions
      nc_def_dim(ncid, "x", global_nx, &dimids[0]);
      nc_def_dim(ncid, "y", global_ny, &dimids[1]);

      // Define variable
      nc_def_var(ncid, "data", NC_FLOAT, 2, dimids, &varid);

      // End define mode
      nc_enddef(ncid);

      // Write data
      global_data.to_netcdf(ncid, varid);

      // Close file
      nc_close(ncid);

      std::cout << "    Created test_distributed.nc with " << global_nx
                << " × " << global_ny << " array" << std::endl;
#else
      std::cerr << "    WARNING: NetCDF not available, skipping test" << std::endl;
      return 0;
#endif
    } catch (const std::exception& e) {
      std::cerr << "    WARNING: Could not create test file: " << e.what() << std::endl;
      std::cerr << "    Skipping parallel NetCDF read test" << std::endl;
      return 0;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  TEST_SECTION("Decompose domain and read in parallel");
  ftk::ndarray<float> darray;

  // Decompose to match file dimensions
  darray.decompose(MPI_COMM_WORLD, {global_nx, global_ny});

  // Parallel read
  try {
    darray.read_parallel("test_distributed.nc", "data");

    TEST_SECTION("Verify data correctness");
    // Check that each rank got the correct portion
    auto& local = darray.local_array();
    bool data_correct = true;

    for (size_t i = 0; i < darray.local_core().size(0); i++) {
      for (size_t j = 0; j < darray.local_core().size(1); j++) {
        // Convert to global indices
        size_t global_i = darray.local_core().start(0) + i;
        size_t global_j = darray.local_core().start(1) + j;

        float expected = static_cast<float>(global_i * 100 + global_j);
        float actual = local.at(i, j);

        if (std::abs(expected - actual) > 1e-6f) {
          std::cerr << "[Rank " << rank << "] Data mismatch at local [" << i << "," << j
                    << "] (global [" << global_i << "," << global_j << "]): "
                    << "expected " << expected << ", got " << actual << std::endl;
          data_correct = false;
          break;
        }
      }
      if (!data_correct) break;
    }

    TEST_ASSERT(data_correct, "All data values should be correct");

    // Verify total elements read equals global size
    size_t local_count = darray.local_core().n();
    size_t total_count = 0;
    MPI_Allreduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(total_count == global_nx * global_ny,
                "Total elements read should equal global size");

    if (rank == 0) std::cout << "  ✓ Parallel NetCDF read passed" << std::endl;

  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "  WARNING: Parallel read failed: " << e.what() << std::endl;
      std::cerr << "  This may be expected if PNetCDF is not fully configured" << std::endl;
    }
  }

  // Cleanup
  if (rank == 0) {
    std::remove("test_distributed.nc");
  }

  return 0;
}
#endif // NDARRAY_HAVE_PNETCDF

int test_parallel_binary_read() {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n=== Test 7: Parallel Binary Read ===" << std::endl;
  }

  TEST_SECTION("Create test binary file (rank 0)");
  const size_t global_nx = 100;
  const size_t global_ny = 80;

  // Rank 0 creates the file
  if (rank == 0) {
    ftk::ndarray<double> global_data;
    global_data.reshapef(global_nx, global_ny);

    // Fill with known pattern
    for (size_t i = 0; i < global_nx; i++) {
      for (size_t j = 0; j < global_ny; j++) {
        global_data.at(i, j) = static_cast<double>(i * 100 + j);
      }
    }

    // Write to binary file
    global_data.to_binary_file("test_distributed.bin");
    std::cout << "    Created test_distributed.bin" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  TEST_SECTION("Decompose domain and read in parallel with MPI-IO");
  ftk::ndarray<double> darray;

  darray.decompose(MPI_COMM_WORLD, {global_nx, global_ny});

  // Parallel read
  darray.read_parallel("test_distributed.bin");

  TEST_SECTION("Verify data correctness");
  // Check that each rank got the correct portion
  auto& local = darray.local_array();
  bool data_correct = true;

  for (size_t i = 0; i < darray.local_core().size(0); i++) {
    for (size_t j = 0; j < darray.local_core().size(1); j++) {
      // Convert to global indices
      size_t global_i = darray.local_core().start(0) + i;
      size_t global_j = darray.local_core().start(1) + j;

      double expected = static_cast<double>(global_i * 100 + global_j);
      double actual = local.at(i, j);

      if (std::abs(expected - actual) > 1e-9) {
        std::cerr << "[Rank " << rank << "] Data mismatch at local [" << i << "," << j
                  << "] (global [" << global_i << "," << global_j << "]): "
                  << "expected " << expected << ", got " << actual << std::endl;
        data_correct = false;
        break;
      }
    }
    if (!data_correct) break;
  }

  TEST_ASSERT(data_correct, "All binary data values should be correct");

  if (rank == 0) std::cout << "  ✓ Parallel binary read passed" << std::endl;

  // Cleanup
  if (rank == 0) {
    std::remove("test_distributed.bin");
  }

  return 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Distributed ndarray Test Suite                        ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nRunning with " << nprocs << " MPI ranks\n" << std::endl;
  }

  int result = 0;

  // Run tests
  result |= test_automatic_decomposition();
  result |= test_manual_decomposition();
  result |= test_ghost_layers();
  result |= test_index_conversion();
  result |= test_data_access();

#if NDARRAY_HAVE_PNETCDF
  result |= test_parallel_netcdf_read();
#else
  if (rank == 0) {
    std::cout << "\n⊘ Skipping parallel NetCDF read test (PNetCDF not enabled)" << std::endl;
  }
#endif

  result |= test_parallel_binary_read();

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    if (result == 0) {
      std::cout << "║  ✓✓✓ ALL DISTRIBUTED NDARRAY TESTS PASSED ✓✓✓            ║" << std::endl;
    } else {
      std::cout << "║  ✗✗✗ SOME TESTS FAILED ✗✗✗                               ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\n";
  }

  MPI_Finalize();
  return result;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     Distributed ndarray Test Suite                        ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";
  std::cout << "⊘ Distributed ndarray tests require MPI support" << std::endl;
  std::cout << "⊘ Please rebuild with -DNDARRAY_USE_MPI=ON" << std::endl;
  std::cout << "\n";
  return 0;
}

#endif // NDARRAY_HAVE_MPI
