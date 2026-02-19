/**
 * HDF5 Distribution-Aware I/O Tests
 *
 * Tests the unified HDF5 I/O API:
 * - write_hdf5_auto()
 * - read_hdf5_auto()
 * - Distributed and replicated modes
 *
 * Run with: mpirun -np 4 ./test_hdf5_auto
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <mpi.h>

#if NDARRAY_HAVE_HDF5
#include <hdf5.h>
#endif

#define TEST_ASSERT(condition, message) 
  do { 
    if (!(condition)) { 
      std::cerr << "[Rank " << rank << "] FAILED: " << message << std::endl; 
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; 
      MPI_Abort(MPI_COMM_WORLD, 1); 
    } 
  } while (0)

#define TEST_SECTION(name) 
  if (rank == 0) std::cout << "
--- Testing: " << name << " ---" << std::endl

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_HDF5
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Running ndarray HDF5 Auto I/O Tests ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI processes" << std::endl;
  }

  const std::string filename = "test_auto_hdf5.h5";
  const size_t nx = 40, ny = 30;

  // Test 1: Distributed Write
  {
    TEST_SECTION("Distributed Write");

    ftk::ndarray<float> darray;
    darray.decompose(MPI_COMM_WORLD, {nx, ny});
    
    const auto& core = darray.local_core();
    const auto& extent = darray.local_extent();
    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = core.start(1) - extent.start(1);

    for (size_t i = 0; i < core.size(0); i++) {
      for (size_t j = 0; j < core.size(1); j++) {
        size_t gi = core.start(0) + i;
        size_t gj = core.start(1) + j;
        darray.f(off_i + i, off_j + j) = static_cast<float>(gi * 100 + gj);
      }
    }

    if (rank == 0) std::cout << "    - Writing distributed array to " << filename << std::endl;
    darray.write_hdf5_auto(filename, "test_var");
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Distributed Read
  {
    TEST_SECTION("Distributed Read");

    ftk::ndarray<float> darray;
    darray.decompose(MPI_COMM_WORLD, {nx, ny});
    
    if (rank == 0) std::cout << "    - Reading distributed array from " << filename << std::endl;
    darray.read_hdf5_auto(filename, "test_var");

    const auto& core = darray.local_core();
    const auto& extent = darray.local_extent();
    size_t off_i = core.start(0) - extent.start(0);
    size_t off_j = core.start(1) - extent.start(1);

    bool correct = true;
    for (size_t i = 0; i < core.size(0); i++) {
      for (size_t j = 0; j < core.size(1); j++) {
        size_t gi = core.start(0) + i;
        size_t gj = core.start(1) + j;
        float expected = static_cast<float>(gi * 100 + gj);
        if (std::abs(darray.f(off_i + i, off_j + j) - expected) > 1e-5f) {
          correct = false;
          std::cerr << "[Rank " << rank << "] Value mismatch at (" << gi << "," << gj 
                    << "): expected " << expected << ", got " << darray.f(off_i + i, off_j + j) << std::endl;
        }
      }
    }

    TEST_ASSERT(correct, "Data verification failed in distributed read");
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Replicated Read
  {
    TEST_SECTION("Replicated Read");

    ftk::ndarray<float> rarray;
    rarray.set_replicated(MPI_COMM_WORLD);
    rarray.reshapef(nx, ny);
    
    if (rank == 0) std::cout << "    - Reading replicated array (all ranks get full data)" << std::endl;
    rarray.read_hdf5_auto(filename, "test_var");

    bool correct = true;
    for (size_t gi = 0; gi < nx; gi++) {
      for (size_t gj = 0; gj < ny; gj++) {
        float expected = static_cast<float>(gi * 100 + gj);
        if (std::abs(rarray.f(gi, gj) - expected) > 1e-5f) {
          correct = false;
        }
      }
    }

    TEST_ASSERT(correct, "Data verification failed in replicated read");
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "    PASSED" << std::endl;
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::remove(filename.c_str());
    std::cout << "
=== All HDF5 Auto I/O Tests Passed ===
" << std::endl;
  }

  MPI_Finalize();
  return 0;

#else
  std::cout << "HDF5 Auto I/O tests require MPI and HDF5 support" << std::endl;
  return 0;
#endif
}
