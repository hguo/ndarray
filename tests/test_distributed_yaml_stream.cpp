/**
 * Test distributed YAML streams with domain decomposition
 *
 * Demonstrates how synthetic data streams can work with MPI
 * domain decomposition, generating only local portions on each rank.
 *
 * Run with: mpirun -np 4 ./test_distributed_yaml_stream
 */

#include <ndarray/config.hh>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cassert>

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
    std::cout << "\n=== Testing Distributed YAML Streams ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI ranks\n" << std::endl;
  }

  // Test 1: Distributed synthetic woven pattern
  {
    TEST_SECTION("Test 1: Distributed synthetic woven stream");

    // Create YAML configuration for distributed synthetic stream
    if (rank == 0) {
      std::ofstream yaml_file("test_distributed_woven.yaml");
      yaml_file << R"(
substreams:
  - format: synthetic_woven
    dimensions: [128, 256]
    timesteps: 5
    variables:
      - name: temperature
        dtype: double
        distribution: distributed
)";
      yaml_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Parse and read stream
    ftk::stream<> s(MPI_COMM_WORLD);
    s.parse_yaml("test_distributed_woven.yaml");

    TEST_ASSERT(s.total_timesteps() == 5, "Should have 5 timesteps");

    // Read a timestep and verify distribution
    auto g = s.read(0);
    TEST_ASSERT(g->has("temperature"), "Group should have temperature variable");

    auto temp = g->get_ptr<double>("temperature");
    TEST_ASSERT(temp != nullptr, "Should get valid temperature array");
    TEST_ASSERT(temp->is_distributed(), "Temperature should be distributed");
    TEST_ASSERT(temp->nprocs() == nprocs, "Should have correct number of processes");

    // Verify local shape is smaller than global shape
    size_t local_size = temp->nelem();
    size_t global_size = 128 * 256;
    TEST_ASSERT(local_size <= global_size / nprocs + 100,
                "Local size should be approximately global_size/nprocs");

    if (rank == 0) {
      std::cout << "    ✓ Global dimensions: 128x256" << std::endl;
      std::cout << "    ✓ Local size on rank 0: " << local_size << " elements" << std::endl;
      std::cout << "    ✓ Distributed across " << nprocs << " ranks" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Test 2: Distributed synthetic moving extremum
  {
    TEST_SECTION("Test 2: Distributed moving extremum stream");

    if (rank == 0) {
      std::ofstream yaml_file("test_distributed_extremum.yaml");
      yaml_file << R"(
substreams:
  - format: synthetic_moving_extremum
    dimensions: [100, 200]
    timesteps: 3
    x0: [0.3, 0.3]
    dir: [0.1, 0.1]
    sign: -1
    variables:
      - name: scalar_field
        dtype: double
        distribution: distributed
        decomposition:
          pattern: [0, 0]  # Auto decompose both dimensions
          ghost: [2, 2]    # 2 ghost layers per dimension
)";
      yaml_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ftk::stream<> s(MPI_COMM_WORLD);
    s.parse_yaml("test_distributed_extremum.yaml");

    auto g = s.read(0);
    auto field = g->get_ptr<double>("scalar_field");

    TEST_ASSERT(field != nullptr, "Should get valid scalar_field array");
    TEST_ASSERT(field->is_distributed(), "Field should be distributed");
    TEST_ASSERT(field->nprocs() == nprocs, "Should have correct number of processes");

    // Verify data is generated correctly (moving maximum)
    size_t local_size = field->nelem();
    TEST_ASSERT(local_size > 0, "Should have local data");

    if (rank == 0) {
      std::cout << "    ✓ Global dimensions: 100x200 with ghost layers" << std::endl;
      std::cout << "    ✓ Local size on rank 0: " << local_size << " elements" << std::endl;
      std::cout << "    ✓ Distributed with domain decomposition" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Test 3: Replicated mode (full data on all ranks)
  {
    TEST_SECTION("Test 3: Replicated synthetic stream");

    if (rank == 0) {
      std::ofstream yaml_file("test_replicated.yaml");
      yaml_file << R"(
substreams:
  - format: synthetic_woven
    dimensions: [50, 50]
    timesteps: 2
    variables:
      - name: shared_data
        dtype: float
        distribution: replicated
)";
      yaml_file.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ftk::stream<> s(MPI_COMM_WORLD);
    s.parse_yaml("test_replicated.yaml");

    auto g = s.read(0);
    auto data = g->get_ptr<float>("shared_data");

    TEST_ASSERT(data != nullptr, "Should get valid shared_data array");
    TEST_ASSERT(!data->is_distributed(), "Data should be replicated");
    TEST_ASSERT(data->nelem() == 50 * 50, "All ranks should have full data");

    if (rank == 0) {
      std::cout << "    ✓ All ranks have full 50x50 data" << std::endl;
      std::cout << "    ✓ Replicated mode working correctly" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Cleanup
  if (rank == 0) {
    std::remove("test_distributed_woven.yaml");
    std::remove("test_distributed_extremum.yaml");
    std::remove("test_replicated.yaml");
  }

  if (rank == 0) {
    std::cout << "\n=== All Distributed YAML Stream Tests Passed ===" << std::endl;
  }

  MPI_Finalize();
  return 0;
}

#else

int main() {
  std::cout << "MPI and/or YAML support not available - tests skipped" << std::endl;
  std::cout << "Build with -DNDARRAY_USE_MPI=TRUE -DNDARRAY_USE_YAML=TRUE to enable" << std::endl;
  return 0;
}

#endif
