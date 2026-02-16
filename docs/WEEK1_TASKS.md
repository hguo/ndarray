# Week 1 Tasks: Testing Infrastructure Setup

**Goal**: Establish baseline correctness for distributed features through comprehensive local testing

**Time**: 5 days
**Prerequisites**: MPI installed (OpenMPI or MPICH)

---

## Day 1: Comprehensive Distributed Array Tests

### Task 1.1: Extend test_distributed_ndarray.cpp

**Current state**: 543 lines, basic functionality
**Target state**: 1000+ lines, comprehensive coverage

**Add these test cases**:

```cpp
// Test 1: Ghost exchange correctness (reference comparison)
TEST_CASE("Ghost exchange produces correct values") {
  // Create distributed array
  // Fill with known pattern (e.g., rank + index)
  // Exchange ghosts
  // Verify ghost values match neighbor core values
  // Compare with serial reference implementation
}

// Test 2: Various decomposition patterns
TEST_CASE("1D decomposition") { /* 4 ranks, split only first dim */ }
TEST_CASE("2D decomposition") { /* 4 ranks, split two dims */ }
TEST_CASE("Odd decompositions") { /* 3 ranks, 5 ranks, 7 ranks */ }
TEST_CASE("Non-square arrays") { /* 1000x500, 100x2000, etc. */ }

// Test 3: Array size stress tests
TEST_CASE("Small arrays (10x10)") { /* Edge case */ }
TEST_CASE("Medium arrays (1000x800)") { /* Typical */ }
TEST_CASE("Large arrays (10000x10000)") { /* If memory allows */ }

// Test 4: Ghost layer widths
TEST_CASE("1-layer ghosts") { /* Typical */ }
TEST_CASE("2-layer ghosts") { /* 5-point stencil */ }
TEST_CASE("3-layer ghosts") { /* 9-point stencil */ }

// Test 5: Multicomponent arrays
TEST_CASE("Velocity field [1000,800,3] with decomp [4,2,0]") {
  // Should NOT split vector components (dimension 2)
  // Verify each rank has full 3-component vectors
}

// Test 6: Replicated vs distributed
TEST_CASE("Replicated array") { /* All ranks have full data */ }
TEST_CASE("Mixed: some distributed, some replicated") { }

// Test 7: Edge cases
TEST_CASE("Single rank (no MPI communication)") { }
TEST_CASE("Zero ghost layers") { /* No communication needed */ }
TEST_CASE("Array smaller than number of ranks") { /* Some ranks idle */ }

// Test 8: Error handling
TEST_CASE("Accessing global index not owned") { /* Should throw */ }
TEST_CASE("Invalid decomposition") { /* Should catch */ }
```

**Commands to run**:
```bash
cd build
mpirun -np 2 ./bin/test_distributed_ndarray
mpirun -np 4 ./bin/test_distributed_ndarray
mpirun -np 8 ./bin/test_distributed_ndarray
```

**Success criteria**:
- All tests pass with 2, 4, 8 ranks
- Ghost values match expected (verify against serial reference)
- No deadlocks or hangs

**Deliverable**: Extended test_distributed_ndarray.cpp committed

---

## Day 2: Ghost Exchange Verification

### Task 2.1: Create reference implementation

**Create reference_ghost_exchange.cpp**:
```cpp
// Serial reference implementation for verification
template <typename T>
void reference_ghost_exchange_2d(
    std::vector<T>& data,
    size_t width, size_t height,
    size_t ghost_width,
    const std::vector<lattice>& subdomains)
{
  // For each subdomain:
  //   For each ghost cell:
  //     Find which subdomain owns that data
  //     Copy from owner's core to this subdomain's ghost

  // This is slow but obviously correct
  // Use for validation only, not production
}
```

**Use in tests**:
```cpp
// In test: compare distributed ghost exchange vs reference
ftk::ndarray<float> ref_data = /* serial reference */;
reference_ghost_exchange_2d(ref_data, ...);

ftk::ndarray<float> dist_data = /* distributed */;
dist_data.exchange_ghosts();

// Gather to rank 0, compare
REQUIRE(arrays_match(ref_data, gathered_dist_data));
```

**Success criteria**:
- Reference implementation obviously correct (can verify by inspection)
- Distributed ghost exchange matches reference
- Provides confidence in correctness

**Deliverable**: tests/reference_ghost_exchange.cpp

---

## Day 3: Memory Leak Detection

### Task 3.1: Run valgrind on all tests

**Commands**:
```bash
cd build

# Single rank (easier to debug)
valgrind --leak-check=full --show-leak-kinds=all \
  --track-origins=yes --verbose \
  ./bin/test_distributed_ndarray

# Multi-rank
mpirun -np 4 valgrind --leak-check=full --show-leak-kinds=all \
  --log-file=valgrind-%p.log \
  ./bin/test_distributed_ndarray

# Check logs
grep "definitely lost" valgrind-*.log
grep "indirectly lost" valgrind-*.log
grep "possibly lost" valgrind-*.log
```

**Common leaks to look for**:
- MPI buffer allocations not freed
- Device memory not freed (if GPU code)
- Ghost exchange temporary buffers
- Lattice/partitioner allocations

**Fix any leaks found**:
```cpp
// Before:
float* buffer = new float[size];
MPI_Send(buffer, size, ...);
// Missing: delete[] buffer;

// After:
std::vector<float> buffer(size);
MPI_Send(buffer.data(), size, ...);
// Automatic cleanup
```

**Success criteria**:
- Zero "definitely lost" bytes
- Zero "indirectly lost" bytes
- "still reachable" OK (MPI runtime)
- No invalid reads/writes

**Deliverable**: Valgrind-clean codebase, commit fixes

---

## Day 4: Parallel I/O Testing Setup

### Task 4.1: Create test_parallel_io.cpp

**Test structure**:
```cpp
#include <ndarray/ndarray.hh>
#include <catch.hpp>
#include <mpi.h>

TEST_CASE("NetCDF parallel write-read round-trip") {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create distributed array with known values
  ftk::ndarray<float> data;
  data.decompose(MPI_COMM_WORLD, {1000, 800});

  // Fill with pattern: value = rank*1000000 + i*1000 + j
  for (size_t i = 0; i < data.local_core().size(0); i++) {
    for (size_t j = 0; j < data.local_core().size(1); j++) {
      auto global_idx = data.local_to_global({i, j});
      data.at(i, j) = rank * 1000000 + global_idx[0] * 1000 + global_idx[1];
    }
  }

  // Write in parallel
  data.write_netcdf_auto("test_parallel.nc", "data");

  // Read back with DIFFERENT decomposition
  ftk::ndarray<float> data2;
  if (nprocs >= 2) {
    // Use different decomposition pattern
    data2.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {1, nprocs});
  } else {
    data2.decompose(MPI_COMM_WORLD, {1000, 800});
  }
  data2.read_netcdf_auto("test_parallel.nc", "data");

  // Verify values match
  for (size_t i = 0; i < data2.local_core().size(0); i++) {
    for (size_t j = 0; j < data2.local_core().size(1); j++) {
      auto global_idx = data2.local_to_global({i, j});
      float expected = /* calculate from global_idx */;
      REQUIRE(data2.at(i, j) == expected);
    }
  }
}

TEST_CASE("HDF5 parallel write-read") { /* Similar */ }
TEST_CASE("Binary MPI-IO write-read") { /* Similar */ }
TEST_CASE("Large file handling (1GB+)") { /* If disk space */ }
TEST_CASE("Error handling: permission denied") { /* Mock or actual */ }
```

**Success criteria**:
- Write with one decomposition, read with different decomposition
- All values match expected
- Works with NetCDF parallel, HDF5 parallel, MPI-IO binary
- Error handling doesn't deadlock

**Deliverable**: tests/test_parallel_io.cpp

---

## Day 5: Performance Measurement Infrastructure

### Task 5.1: Create benchmark_distributed.cpp

**Simple benchmarking harness**:
```cpp
#include <ndarray/ndarray.hh>
#include <chrono>
#include <fstream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Test parameters
  std::vector<size_t> sizes = {100, 500, 1000, 5000, 10000};
  int num_iterations = 10;

  if (rank == 0) {
    std::cout << "array_size,num_ranks,operation,time_ms" << std::endl;
  }

  for (auto size : sizes) {
    // Create distributed array
    ftk::ndarray<float> data;
    data.decompose(MPI_COMM_WORLD, {size, size}, 0, {}, {1, 1});
    data.fill(rank);

    // Benchmark ghost exchange
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; iter++) {
      data.exchange_ghosts();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / num_iterations;

    // Gather timing from all ranks
    double max_time;
    MPI_Reduce(&avg_ms, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << size << "," << nprocs << ",ghost_exchange," << max_time << std::endl;
    }

    // Benchmark parallel I/O
    std::string filename = "bench_" + std::to_string(size) + ".nc";

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::high_resolution_clock::now();
    data.write_netcdf_auto(filename, "data");
    end = std::chrono::high_resolution_clock::now();

    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    MPI_Reduce(&elapsed_ms, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      size_t bytes = size * size * sizeof(float);
      double mb = bytes / (1024.0 * 1024.0);
      double bandwidth = mb / (max_time / 1000.0);
      std::cout << size << "," << nprocs << ",parallel_write," << max_time << std::endl;
      std::cout << "# Bandwidth: " << bandwidth << " MB/s" << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
```

**Run benchmarks**:
```bash
cd build
mpirun -np 2 ./bin/benchmark_distributed > results_2ranks.csv
mpirun -np 4 ./bin/benchmark_distributed > results_4ranks.csv
mpirun -np 8 ./bin/benchmark_distributed > results_8ranks.csv
```

**Deliverable**:
- benchmark/benchmark_distributed.cpp
- Initial benchmark results (CSV files)

---

## Week 1 Summary

**By end of Week 1, you will have**:

✅ Comprehensive test suite (1000+ lines)
- Ghost exchange correctness verified
- Various decompositions tested
- Edge cases covered
- Reference implementation for validation

✅ Memory-clean codebase
- All tests pass under valgrind
- No memory leaks

✅ Parallel I/O test infrastructure
- Write-read round-trips working
- Different formats tested
- Error handling verified

✅ Performance measurement baseline
- Benchmark infrastructure ready
- Initial local measurements taken
- CSV data for analysis

**Success metric**: All tests pass with 2, 4, 8 ranks locally, zero memory leaks

**Next steps**: Week 2 - GPU feature validation (if GPU available) or more extensive I/O testing

---

## Quick Start Commands

```bash
# Day 1-2: Build and test
cd /Users/guo.2154/workspace/projects/ndarray/build
cmake .. -DNDARRAY_BUILD_TESTS=ON -DNDARRAY_USE_MPI=ON
make -j8

# Run distributed tests
mpirun -np 4 ./bin/test_distributed_ndarray

# Day 3: Check for leaks
mpirun -np 4 valgrind --leak-check=full --log-file=valgrind-%p.log ./bin/test_distributed_ndarray
grep "definitely lost" valgrind-*.log

# Day 4-5: Create new tests
# Edit tests/test_parallel_io.cpp
# Edit benchmark/benchmark_distributed.cpp
make -j8
mpirun -np 4 ./bin/test_parallel_io
mpirun -np 4 ./bin/benchmark_distributed > results.csv
```

---

**Status**: Ready to start
**Estimated time**: 5 days of focused work
**Output**: Validated local multi-process functionality
