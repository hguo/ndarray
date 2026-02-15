/**
 * Distributed Memory Parallel I/O Example
 *
 * Demonstrates reading large time-series data in parallel across MPI ranks.
 * Each rank reads its local portion, performs computation, and can optionally
 * write results.
 *
 * Typical vis/analysis workflow:
 * 1. Decompose global domain across ranks
 * 2. Each rank reads its local portion in parallel
 * 3. Exchange ghost cells if needed for stencil operations
 * 4. Perform local computation/analysis
 * 5. Optionally gather results or write in parallel
 *
 * Compile with MPI and run: mpirun -np 4 ./distributed_io
 */

#include <ndarray/config.hh>
#include <iostream>

#if NDARRAY_HAVE_MPI

#include <ndarray/distributed_ndarray.hh>
#include <mpi.h>
#include <cmath>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Distributed Memory Parallel I/O Example ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI ranks" << std::endl << std::endl;
  }

  // ============================================================================
  // Step 1: Create test data (normally this would be your large simulation data)
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 1: Creating synthetic test data..." << std::endl;

    // Create a synthetic 1000×800 array
    ftk::ndarray<float> test_data;
    test_data.reshapef(1000, 800);

    // Fill with synthetic temperature field: T(x,y) = sin(x/50) * cos(y/40)
    for (size_t i = 0; i < 1000; i++) {
      for (size_t j = 0; j < 800; j++) {
        float x = static_cast<float>(i);
        float y = static_cast<float>(j);
        test_data.at(i, j) = std::sin(x / 50.0f) * std::cos(y / 40.0f) * 100.0f + 273.15f;
      }
    }

    // Write to binary file for demonstration
    test_data.to_binary_file("temperature_field.bin");
    std::cout << "  Created temperature_field.bin (1000 × 800)" << std::endl << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ============================================================================
  // Step 2: Decompose domain across MPI ranks
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 2: Decomposing domain across " << nprocs << " ranks..." << std::endl;
  }

  ftk::distributed_ndarray<float> temperature(MPI_COMM_WORLD);

  // Automatic decomposition with 1-layer ghost cells
  // Library will balance load across ranks based on prime factorization
  temperature.decompose({1000, 800}, 0, {}, {1, 1});

  std::cout << "  Rank " << rank << " owns: " << temperature.local_core() << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // ============================================================================
  // Step 3: Parallel read - each rank reads its portion
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 3: Reading data in parallel (MPI-IO)..." << std::endl;
  }

  auto start_time = MPI_Wtime();

  // Each rank reads its local portion automatically
  temperature.read_parallel("temperature_field.bin");

  auto read_time = MPI_Wtime() - start_time;

  // Report timing
  double max_read_time;
  MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "  Parallel read completed in " << max_read_time << " seconds" << std::endl;
  }

  // Access local data
  auto& local_temp = temperature.local_array();
  size_t local_size = temperature.local_core().n();

  std::cout << "  Rank " << rank << " read " << local_size << " elements" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // ============================================================================
  // Step 4: Perform local analysis
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 4: Performing local analysis..." << std::endl;
  }

  // Example: Compute statistics on local data
  float local_min = std::numeric_limits<float>::max();
  float local_max = std::numeric_limits<float>::lowest();
  double local_sum = 0.0;
  size_t local_count = 0;

  for (size_t i = 0; i < temperature.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature.local_core().size(1); j++) {
      float value = local_temp.at(i, j);
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
      local_sum += value;
      local_count++;
    }
  }

  double local_mean = local_sum / local_count;

  // Global reduction to get overall statistics
  float global_min, global_max;
  double global_sum;
  size_t global_count;

  MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double global_mean = global_sum / global_count;
    std::cout << "  Global statistics:" << std::endl;
    std::cout << "    Min temperature: " << global_min << " K" << std::endl;
    std::cout << "    Max temperature: " << global_max << " K" << std::endl;
    std::cout << "    Mean temperature: " << global_mean << " K" << std::endl;
    std::cout << "    Total elements: " << global_count << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Local statistics per rank
  std::cout << "  Rank " << rank << " local stats: "
            << "min=" << local_min << ", max=" << local_max
            << ", mean=" << local_mean << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // ============================================================================
  // Step 5: Example - find regions above threshold
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 5: Finding hot regions (T > 300K)..." << std::endl;
  }

  size_t local_hot_count = 0;
  for (size_t i = 0; i < temperature.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature.local_core().size(1); j++) {
      if (local_temp.at(i, j) > 300.0f) {
        local_hot_count++;
      }
    }
  }

  size_t global_hot_count;
  MPI_Reduce(&local_hot_count, &global_hot_count, 1, MPI_UNSIGNED_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double hot_percentage = (100.0 * global_hot_count) / global_count;
    std::cout << "  Hot regions: " << global_hot_count << " cells ("
              << hot_percentage << "%)" << std::endl << std::endl;
  }

  // ============================================================================
  // Step 6: Convert global indices to local (example)
  // ============================================================================

  if (rank == 0) {
    std::cout << "Step 6: Example of global/local index conversion..." << std::endl;
  }

  // Example: Check if point [500, 400] is on this rank
  std::vector<size_t> global_point = {500, 400};

  if (temperature.is_local(global_point)) {
    auto local_point = temperature.global_to_local(global_point);
    float value = local_temp.at(local_point[0], local_point[1]);

    std::cout << "  Rank " << rank << ": Global point [500, 400] is local ["
              << local_point[0] << ", " << local_point[1]
              << "] with temperature " << value << " K" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // ============================================================================
  // Cleanup
  // ============================================================================

  if (rank == 0) {
    std::cout << "=== Example completed successfully ===" << std::endl;
    std::cout << "\nKey takeaways:" << std::endl;
    std::cout << "  ✓ Each rank reads only its local portion (scalable I/O)" << std::endl;
    std::cout << "  ✓ Automatic domain decomposition balances load" << std::endl;
    std::cout << "  ✓ Local computation with global reductions for statistics" << std::endl;
    std::cout << "  ✓ Flexible index conversion (global ↔ local)" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "  - Use exchange_ghosts() for stencil operations (Phase 3)" << std::endl;
    std::cout << "  - Read from NetCDF/HDF5 with .read_parallel()" << std::endl;
    std::cout << "  - Process multiple timesteps in a loop" << std::endl;

    // Cleanup test file
    std::remove("temperature_field.bin");
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "=== Distributed Memory Parallel I/O Example ===" << std::endl;
  std::cout << "ERROR: MPI support not enabled!" << std::endl;
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON" << std::endl;
  std::cout << "and run with: mpirun -np 4 ./distributed_io" << std::endl;
  return 1;
}

#endif // NDARRAY_HAVE_MPI
