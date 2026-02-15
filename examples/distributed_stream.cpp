/**
 * Distributed Stream Example
 *
 * Demonstrates using distributed_stream for easy time-series processing.
 * The stream interface simplifies iteration over timesteps and automatic
 * parallel reading.
 *
 * Workflow:
 * 1. Configure stream (decomposition, variables)
 * 2. Set input source (file pattern or YAML)
 * 3. Iterate through timesteps with simple API
 *
 * Compile and run: mpirun -np 4 ./distributed_stream
 */

#include <ndarray/config.hh>
#include <iostream>
#include <cmath>

#if NDARRAY_HAVE_MPI

#include <ndarray/distributed_ndarray_stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Distributed Stream Example ===\n";
    std::cout << "Running with " << nprocs << " MPI ranks\n" << std::endl;
  }

  // ===========================================================================
  // Step 1: Create synthetic time-series data
  // ===========================================================================

  const size_t NX = 300, NY = 200;
  const int NUM_TIMESTEPS = 5;

  if (rank == 0) {
    std::cout << "Step 1: Creating synthetic temperature time-series...\n";

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      ftk::ndarray<float> temperature;
      temperature.reshapef(NX, NY);

      // Temperature evolves over time: T(x,y,t) = sin(x/30 + t) * cos(y/20)
      for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
          float x = static_cast<float>(i);
          float y = static_cast<float>(j);
          float temp = std::sin(x / 30.0f + t * 0.5f) * std::cos(y / 20.0f);
          temperature.at(i, j) = temp * 50.0f + 273.15f;  // Scale to ~273K
        }
      }

      // Write to binary file
      temperature.to_binary_file("temperature_t" + std::to_string(t) + ".bin");
    }

    std::cout << "  Created " << NUM_TIMESTEPS << " timesteps\n" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ===========================================================================
  // Step 2: Configure distributed stream
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 2: Configuring distributed stream...\n";
  }

  // Create stream
  ftk::distributed_stream<float> stream(MPI_COMM_WORLD);

  // Set domain decomposition (automatic, with 1-layer ghosts)
  stream.set_decomposition({NX, NY}, 0, {}, {1, 1});

  // Add variables to read
  stream.add_variable("temperature");  // Note: not used for binary, but kept for consistency

  // Set input source (file pattern with {timestep} placeholder)
  stream.set_input_source("temperature_t{timestep}.bin");
  stream.set_n_timesteps(NUM_TIMESTEPS);

  if (rank == 0) {
    std::cout << "  Stream configured for " << stream.n_timesteps()
              << " timesteps\n" << std::endl;
  }

  // ===========================================================================
  // Step 3: Process time-series with stream API
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 3: Processing time-series...\n";
  }

  // Method 1: Traditional loop
  std::cout << "  Rank " << rank << " processing..." << std::endl;

  std::vector<float> mean_temperature_over_time;

  for (int t = 0; t < stream.n_timesteps(); t++) {
    // Read all variables at this timestep (returns map)
    auto vars = stream.read(t);

    // Access specific variable
    auto& temperature = vars["temperature"];

    // Exchange ghosts if needed for stencil operations
    temperature.exchange_ghosts();

    // Compute local statistics
    double local_sum = 0.0;
    size_t local_count = temperature.local_core().n();

    auto& local = temperature.local_array();
    for (size_t i = 0; i < temperature.local_core().size(0); i++) {
      for (size_t j = 0; j < temperature.local_core().size(1); j++) {
        local_sum += local.at(i, j);
      }
    }

    // Global mean
    double global_sum;
    size_t global_count;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      float global_mean = global_sum / global_count;
      mean_temperature_over_time.push_back(global_mean);
      std::cout << "  t=" << t << ": mean temperature = "
                << global_mean << " K" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Step 4: Alternative - functional style with for_each_timestep
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 4: Alternative - functional style...\n";
  }

  // Reset stream for demonstration
  std::vector<float> max_gradient_over_time;

  stream.for_each_timestep([&](int t, auto& vars) {
    auto& temperature = vars["temperature"];
    temperature.exchange_ghosts();

    // Compute gradient magnitude
    auto& local = temperature.local_array();
    float local_max_grad = 0.0f;

    for (size_t i = 1; i < local.dim(0) - 1; i++) {
      for (size_t j = 1; j < local.dim(1) - 1; j++) {
        float dx = (local.at(i+1, j) - local.at(i-1, j)) / 2.0f;
        float dy = (local.at(i, j+1) - local.at(i, j-1)) / 2.0f;
        float grad = std::sqrt(dx*dx + dy*dy);
        local_max_grad = std::max(local_max_grad, grad);
      }
    }

    // Global max
    float global_max_grad;
    MPI_Reduce(&local_max_grad, &global_max_grad, 1, MPI_FLOAT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      max_gradient_over_time.push_back(global_max_grad);
      std::cout << "  t=" << t << ": max gradient = "
                << global_max_grad << std::endl;
    }
  });

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Step 5: Read single variable at specific timestep
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 5: Reading single variable at specific timestep...\n";
  }

  // Read just temperature at timestep 2
  auto temperature_t2 = stream.read_var("temperature", 2);
  temperature_t2.exchange_ghosts();

  // Compute some property
  auto& local = temperature_t2.local_array();
  float local_max = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < temperature_t2.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature_t2.local_core().size(1); j++) {
      local_max = std::max(local_max, local.at(i, j));
    }
  }

  float global_max;
  MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "  Max temperature at t=2: " << global_max << " K\n";
    std::cout << std::endl;
  }

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  if (rank == 0) {
    std::cout << "=== Stream Example Complete ===\n\n";

    std::cout << "Key features demonstrated:\n";
    std::cout << "  ✓ Simple stream configuration\n";
    std::cout << "  ✓ Automatic parallel reading for each timestep\n";
    std::cout << "  ✓ Traditional loop interface: for (t...) stream.read(t)\n";
    std::cout << "  ✓ Functional interface: stream.for_each_timestep(lambda)\n";
    std::cout << "  ✓ Single variable reading: stream.read_var(name, t)\n\n";

    std::cout << "Stream benefits over manual reading:\n";
    std::cout << "  - Less boilerplate code\n";
    std::cout << "  - Consistent interface across formats\n";
    std::cout << "  - Easy to switch between file patterns\n";
    std::cout << "  - Automatic decomposition handling\n\n";

    std::cout << "Temporal analysis results:\n";
    std::cout << "  Mean temperature evolution: ";
    for (float val : mean_temperature_over_time) {
      std::cout << val << " ";
    }
    std::cout << "\n";

    std::cout << "  Max gradient evolution: ";
    for (float val : max_gradient_over_time) {
      std::cout << val << " ";
    }
    std::cout << "\n" << std::endl;

    // Cleanup test files
    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      std::remove(("temperature_t" + std::to_string(t) + ".bin").c_str());
    }
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "=== Distributed Stream Example ===\n";
  std::cout << "ERROR: MPI support not enabled!\n";
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON\n";
  std::cout << "and run with: mpirun -np 4 ./distributed_stream\n";
  return 1;
}

#endif // NDARRAY_HAVE_MPI
