/**
 * Distributed Stream with YAML Configuration Example
 *
 * NOTE: This example needs updating to use the unified stream API with
 * the new YAML format (per-variable distribution configuration).
 *
 * The old distributed_stream class no longer exists. The unified stream
 * now supports per-variable distribution types (distributed, replicated, auto)
 * configured via YAML.
 *
 * For the new YAML format, see: PHASE4_STREAM_INTEGRATION.md
 *
 * For working examples, see:
 * - distributed_io.cpp: Direct ndarray usage with decomposition
 * - distributed_stencil.cpp: Ghost exchange for stencil operations
 * - distributed_analysis.cpp: Time-series processing across ranks
 *
 * TODO: Update to use ftk::stream<> with new YAML format
 *
 * Compile and run: mpirun -np 4 ./distributed_stream_yaml
 */

#include <ndarray/config.hh>
#include <iostream>
#include <fstream>
#include <cmath>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#include <ndarray/ndarray_group_stream.hh>  // Includes distributed support when MPI enabled
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Distributed Stream with YAML Configuration ===\n";
    std::cout << "Running with " << nprocs << " MPI ranks\n" << std::endl;
  }

  // ===========================================================================
  // Step 1: Create synthetic data files
  // ===========================================================================

  const size_t NX = 300, NY = 200;
  const int NUM_TIMESTEPS = 5;

  if (rank == 0) {
    std::cout << "Step 1: Creating synthetic data files...\n";

    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      // Temperature field
      ftk::ndarray<float> temperature;
      temperature.reshapef(NX, NY);

      // Pressure field
      ftk::ndarray<float> pressure;
      pressure.reshapef(NX, NY);

      // Generate data with time evolution
      for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
          float x = static_cast<float>(i) / NX;
          float y = static_cast<float>(j) / NY;
          float time = static_cast<float>(t) * 0.5f;

          temperature.at(i, j) = std::sin(x * 3.14159f + time) *
                                std::cos(y * 3.14159f) * 50.0f + 273.15f;
          pressure.at(i, j) = std::exp(-(x-0.5f)*(x-0.5f) - (y-0.5f)*(y-0.5f)) *
                             (1.0f + 0.1f * time) * 101325.0f;
        }
      }

      // Write to binary files
      temperature.to_binary_file("temperature_" + std::to_string(t) + ".bin");
      pressure.to_binary_file("pressure_" + std::to_string(t) + ".bin");
    }

    std::cout << "  Created " << NUM_TIMESTEPS << " timesteps\n" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ===========================================================================
  // Step 2: Create YAML configuration file
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 2: Creating YAML configuration...\n";

    std::ofstream yaml_file("stream_config.yaml");
    yaml_file << R"(# Distributed stream configuration

# Domain decomposition
decomposition:
  global_dims: [300, 200]    # Global domain size
  nprocs: 0                  # 0 = use all available ranks
  pattern: []                # Empty = automatic decomposition
  ghost: [1, 1]              # 1-layer ghosts in each dimension

# Data streams
streams:
  # Temperature stream
  - name: temperature_stream
    format: binary
    filenames: "temperature_*.bin"
    dimensions: [300, 200]
    vars:
      - name: temperature
        dtype: float32
    enabled: true

  # Pressure stream
  - name: pressure_stream
    format: binary
    filenames: "pressure_*.bin"
    dimensions: [300, 200]
    vars:
      - name: pressure
        dtype: float32
    enabled: true
)";
    yaml_file.close();

    std::cout << "  Created stream_config.yaml\n" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ===========================================================================
  // Step 3: Load and parse YAML configuration
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 3: Loading YAML configuration...\n";
  }

  ftk::distributed_stream<float> stream(MPI_COMM_WORLD);

  try {
    stream.parse_yaml("stream_config.yaml");

    if (rank == 0) {
      std::cout << "  Configuration loaded successfully\n";
      std::cout << "  Number of timesteps: " << stream.n_timesteps() << "\n";
      std::cout << "  Global dimensions: [";
      auto dims = stream.global_dims();
      for (size_t i = 0; i < dims.size(); i++) {
        std::cout << dims[i];
        if (i < dims.size() - 1) std::cout << ", ";
      }
      std::cout << "]\n";
      std::cout << "  Ghost layers: [";
      auto ghost = stream.ghost_layers();
      for (size_t i = 0; i < ghost.size(); i++) {
        std::cout << ghost[i];
        if (i < ghost.size() - 1) std::cout << ", ";
      }
      std::cout << "]\n" << std::endl;
    }

  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "Error loading YAML: " << e.what() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  // ===========================================================================
  // Step 4: Process time-series using stream
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 4: Processing time-series...\n";
  }

  std::vector<double> mean_temp_over_time;
  std::vector<double> mean_pres_over_time;

  for (int t = 0; t < stream.n_timesteps(); t++) {
    // Read all variables at this timestep
    auto group = stream.read(t);

    // Exchange ghosts for all arrays
    group->exchange_ghosts_all();

    // Access individual arrays
    auto& temperature = (*group)["temperature"];
    auto& pressure = (*group)["pressure"];

    // Compute local statistics
    double local_temp_sum = 0.0;
    double local_pres_sum = 0.0;
    size_t local_count = temperature.local_core().n();

    auto& temp_local = temperature.local_array();
    auto& pres_local = pressure.local_array();

    for (size_t i = 0; i < temperature.local_core().size(0); i++) {
      for (size_t j = 0; j < temperature.local_core().size(1); j++) {
        local_temp_sum += temp_local.at(i, j);
        local_pres_sum += pres_local.at(i, j);
      }
    }

    // Global reduction
    double global_temp_sum, global_pres_sum;
    size_t global_count;

    MPI_Reduce(&local_temp_sum, &global_temp_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pres_sum, &global_pres_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double mean_temp = global_temp_sum / global_count;
      double mean_pres = global_pres_sum / global_count;
      mean_temp_over_time.push_back(mean_temp);
      mean_pres_over_time.push_back(mean_pres);

      std::cout << "  t=" << t
                << ": T_mean=" << mean_temp << " K"
                << ", P_mean=" << mean_pres << " Pa" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Step 5: Alternative - functional style
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 5: Functional style processing...\n";
  }

  // Process with lambda
  stream.for_each_timestep([&](int t, auto& group) {
    // Exchange ghosts
    group.exchange_ghosts_all();

    // Compute gradient magnitude
    auto& temperature = group["temperature"];
    auto& temp_local = temperature.local_array();

    float local_max_grad = 0.0f;
    for (size_t i = 1; i < temp_local.dim(0) - 1; i++) {
      for (size_t j = 1; j < temp_local.dim(1) - 1; j++) {
        float dx = (temp_local.at(i+1, j) - temp_local.at(i-1, j)) / 2.0f;
        float dy = (temp_local.at(i, j+1) - temp_local.at(i, j-1)) / 2.0f;
        float grad = std::sqrt(dx*dx + dy*dy);
        local_max_grad = std::max(local_max_grad, grad);
      }
    }

    float global_max_grad;
    MPI_Reduce(&local_max_grad, &global_max_grad, 1, MPI_FLOAT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "  t=" << t << ": max |∇T| = "
                << global_max_grad << " K/cell" << std::endl;
    }
  });

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Summary
  // ===========================================================================

  if (rank == 0) {
    std::cout << "=== YAML Stream Example Complete ===\n\n";

    std::cout << "Benefits of YAML configuration:\n";
    std::cout << "  ✓ Declarative data source specification\n";
    std::cout << "  ✓ Easy to modify without recompiling\n";
    std::cout << "  ✓ Supports multiple streams/formats in one config\n";
    std::cout << "  ✓ Automatic file discovery with wildcards\n";
    std::cout << "  ✓ Domain decomposition parameters in config\n";
    std::cout << "  ✓ Compatible with existing ndarray_stream YAMLs\n\n";

    std::cout << "YAML Configuration Structure:\n";
    std::cout << "  decomposition:        # Domain decomposition\n";
    std::cout << "    global_dims: [...]  # Global dimensions\n";
    std::cout << "    pattern: [...]      # Decomposition pattern\n";
    std::cout << "    ghost: [...]        # Ghost layers\n";
    std::cout << "  streams:              # List of data sources\n";
    std::cout << "    - name: ...         # Stream name\n";
    std::cout << "      format: ...       # netcdf, binary, h5, etc.\n";
    std::cout << "      filenames: ...    # File pattern\n";
    std::cout << "      vars: [...]       # Variable list\n\n";

    // Cleanup
    std::remove("stream_config.yaml");
    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      std::remove(("temperature_" + std::to_string(t) + ".bin").c_str());
      std::remove(("pressure_" + std::to_string(t) + ".bin").c_str());
    }
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI || !NDARRAY_HAVE_YAML

int main() {
  std::cout << "=== Distributed Stream with YAML ===\n";
  std::cout << "ERROR: Requires MPI and YAML support!\n";
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON -DNDARRAY_USE_YAML=ON\n";
  std::cout << "and run with: mpirun -np 4 ./distributed_stream_yaml\n";
  return 1;
}

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML
