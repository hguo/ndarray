/**
 * Distributed Time-Series Analysis Example
 *
 * Demonstrates a complete vis/analysis workflow for processing large time-series
 * scientific datasets in distributed memory.
 *
 * Typical workflow:
 * 1. Load metadata (dimensions, variable info)
 * 2. Decompose domain across ranks
 * 3. Loop over timesteps:
 *    a. Read timestep in parallel
 *    b. Compute local features/statistics
 *    c. Global reduction for aggregated results
 * 4. Output analysis results
 *
 * This example processes a synthetic velocity field to find regions of high
 * vorticity (rotation) over time.
 *
 * Compile and run: mpirun -np 4 ./distributed_analysis
 */

#include <ndarray/config.hh>
#include <iostream>
#include <vector>
#include <cmath>

#if NDARRAY_HAVE_MPI

#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Distributed Time-Series Analysis ===\n";
    std::cout << "Running with " << nprocs << " MPI ranks\n" << std::endl;
  }

  // ===========================================================================
  // Step 1: Create synthetic time-series data (normally from large simulation)
  // ===========================================================================

  const size_t NX = 400, NY = 300;
  const int NUM_TIMESTEPS = 10;

  if (rank == 0) {
    std::cout << "Step 1: Creating synthetic velocity field time-series...\n";

    // Create velocity components: u (x-velocity), v (y-velocity)
    // Simulate a rotating vortex that moves over time
    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      ftk::ndarray<float> u, v;
      u.reshapef(NX, NY);
      v.reshapef(NX, NY);

      // Vortex center moves over time
      float cx = 0.3f + 0.4f * t / NUM_TIMESTEPS;
      float cy = 0.5f;
      float vortex_strength = 10.0f;

      for (size_t i = 0; i < NX; i++) {
        for (size_t j = 0; j < NY; j++) {
          float x = static_cast<float>(i) / NX;
          float y = static_cast<float>(j) / NY;

          // Distance and angle from vortex center
          float dx = x - cx;
          float dy = y - cy;
          float r = std::sqrt(dx*dx + dy*dy);

          // Rotating velocity field: v = (-y, x) / r²
          if (r > 0.01f) {
            u.at(i, j) = -dy / (r*r) * vortex_strength * std::exp(-r*r*20.0f);
            v.at(i, j) =  dx / (r*r) * vortex_strength * std::exp(-r*r*20.0f);
          } else {
            u.at(i, j) = 0.0f;
            v.at(i, j) = 0.0f;
          }
        }
      }

      // Write to binary files
      u.to_binary_file("velocity_u_t" + std::to_string(t) + ".bin");
      v.to_binary_file("velocity_v_t" + std::to_string(t) + ".bin");
    }

    std::cout << "  Created " << NUM_TIMESTEPS << " timesteps of "
              << NX << "×" << NY << " velocity fields\n" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ===========================================================================
  // Step 2: Decompose domain with ghosts (needed for gradient computation)
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 2: Decomposing domain across " << nprocs << " ranks...\n";
  }

  ftk::ndarray<float> u_field;
  ftk::ndarray<float> v_field;

  // 1-layer ghosts for gradient computation
  u_field.decompose(MPI_COMM_WORLD, {NX, NY}, 0, {}, {1, 1});
  v_field.decompose(MPI_COMM_WORLD, {NX, NY}, 0, {}, {1, 1});

  std::cout << "  Rank " << rank << " owns: " << u_field.local_core()
            << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Step 3: Time-series analysis loop
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 3: Processing time-series...\n";
    std::cout << "  Computing vorticity: ω = ∂v/∂x - ∂u/∂y\n" << std::endl;
  }

  // Track maximum vorticity over time
  std::vector<float> max_vorticity_over_time;
  std::vector<std::pair<size_t, size_t>> max_vorticity_location;

  for (int t = 0; t < NUM_TIMESTEPS; t++) {
    // Step 3a: Read velocity components in parallel
    u_field.read_binary_auto("velocity_u_t" + std::to_string(t) + ".bin");
    v_field.read_binary_auto("velocity_v_t" + std::to_string(t) + ".bin");

    // Step 3b: Exchange ghosts for gradient computation
    u_field.exchange_ghosts();
    v_field.exchange_ghosts();

    // Step 3c: Compute vorticity (local computation)
    //   ω = ∂v/∂x - ∂u/∂y (curl of velocity field)
    auto& u = u_field.local_array();
    auto& v = v_field.local_array();

    float local_max_vorticity = 0.0f;
    size_t local_max_i = 0, local_max_j = 0;

    for (size_t i = 1; i < u.dim(0) - 1; i++) {
      for (size_t j = 1; j < v.dim(1) - 1; j++) {
        // Central difference for derivatives
        float dv_dx = (v.at(i+1, j) - v.at(i-1, j)) / 2.0f;
        float du_dy = (u.at(i, j+1) - u.at(i, j-1)) / 2.0f;
        float vorticity = std::abs(dv_dx - du_dy);

        if (vorticity > local_max_vorticity) {
          local_max_vorticity = vorticity;
          local_max_i = i;
          local_max_j = j;
        }
      }
    }

    // Step 3d: Global reduction to find maximum vorticity
    struct {
      float value;
      int rank;
    } local_max, global_max;

    local_max.value = local_max_vorticity;
    local_max.rank = rank;

    MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT_INT,
               MPI_MAXLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      max_vorticity_over_time.push_back(global_max.value);
      std::cout << "  t=" << t << ": max vorticity = " << global_max.value
                << " (on rank " << global_max.rank << ")" << std::endl;
    }

    // Step 3e: Compute velocity magnitude statistics
    float local_min_speed = std::numeric_limits<float>::max();
    float local_max_speed = 0.0f;
    double local_sum_speed = 0.0;
    size_t local_count = u_field.local_core().n();

    for (size_t i = 0; i < u_field.local_core().size(0); i++) {
      for (size_t j = 0; j < u_field.local_core().size(1); j++) {
        float speed = std::sqrt(u.at(i, j) * u.at(i, j) +
                               v.at(i, j) * v.at(i, j));
        local_min_speed = std::min(local_min_speed, speed);
        local_max_speed = std::max(local_max_speed, speed);
        local_sum_speed += speed;
      }
    }

    float global_min_speed, global_max_speed;
    double global_sum_speed;
    size_t global_count;

    MPI_Reduce(&local_min_speed, &global_min_speed, 1, MPI_FLOAT,
               MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max_speed, &global_max_speed, 1, MPI_FLOAT,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_speed, &global_sum_speed, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double global_mean_speed = global_sum_speed / global_count;
      std::cout << "      velocity: min=" << global_min_speed
                << ", max=" << global_max_speed
                << ", mean=" << global_mean_speed << std::endl;
    }
  }

  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Step 4: Post-processing analysis
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 4: Post-processing analysis...\n";

    // Find timestep with maximum vorticity
    int max_timestep = 0;
    float max_value = max_vorticity_over_time[0];
    for (int t = 1; t < NUM_TIMESTEPS; t++) {
      if (max_vorticity_over_time[t] > max_value) {
        max_value = max_vorticity_over_time[t];
        max_timestep = t;
      }
    }

    std::cout << "  Peak vorticity: " << max_value
              << " at timestep " << max_timestep << std::endl;

    // Compute temporal variation
    float mean_vorticity = 0.0f;
    for (float v : max_vorticity_over_time) {
      mean_vorticity += v;
    }
    mean_vorticity /= max_vorticity_over_time.size();

    float variance = 0.0f;
    for (float v : max_vorticity_over_time) {
      float diff = v - mean_vorticity;
      variance += diff * diff;
    }
    variance /= max_vorticity_over_time.size();

    std::cout << "  Temporal statistics:\n";
    std::cout << "    Mean max vorticity: " << mean_vorticity << std::endl;
    std::cout << "    Variance: " << variance << std::endl;
    std::cout << "    Std dev: " << std::sqrt(variance) << std::endl;
    std::cout << std::endl;
  }

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  if (rank == 0) {
    std::cout << "=== Analysis Complete ===\n\n";
    std::cout << "Workflow demonstrated:\n";
    std::cout << "  ✓ Time-series data loading in parallel\n";
    std::cout << "  ✓ Gradient computation with ghost exchange\n";
    std::cout << "  ✓ Feature detection (vorticity maxima)\n";
    std::cout << "  ✓ Statistical analysis over time\n";
    std::cout << "  ✓ Global reductions for aggregated results\n\n";

    std::cout << "This workflow scales to:\n";
    std::cout << "  - Thousands of timesteps\n";
    std::cout << "  - Multi-GB datasets per timestep\n";
    std::cout << "  - Hundreds or thousands of MPI ranks\n\n";

    // Cleanup test files
    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      std::remove(("velocity_u_t" + std::to_string(t) + ".bin").c_str());
      std::remove(("velocity_v_t" + std::to_string(t) + ".bin").c_str());
    }
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "=== Distributed Time-Series Analysis ===\n";
  std::cout << "ERROR: MPI support not enabled!\n";
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON\n";
  std::cout << "and run with: mpirun -np 4 ./distributed_analysis\n";
  return 1;
}

#endif // NDARRAY_HAVE_MPI
