/**
 * Distributed Stencil Computation Example
 *
 * Demonstrates ghost cell exchange for stencil operations in distributed memory.
 * Implements a heat diffusion solver using a 5-point Laplacian stencil.
 *
 * Physics: ∂T/∂t = α∇²T (heat equation)
 * Discretization: T_new = T + α·Δt·(T_i-1,j + T_i+1,j + T_i,j-1 + T_i,j+1 - 4T_i,j)/(Δx²)
 *
 * Workflow:
 * 1. Decompose domain with ghost layers
 * 2. Initialize temperature field
 * 3. Time-stepping loop:
 *    a. Exchange ghosts
 *    b. Apply stencil to compute new values
 *    c. Update field
 * 4. Verify convergence to steady state
 *
 * Compile and run: mpirun -np 4 ./distributed_stencil
 */

#include <ndarray/config.hh>
#include <iostream>
#include <cmath>

#if NDARRAY_HAVE_MPI

#include <ndarray/distributed_ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Distributed Stencil Computation: Heat Diffusion ===\n";
    std::cout << "Running with " << nprocs << " MPI ranks\n" << std::endl;
  }

  // ===========================================================================
  // Setup: Domain decomposition with ghost layers
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 1: Domain decomposition...\n";
  }

  ftk::distributed_ndarray<float> temperature(MPI_COMM_WORLD);
  ftk::distributed_ndarray<float> temperature_new(MPI_COMM_WORLD);

  // 2D domain with 1-layer ghosts for 5-point stencil
  const size_t NX = 200, NY = 160;
  temperature.decompose({NX, NY}, 0, {}, {1, 1});
  temperature_new.decompose({NX, NY}, 0, {}, {1, 1});

  std::cout << "  Rank " << rank << " owns: " << temperature.local_core()
            << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // ===========================================================================
  // Initialization: Set initial temperature field
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 2: Initializing temperature field...\n";
  }

  auto& T = temperature.local_array();
  auto& T_new = temperature_new.local_array();

  // Initialize: hot center, cold boundaries
  for (size_t i = 0; i < temperature.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature.local_core().size(1); j++) {
      // Get global coordinates
      size_t gi = temperature.local_core().start(0) + i;
      size_t gj = temperature.local_core().start(1) + j;

      // Distance from center
      float x = static_cast<float>(gi) / NX - 0.5f;
      float y = static_cast<float>(gj) / NY - 0.5f;
      float r = std::sqrt(x*x + y*y);

      // Hot center (100°C), cold edges (0°C)
      if (r < 0.2f) {
        T.at(i, j) = 100.0f;  // Hot center
      } else {
        T.at(i, j) = 0.0f;    // Cold everywhere else
      }

      T_new.at(i, j) = T.at(i, j);
    }
  }

  // Compute initial statistics
  float local_energy = 0.0f;
  size_t local_count = temperature.local_core().n();
  for (size_t i = 0; i < temperature.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature.local_core().size(1); j++) {
      local_energy += T.at(i, j);
    }
  }

  float global_energy;
  MPI_Reduce(&local_energy, &global_energy, 1, MPI_FLOAT,
             MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "  Initial total energy: " << global_energy << " (arbitrary units)\n";
    std::cout << std::endl;
  }

  // ===========================================================================
  // Time-stepping: Solve heat diffusion with 5-point Laplacian
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 3: Time-stepping heat diffusion...\n";
  }

  // Diffusion parameters
  const float alpha = 0.1f;     // Thermal diffusivity
  const float dx = 1.0f;        // Grid spacing
  const float dt = 0.01f;       // Time step (stability: dt < dx²/(4α))
  const int num_steps = 100;
  const int report_interval = 20;

  auto t_start = MPI_Wtime();

  for (int step = 0; step < num_steps; step++) {
    // Step 3a: Exchange ghost cells
    temperature.exchange_ghosts();

    // Step 3b: Apply 5-point Laplacian stencil
    //   ∇²T ≈ (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4T[i,j]) / dx²
    for (size_t i = 1; i < T.dim(0) - 1; i++) {
      for (size_t j = 1; j < T.dim(1) - 1; j++) {
        float laplacian = (T.at(i-1, j) + T.at(i+1, j) +
                          T.at(i, j-1) + T.at(i, j+1) - 4.0f * T.at(i, j)) / (dx * dx);
        T_new.at(i, j) = T.at(i, j) + alpha * dt * laplacian;
      }
    }

    // Step 3c: Update temperature field (swap pointers)
    std::swap(T, T_new);
    std::swap(temperature.local_array(), temperature_new.local_array());

    // Report progress periodically
    if (step % report_interval == 0 || step == num_steps - 1) {
      // Compute total energy (should be conserved approximately)
      local_energy = 0.0f;
      for (size_t i = 0; i < temperature.local_core().size(0); i++) {
        for (size_t j = 0; j < temperature.local_core().size(1); j++) {
          local_energy += temperature.local_array().at(i, j);
        }
      }

      float step_energy;
      MPI_Reduce(&local_energy, &step_energy, 1, MPI_FLOAT,
                 MPI_SUM, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        std::cout << "  Step " << step << ": total energy = "
                  << step_energy << std::endl;
      }
    }
  }

  auto t_elapsed = MPI_Wtime() - t_start;

  double max_time;
  MPI_Reduce(&t_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n  Completed " << num_steps << " steps in "
              << max_time << " seconds" << std::endl;
    std::cout << "  Time per step: " << (max_time / num_steps) << " seconds"
              << std::endl;
    std::cout << std::endl;
  }

  // ===========================================================================
  // Verification: Check that heat has diffused
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 4: Verification...\n";
  }

  // Compute final statistics
  float local_min = std::numeric_limits<float>::max();
  float local_max = std::numeric_limits<float>::lowest();
  double local_sum = 0.0;

  for (size_t i = 0; i < temperature.local_core().size(0); i++) {
    for (size_t j = 0; j < temperature.local_core().size(1); j++) {
      float value = temperature.local_array().at(i, j);
      local_min = std::min(local_min, value);
      local_max = std::max(local_max, value);
      local_sum += value;
    }
  }

  float global_min, global_max;
  double global_sum;

  MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double global_mean = global_sum / (NX * NY);
    std::cout << "  Final temperature field:\n";
    std::cout << "    Min: " << global_min << " °C\n";
    std::cout << "    Max: " << global_max << " °C\n";
    std::cout << "    Mean: " << global_mean << " °C\n";
    std::cout << std::endl;

    // Verify heat has diffused (max should be lower than initial 100°C)
    if (global_max < 99.0f) {
      std::cout << "  ✓ Heat has diffused as expected\n";
    } else {
      std::cout << "  ✗ Warning: max temperature still very high\n";
    }
    std::cout << std::endl;
  }

  // ===========================================================================
  // Summary
  // ===========================================================================

  if (rank == 0) {
    std::cout << "=== Stencil Computation Complete ===\n\n";
    std::cout << "Key points demonstrated:\n";
    std::cout << "  ✓ Ghost cell exchange before stencil application\n";
    std::cout << "  ✓ 5-point Laplacian on distributed 2D grid\n";
    std::cout << "  ✓ Time-stepping with boundary synchronization\n";
    std::cout << "  ✓ Load-balanced computation across " << nprocs << " ranks\n\n";

    std::cout << "Try with different numbers of ranks:\n";
    std::cout << "  mpirun -np 1 ./distributed_stencil  (no decomposition)\n";
    std::cout << "  mpirun -np 2 ./distributed_stencil  (1D or 2D split)\n";
    std::cout << "  mpirun -np 4 ./distributed_stencil  (2×2 grid)\n";
    std::cout << "  mpirun -np 8 ./distributed_stencil  (2×4 or 4×2 grid)\n";
    std::cout << std::endl;
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI

int main() {
  std::cout << "=== Distributed Stencil Computation ===\n";
  std::cout << "ERROR: MPI support not enabled!\n";
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON\n";
  std::cout << "and run with: mpirun -np 4 ./distributed_stencil\n";
  return 1;
}

#endif // NDARRAY_HAVE_MPI
