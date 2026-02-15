/**
 * Distributed Stream Dry-Run Example
 *
 * Demonstrates using dry-run mode to test and validate distributed stream
 * configurations without requiring mpirun. Useful for:
 * - Testing YAML configuration syntax
 * - Validating file paths and discovery
 * - Checking decomposition parameters
 * - Quick iteration during development
 *
 * Compile and run: ./distributed_stream_dryrun (no mpirun needed!)
 */

#include <ndarray/config.hh>
#include <iostream>
#include <fstream>
#include <cmath>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#include <ndarray/ndarray_group_stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "=== Distributed Stream Dry-Run Example ===\n" << std::endl;
  }

  // ===========================================================================
  // Step 1: Create test files and YAML configuration
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 1: Setting up test configuration...\n";

    // Create some dummy files for testing file discovery
    const int NUM_FILES = 3;
    for (int i = 0; i < NUM_FILES; i++) {
      ftk::ndarray<float> data;
      data.reshapef(100, 80);
      data.to_binary_file("test_data_" + std::to_string(i) + ".bin");
    }

    // Create YAML configuration
    std::ofstream yaml_file("dryrun_config.yaml");
    yaml_file << R"(# Test configuration for dry-run

decomposition:
  global_dims: [100, 80]
  nprocs: 4
  pattern: [2, 2]  # 2x2 grid
  ghost: [1, 1]    # 1-layer ghosts

streams:
  - name: test_stream
    format: binary
    filenames: "test_data_*.bin"
    dimensions: [100, 80]
    vars:
      - name: temperature
        dtype: float32
    enabled: true
)";
    yaml_file.close();

    std::cout << "  Created test files and YAML configuration\n" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ===========================================================================
  // Step 2: Test dry-run mode (report only)
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 2: Testing dry-run mode (report only)...\n";
  }

  ftk::distributed_stream<float> stream(MPI_COMM_WORLD);

  // Enable dry-run mode BEFORE parsing YAML
  stream.set_dry_run(true, true);  // true = dry-run, true = report only

  // Parse YAML - will automatically report configuration
  stream.parse_yaml("dryrun_config.yaml");

  if (rank == 0) {
    std::cout << "Configuration validated successfully!\n" << std::endl;
  }

  // ===========================================================================
  // Step 3: Test dry-run with read operations
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 3: Testing read operations in dry-run mode...\n";
  }

  // Simulate reading timesteps
  for (int t = 0; t < stream.n_timesteps(); t++) {
    auto group = stream.read(t);
    // In report-only mode, group will be nullptr
    // Information is printed to console instead
  }

  if (rank == 0) {
    std::cout << "\n" << std::endl;
  }

  // ===========================================================================
  // Step 4: Test different decomposition patterns
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 4: Testing different decomposition patterns...\n" << std::endl;
  }

  // Test 1D decomposition
  if (rank == 0) {
    std::cout << "--- 1D Decomposition (4x1) ---" << std::endl;
  }

  ftk::distributed_stream<float> stream_1d(MPI_COMM_WORLD);
  stream_1d.set_dry_run(true, true);
  stream_1d.parse_yaml("dryrun_config.yaml");
  stream_1d.set_decomposition({100, 80}, 4, {4, 1}, {1, 0});

  if (rank == 0) {
    stream_1d.report_configuration();
  }

  // Test automatic decomposition
  if (rank == 0) {
    std::cout << "--- Automatic Decomposition ---" << std::endl;
  }

  ftk::distributed_stream<float> stream_auto(MPI_COMM_WORLD);
  stream_auto.set_dry_run(true, true);
  stream_auto.parse_yaml("dryrun_config.yaml");
  stream_auto.set_decomposition({100, 80}, 4, {}, {1, 1});  // Empty pattern = auto

  if (rank == 0) {
    stream_auto.report_configuration();
  }

  // ===========================================================================
  // Step 5: Test invalid configuration detection
  // ===========================================================================

  if (rank == 0) {
    std::cout << "Step 5: Testing error detection...\n";
  }

  // Create invalid YAML
  if (rank == 0) {
    std::ofstream bad_yaml("bad_config.yaml");
    bad_yaml << R"(
decomposition:
  global_dims: [100, 80]
  ghost: [1, 1]

streams:
  - name: nonexistent
    format: binary
    filenames: "doesnotexist_*.bin"  # No matching files
    dimensions: [100, 80]
    vars:
      - name: data
    enabled: true
    optional: false  # Will cause error if files not found
)";
    bad_yaml.close();

    std::cout << "  Testing with invalid configuration..." << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  ftk::distributed_stream<float> stream_bad(MPI_COMM_WORLD);
  stream_bad.set_dry_run(true, true);

  try {
    stream_bad.parse_yaml("bad_config.yaml");
    if (rank == 0) {
      std::cout << "  Warning: Expected error not caught" << std::endl;
    }
  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cout << "  ✓ Error correctly detected: " << e.what() << std::endl;
    }
  }

  if (rank == 0) {
    std::cout << std::endl;
  }

  // ===========================================================================
  // Cleanup and summary
  // ===========================================================================

  if (rank == 0) {
    std::cout << "=== Dry-Run Example Complete ===\n" << std::endl;

    std::cout << "Dry-run mode benefits:" << std::endl;
    std::cout << "  ✓ Test YAML configuration without mpirun" << std::endl;
    std::cout << "  ✓ Validate file discovery and paths" << std::endl;
    std::cout << "  ✓ Check decomposition parameters" << std::endl;
    std::cout << "  ✓ Debug configuration issues quickly" << std::endl;
    std::cout << "  ✓ Works on workstation before HPC submission" << std::endl;
    std::cout << std::endl;

    std::cout << "Usage patterns:" << std::endl;
    std::cout << "  1. Report only (no data reading):" << std::endl;
    std::cout << "     stream.set_dry_run(true, true);" << std::endl;
    std::cout << std::endl;
    std::cout << "  2. Serial read for validation:" << std::endl;
    std::cout << "     stream.set_dry_run(true, false);" << std::endl;
    std::cout << std::endl;
    std::cout << "  3. Normal MPI execution:" << std::endl;
    std::cout << "     stream.set_dry_run(false);  // or don't call" << std::endl;
    std::cout << std::endl;

    std::cout << "Typical workflow:" << std::endl;
    std::cout << "  1. Create YAML configuration" << std::endl;
    std::cout << "  2. Test with dry-run on workstation" << std::endl;
    std::cout << "  3. Fix any configuration issues" << std::endl;
    std::cout << "  4. Run with mpirun on HPC system" << std::endl;
    std::cout << std::endl;

    // Cleanup
    for (int i = 0; i < 3; i++) {
      std::remove(("test_data_" + std::to_string(i) + ".bin").c_str());
    }
    std::remove("dryrun_config.yaml");
    std::remove("bad_config.yaml");
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI || !NDARRAY_HAVE_YAML

int main() {
  std::cout << "=== Distributed Stream Dry-Run Example ===" << std::endl;
  std::cout << "ERROR: Requires MPI and YAML support!" << std::endl;
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON -DNDARRAY_USE_YAML=ON" << std::endl;
  return 1;
}

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML
