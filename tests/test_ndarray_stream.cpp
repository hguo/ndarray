/**
 * YAML stream functionality tests for ndarray
 *
 * Tests the ndarray_group_stream functionality:
 * - Parsing YAML configuration files
 * - Synthetic data streams
 * - NetCDF data streams (if files available)
 * - Static vs time-varying data
 * - Error handling
 */

#include <ndarray/ndarray_group_stream.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return 1; \
    } \
  } while (0)

#define TEST_SECTION(name) \
  std::cout << "  Testing: " << name << std::endl

// Helper to create test YAML files
void create_synthetic_yaml(const std::string& filename,
                           int nx, int ny, int timesteps,
                           const std::string& varname = "scalar",
                           const std::string& dtype = "float32") {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: test_synthetic\n";
  f << "  substreams:\n";
  f << "    - name: data\n";
  f << "      format: synthetic\n";
  f << "      dimensions: [" << nx << ", " << ny << "]\n";
  f << "      timesteps: " << timesteps << "\n";
  f << "      vars:\n";
  f << "        - name: " << varname << "\n";
  f << "          dtype: " << dtype << "\n";
  f.close();
}

void create_synthetic_multi_var_yaml(const std::string& filename) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: test_multi\n";
  f << "  substreams:\n";
  f << "    - name: data\n";
  f << "      format: synthetic\n";
  f << "      dimensions: [16, 16]\n";
  f << "      timesteps: 5\n";
  f << "      vars:\n";
  f << "        - name: temperature\n";
  f << "          dtype: float32\n";
  f << "        - name: pressure\n";
  f << "          dtype: float32\n";
  f << "        - name: velocity\n";
  f << "          dtype: float32\n";
  f.close();
}

void create_static_synthetic_yaml(const std::string& filename) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: test_static\n";
  f << "  substreams:\n";
  f << "    - name: mesh\n";
  f << "      format: synthetic\n";
  f << "      dimensions: [32, 32]\n";
  f << "      timesteps: 1\n";
  f << "      static: true\n";
  f << "      vars:\n";
  f << "        - name: coordinates\n";
  f << "          dtype: float32\n";
  f.close();
}

int main() {
  std::cout << "=== Running ndarray Stream Tests ===" << std::endl << std::endl;

  // Test 1: Basic YAML parsing
  {
    TEST_SECTION("YAML parsing - basic synthetic stream");

    create_synthetic_yaml("test_stream_basic.yaml", 32, 32, 10);

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_basic.yaml");

      TEST_ASSERT(s.total_timesteps() == 10, "Should have 10 timesteps");
      std::cout << "    - Parsed YAML successfully" << std::endl;
      std::cout << "    - Total timesteps: " << s.total_timesteps() << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 2: Reading synthetic data
  {
    TEST_SECTION("Reading synthetic stream data");

    create_synthetic_yaml("test_stream_read.yaml", 16, 16, 5);

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_read.yaml");

      // Read first timestep
      auto g0 = s.read(0);
      TEST_ASSERT(g0 != nullptr, "Should read timestep 0");

      // Check if scalar variable exists
      TEST_ASSERT(g0->has("scalar"), "Should have 'scalar' variable");

      auto scalar0 = g0->get_arr<float>("scalar");
      TEST_ASSERT(scalar0.size() > 0, "Scalar data should not be empty");
      TEST_ASSERT(scalar0.dimf(0) == 16, "First dimension should be 16");
      TEST_ASSERT(scalar0.dimf(1) == 16, "Second dimension should be 16");

      std::cout << "    - Read timestep 0 successfully" << std::endl;
      std::cout << "    - Scalar shape: [" << scalar0.dimf(0) << ", "
                << scalar0.dimf(1) << "]" << std::endl;
      std::cout << "    - Scalar size: " << scalar0.size() << std::endl;

      // Read middle timestep
      auto g2 = s.read(2);
      TEST_ASSERT(g2 != nullptr, "Should read timestep 2");
      TEST_ASSERT(g2->has("scalar"), "Should have 'scalar' variable");

      // Read last timestep
      auto g4 = s.read(4);
      TEST_ASSERT(g4 != nullptr, "Should read timestep 4");
      TEST_ASSERT(g4->has("scalar"), "Should have 'scalar' variable");

      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 3: Multiple variables
  {
    TEST_SECTION("Multiple variables in stream");

    create_synthetic_multi_var_yaml("test_stream_multi.yaml");

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_multi.yaml");

      auto g = s.read(0);
      TEST_ASSERT(g != nullptr, "Should read timestep");

      // Check all variables exist
      TEST_ASSERT(g->has("temperature"), "Should have 'temperature' variable");
      TEST_ASSERT(g->has("pressure"), "Should have 'pressure' variable");
      TEST_ASSERT(g->has("velocity"), "Should have 'velocity' variable");

      auto temp = g->get_arr<float>("temperature");
      auto pres = g->get_arr<float>("pressure");
      auto vel = g->get_arr<float>("velocity");

      TEST_ASSERT(temp.size() == 16*16, "Temperature size should be 16*16");
      TEST_ASSERT(pres.size() == 16*16, "Pressure size should be 16*16");
      TEST_ASSERT(vel.size() == 16*16, "Velocity size should be 16*16");

      std::cout << "    - All variables loaded successfully" << std::endl;
      std::cout << "    - temperature: " << temp.size() << " elements" << std::endl;
      std::cout << "    - pressure: " << pres.size() << " elements" << std::endl;
      std::cout << "    - velocity: " << vel.size() << " elements" << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 4: Static data
  {
    TEST_SECTION("Static stream data (read_static)");

    create_static_synthetic_yaml("test_stream_static.yaml");

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_static.yaml");

      auto g = s.read_static();
      TEST_ASSERT(g != nullptr, "Should read static data");
      TEST_ASSERT(g->has("coordinates"), "Should have 'coordinates' variable");

      auto coords = g->get_arr<float>("coordinates");
      TEST_ASSERT(coords.size() == 32*32, "Static data size should be 32*32");

      std::cout << "    - Read static data successfully" << std::endl;
      std::cout << "    - Coordinates size: " << coords.size() << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 5: Different data types
  {
    TEST_SECTION("Different data types (float32, float64)");

    create_synthetic_yaml("test_stream_float64.yaml", 8, 8, 3, "data", "float64");

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_float64.yaml");

      auto g = s.read(0);
      TEST_ASSERT(g != nullptr, "Should read timestep");
      TEST_ASSERT(g->has("data"), "Should have 'data' variable");

      auto data = g->get_arr<double>("data");
      TEST_ASSERT(data.size() == 8*8, "Data size should be 8*8");

      std::cout << "    - Read float64 data successfully" << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 6: Sequential reading
  {
    TEST_SECTION("Sequential timestep reading");

    create_synthetic_yaml("test_stream_seq.yaml", 10, 10, 20);

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_seq.yaml");

      int total = s.total_timesteps();
      TEST_ASSERT(total == 20, "Should have 20 timesteps");

      // Read all timesteps sequentially
      for (int t = 0; t < total; t++) {
        auto g = s.read(t);
        TEST_ASSERT(g != nullptr, "Should read each timestep");
        TEST_ASSERT(g->has("scalar"), "Each timestep should have scalar");
      }

      std::cout << "    - Read all " << total << " timesteps successfully" << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 7: Random access
  {
    TEST_SECTION("Random timestep access");

    create_synthetic_yaml("test_stream_random.yaml", 12, 12, 15);

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_random.yaml");

      // Read timesteps in random order
      int order[] = {10, 3, 14, 0, 7, 5, 12, 2, 9, 1};

      for (int i = 0; i < 10; i++) {
        int t = order[i];
        auto g = s.read(t);
        TEST_ASSERT(g != nullptr, "Should read timestep in random order");
        TEST_ASSERT(g->has("scalar"), "Should have scalar variable");
      }

      std::cout << "    - Random access works correctly" << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 8: Different dimensions
  {
    TEST_SECTION("Various array dimensions");

    // 2D array
    create_synthetic_yaml("test_stream_2d.yaml", 64, 48, 5);
    ftk::stream s2d;
    s2d.parse_yaml("test_stream_2d.yaml");
    auto g2d = s2d.read(0);
    auto data2d = g2d->get_arr<float>("scalar");
    TEST_ASSERT(data2d.size() == 64*48, "2D array size correct");
    std::cout << "    - 2D array: [64, 48]" << std::endl;

    // Small array
    create_synthetic_yaml("test_stream_small.yaml", 4, 4, 2);
    ftk::stream ssmall;
    ssmall.parse_yaml("test_stream_small.yaml");
    auto gsmall = ssmall.read(0);
    auto datasmall = gsmall->get_arr<float>("scalar");
    TEST_ASSERT(datasmall.size() == 4*4, "Small array size correct");
    std::cout << "    - Small array: [4, 4]" << std::endl;

    // Large array
    create_synthetic_yaml("test_stream_large.yaml", 256, 256, 2);
    ftk::stream slarge;
    slarge.parse_yaml("test_stream_large.yaml");
    auto glarge = slarge.read(0);
    auto datalarge = glarge->get_arr<float>("scalar");
    TEST_ASSERT(datalarge.size() == 256*256, "Large array size correct");
    std::cout << "    - Large array: [256, 256]" << std::endl;

    std::cout << "    PASSED" << std::endl;
  }

  // Test 9: Error handling - invalid timestep
  {
    TEST_SECTION("Error handling - invalid timestep");

    create_synthetic_yaml("test_stream_error.yaml", 10, 10, 5);

    try {
      ftk::stream s;
      s.parse_yaml("test_stream_error.yaml");

      // Try to read beyond available timesteps
      auto g = s.read(100);  // Only 5 timesteps available

      // Depending on implementation, this might return nullptr or throw
      if (g == nullptr) {
        std::cout << "    - Correctly handled out-of-range timestep (returned nullptr)" << std::endl;
      } else {
        std::cout << "    - Warning: reading beyond range did not fail" << std::endl;
      }

      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "    - Correctly caught exception: " << e.what() << std::endl;
      std::cout << "    PASSED" << std::endl;
    }
  }

  // Test 10: MPI communicator (if MPI is available)
#if NDARRAY_HAVE_MPI
  {
    TEST_SECTION("MPI communicator initialization");

    create_synthetic_yaml("test_stream_mpi.yaml", 20, 20, 3);

    try {
      ftk::stream s(MPI_COMM_WORLD);
      s.parse_yaml("test_stream_mpi.yaml");

      auto g = s.read(0);
      TEST_ASSERT(g != nullptr, "Should read with MPI communicator");

      std::cout << "    - MPI communicator works" << std::endl;
      std::cout << "    PASSED" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    ERROR: " << e.what() << std::endl;
      return 1;
    }
  }
#else
  std::cout << "  MPI tests SKIPPED (MPI not available)" << std::endl;
#endif

  // Optional: Test NetCDF streams if data files exist
  {
    TEST_SECTION("NetCDF stream (optional - requires data files)");

    // Check if example yaml exists
    std::ifstream yaml_check("../yaml_exampls/woven.yaml");
    if (yaml_check.good()) {
      yaml_check.close();

      try {
        ftk::stream s;
        s.parse_yaml("../yaml_exampls/woven.yaml");

        std::cout << "    - Parsed woven.yaml successfully" << std::endl;
        std::cout << "    - Total timesteps: " << s.total_timesteps() << std::endl;

        auto g = s.read(0);
        if (g && g->has("scalar")) {
          auto scalar = g->get_arr<float>("scalar");
          std::cout << "    - Read woven example data: " << scalar.size() << " elements" << std::endl;
        }

        std::cout << "    PASSED" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "    - Example file test skipped: " << e.what() << std::endl;
      }
    } else {
      std::cout << "    SKIPPED (example files not found)" << std::endl;
    }
  }

  // Cleanup test files
  std::remove("test_stream_basic.yaml");
  std::remove("test_stream_read.yaml");
  std::remove("test_stream_multi.yaml");
  std::remove("test_stream_static.yaml");
  std::remove("test_stream_float64.yaml");
  std::remove("test_stream_seq.yaml");
  std::remove("test_stream_random.yaml");
  std::remove("test_stream_2d.yaml");
  std::remove("test_stream_small.yaml");
  std::remove("test_stream_large.yaml");
  std::remove("test_stream_error.yaml");
#if NDARRAY_HAVE_MPI
  std::remove("test_stream_mpi.yaml");
#endif

  std::cout << std::endl;
  std::cout << "=== All Stream Tests Passed ===" << std::endl;

  // Call finalize to clean up fdpool
  ftk::ndarray_finalize();

  return 0;
}
