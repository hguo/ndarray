/**
 * ADIOS2 stream tests for ndarray
 *
 * Tests the ndarray_group_stream functionality with ADIOS2 BP files:
 * - Reading BP files via YAML stream configuration
 * - Time-series data across multiple BP files
 * - Variable name aliasing
 * - Static vs time-varying substreams
 *
 * Requires: NDARRAY_HAVE_YAML && NDARRAY_HAVE_ADIOS2
 */

#include <ndarray/ndarray_group_stream.hh>
#include <iostream>
#include <cassert>
#include <fstream>

#if NDARRAY_HAVE_ADIOS2
#include <adios2.h>
#endif

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

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

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "=== Running ADIOS2 Stream Tests ===" << std::endl << std::endl;

#if NDARRAY_HAVE_ADIOS2 && NDARRAY_HAVE_YAML

  // Test 1: ADIOS2 stream with time series (one variable per file)
  {
    TEST_SECTION("ADIOS2 stream with time series data");

    try {
      const size_t nx = 10, ny = 12;
      const int num_files = 3;

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_WORLD);
#else
      adios2::ADIOS adios;
#endif

      // Create BP files for time series
      for (int t = 0; t < num_files; t++) {
        std::string filename = "test_stream_adios2_t" + std::to_string(t) + ".bp";

        adios2::IO io = adios.DeclareIO("TestIO_" + std::to_string(t));
        io.SetEngine("BP4");

        adios2::Engine writer = io.Open(filename, adios2::Mode::Write);

        // Create test data
        std::vector<float> temperature(nx * ny);
        for (size_t i = 0; i < nx * ny; i++) {
          temperature[i] = t * 100.0f + i;
        }

        // Define variable with global, offset, and local dimensions
        // ADIOS2 uses C-order: ny, nx
        auto var_temp = io.DefineVariable<float>("temperature",
          {ny, nx},    // Global dimensions
          {0, 0},      // Offset
          {ny, nx});   // Local dimensions

        writer.BeginStep();
        writer.Put(var_temp, temperature.data());
        writer.EndStep();

        writer.Close();
      }

      // Create YAML configuration
      std::ofstream yaml("test_stream_adios2.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_adios2\n";
      yaml << "  substreams:\n";
      yaml << "    - name: bp_data\n";
      yaml << "      format: adios2\n";
      yaml << "      filenames:\n";
      for (int t = 0; t < num_files; t++) {
        yaml << "        - test_stream_adios2_t" << t << ".bp\n";
      }
      yaml << "      vars:\n";
      yaml << "        - name: temperature\n";
      yaml << "          dtype: float32\n";
      yaml.close();

      // Test reading via stream
      ftk::stream s;
      s.parse_yaml("test_stream_adios2.yaml");

      TEST_ASSERT(s.total_timesteps() == num_files, "Should have 3 timesteps");
      std::cout << "    - Total timesteps: " << s.total_timesteps() << std::endl;

      // Read first timestep
      auto g0 = s.read(0);
      TEST_ASSERT(g0 != nullptr, "Failed to read timestep 0");
      TEST_ASSERT(g0->has("temperature"), "Missing temperature at timestep 0");

      auto temp0 = g0->get_arr<float>("temperature");
      TEST_ASSERT(temp0.size() == nx * ny, "Wrong array size");
      TEST_ASSERT(std::abs(temp0[0] - 0.0f) < 1e-5f, "Wrong data at t=0[0]");
      TEST_ASSERT(std::abs(temp0[1] - 1.0f) < 1e-5f, "Wrong data at t=0[1]");

      // Read second and last timesteps
      auto g1 = s.read(1);
      auto temp1 = g1->get_arr<float>("temperature");
      TEST_ASSERT(std::abs(temp1[0] - 100.0f) < 1e-5f, "Wrong data at t=1");

      auto g2 = s.read(2);
      auto temp2 = g2->get_arr<float>("temperature");
      TEST_ASSERT(std::abs(temp2[0] - 200.0f) < 1e-5f, "Wrong data at t=2");

      std::cout << "    - Successfully read all timesteps with correct data values" << std::endl;
      std::cout << "    PASSED" << std::endl;

      // Cleanup
      for (int t = 0; t < num_files; t++) {
        std::string filename = "test_stream_adios2_t" + std::to_string(t) + ".bp";
        std::remove(filename.c_str());
      }
      std::remove("test_stream_adios2.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    ADIOS2 stream test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 2: ADIOS2 stream with multiple variables
  {
    TEST_SECTION("ADIOS2 stream with multiple variables");

    try {
      const size_t nx = 8, ny = 10;

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_WORLD);
#else
      adios2::ADIOS adios;
#endif

      // Create BP file with multiple variables
      adios2::IO io = adios.DeclareIO("TestIO_Multi");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_stream_adios2_multi.bp", adios2::Mode::Write);

      std::vector<float> temp(nx * ny);
      std::vector<double> vel(nx * ny);
      for (size_t i = 0; i < nx * ny; i++) {
        temp[i] = 20.0f + i * 0.1f;
        vel[i] = 5.0 + i * 0.01;
      }

      auto var_temp = io.DefineVariable<float>("temperature",
        {ny, nx}, {0, 0}, {ny, nx});
      auto var_vel = io.DefineVariable<double>("velocity",
        {ny, nx}, {0, 0}, {ny, nx});

      writer.BeginStep();
      writer.Put(var_temp, temp.data());
      writer.Put(var_vel, vel.data());
      writer.EndStep();

      writer.Close();

      // Create YAML config
      std::ofstream yaml("test_stream_adios2_multi.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_multi\n";
      yaml << "  substreams:\n";
      yaml << "    - name: multi_vars\n";
      yaml << "      format: adios2\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_adios2_multi.bp\n";
      yaml << "      vars:\n";
      yaml << "        - name: temperature\n";
      yaml << "          dtype: float32\n";
      yaml << "        - name: velocity\n";
      yaml << "          dtype: float64\n";
      yaml.close();

      // Test reading
      ftk::stream s;
      s.parse_yaml("test_stream_adios2_multi.yaml");

      auto g = s.read(0);
      TEST_ASSERT(g->has("temperature"), "Missing temperature");
      TEST_ASSERT(g->has("velocity"), "Missing velocity");

      auto temp_arr = g->get_arr<float>("temperature");
      auto vel_arr = g->get_arr<double>("velocity");

      TEST_ASSERT(temp_arr.size() == nx * ny, "Wrong temperature size");
      TEST_ASSERT(vel_arr.size() == nx * ny, "Wrong velocity size");
      TEST_ASSERT(std::abs(temp_arr[0] - 20.0f) < 1e-5f, "Wrong temperature value");
      TEST_ASSERT(std::abs(vel_arr[0] - 5.0) < 1e-10, "Wrong velocity value");

      std::cout << "    - Successfully read multiple variables" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_adios2_multi.bp");
      std::remove("test_stream_adios2_multi.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    Multi-variable test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 3: Variable name aliasing
  {
    TEST_SECTION("Variable name aliasing (possible_names)");

    try {
      const size_t nx = 6, ny = 8;

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_WORLD);
#else
      adios2::ADIOS adios;
#endif

      // Create BP file with variable "temperature"
      adios2::IO io = adios.DeclareIO("TestIO_Alias");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_stream_adios2_alias.bp", adios2::Mode::Write);

      std::vector<float> data(nx * ny, 42.0f);
      auto var = io.DefineVariable<float>("temperature",
        {ny, nx}, {0, 0}, {ny, nx});

      writer.BeginStep();
      writer.Put(var, data.data());
      writer.EndStep();
      writer.Close();

      // Create YAML that looks for alternative names
      std::ofstream yaml("test_stream_adios2_alias.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_alias\n";
      yaml << "  substreams:\n";
      yaml << "    - name: alias_test\n";
      yaml << "      format: adios2\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_adios2_alias.bp\n";
      yaml << "      vars:\n";
      yaml << "        - name: temp\n";
      yaml << "          possible_names: [TEMP, Temperature, temperature, temp]\n";
      yaml << "          dtype: float32\n";
      yaml.close();

      // Test reading
      ftk::stream s;
      s.parse_yaml("test_stream_adios2_alias.yaml");

      auto g = s.read(0);
      TEST_ASSERT(g->has("temp"), "Should find 'temp' via aliasing");

      auto arr = g->get_arr<float>("temp");
      TEST_ASSERT(std::abs(arr[0] - 42.0f) < 1e-5f, "Wrong aliased data value");

      std::cout << "    - Variable aliasing works correctly" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_adios2_alias.bp");
      std::remove("test_stream_adios2_alias.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    Aliasing test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 4: Static ADIOS2 substream
  {
    TEST_SECTION("Static ADIOS2 substream");

    try {
      const size_t nx = 10, ny = 10;

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_WORLD);
#else
      adios2::ADIOS adios;
#endif

      // Create static BP file (coordinates/mesh)
      adios2::IO io = adios.DeclareIO("TestIO_Static");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_stream_adios2_static.bp", adios2::Mode::Write);

      std::vector<float> coords(nx * ny);
      for (size_t i = 0; i < nx * ny; i++) {
        coords[i] = static_cast<float>(i);
      }

      auto var = io.DefineVariable<float>("coordinates",
        {ny, nx}, {0, 0}, {ny, nx});

      writer.BeginStep();
      writer.Put(var, coords.data());
      writer.EndStep();
      writer.Close();

      // Create YAML with static substream
      std::ofstream yaml("test_stream_adios2_static.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_static\n";
      yaml << "  substreams:\n";
      yaml << "    - name: static_mesh\n";
      yaml << "      format: adios2\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_adios2_static.bp\n";
      yaml << "      vars:\n";
      yaml << "        - name: coordinates\n";
      yaml << "          dtype: float32\n";
      yaml << "      static: true\n";
      yaml.close();

      // Test reading static data
      ftk::stream s;
      s.parse_yaml("test_stream_adios2_static.yaml");

      auto g_static = s.read_static();
      TEST_ASSERT(g_static != nullptr, "Failed to read static data");
      TEST_ASSERT(g_static->has("coordinates"), "Missing coordinates");

      auto coords_arr = g_static->get_arr<float>("coordinates");
      TEST_ASSERT(coords_arr.size() == nx * ny, "Wrong static array size");
      TEST_ASSERT(std::abs(coords_arr[0] - 0.0f) < 1e-5f, "Wrong static data[0]");
      TEST_ASSERT(std::abs(coords_arr[10] - 10.0f) < 1e-5f, "Wrong static data[10]");

      std::cout << "    - Static substream works correctly" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_adios2_static.bp");
      std::remove("test_stream_adios2_static.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    Static substream test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  std::cout << std::endl << "=== All ADIOS2 Stream Tests Passed ===" << std::endl;

#else
  std::cout << "ADIOS2 and/or YAML support not available - tests skipped" << std::endl;
  std::cout << "Build with -DNDARRAY_USE_ADIOS2=TRUE -DNDARRAY_USE_YAML=TRUE to enable" << std::endl;
#endif

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
