/**
 * YAML stream tests for ADIOS2 data
 *
 * Tests the ndarray_group_stream functionality with ADIOS2 BP files:
 * - Reading time-series data from BP files via YAML config
 * - Static vs time-varying substreams
 * - Variable name aliasing (possible_names)
 * - Optional variables
 * - Multiple variables from BP files
 * - Multi-file time-series
 *
 * Requires: NDARRAY_HAVE_YAML && NDARRAY_HAVE_ADIOS2
 */

#include <ndarray/ndarray_group_stream.hh>
#include <iostream>
#include <cassert>
#include <cmath>
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

#if NDARRAY_HAVE_ADIOS2

// Helper: Create test BP files with time-series data
void create_test_bp_files(int num_timesteps) {
  std::cout << "    Creating test BP files..." << std::endl;

#if NDARRAY_HAVE_MPI
  adios2::ADIOS adios(MPI_COMM_WORLD);
#else
  adios2::ADIOS adios;
#endif

  // Create multiple BP files for time-series
  for (int t = 0; t < num_timesteps; t++) {
    std::string filename = "test_stream_" + std::to_string(t) + ".bp";
    
    adios2::IO io = adios.DeclareIO("TestIO" + std::to_string(t));
    io.SetEngine("BP4");
    
    adios2::Engine writer = io.Open(filename, adios2::Mode::Write);
    
    // Create test data
    const int nx = 20, ny = 30;
    std::vector<float> temperature(nx * ny);
    std::vector<float> pressure(nx * ny);
    std::vector<double> velocity(nx * ny);
    
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        int idx = i + j * nx;
        temperature[idx] = 273.15f + t * 10.0f + i * 0.1f + j * 0.2f;
        pressure[idx] = 101325.0f + t * 100.0f + i * 10.0f;
        velocity[idx] = 5.0 + t * 0.5 + i * 0.01;
      }
    }
    
    // Define variables (note: ADIOS2 uses C-order dimensions)
    auto var_temp = io.DefineVariable<float>("temperature", {(size_t)ny, (size_t)nx});
    auto var_pres = io.DefineVariable<float>("pressure", {(size_t)ny, (size_t)nx});
    auto var_vel = io.DefineVariable<double>("velocity", {(size_t)ny, (size_t)nx});
    
    writer.BeginStep();
    writer.Put(var_temp, temperature.data());
    writer.Put(var_pres, pressure.data());
    writer.Put(var_vel, velocity.data());
    writer.EndStep();
    
    writer.Close();
  }
  
  std::cout << "    Created " << num_timesteps << " BP files" << std::endl;
}

// Helper: Create static BP file
void create_static_bp_file() {
  std::cout << "    Creating static BP file..." << std::endl;

#if NDARRAY_HAVE_MPI
  adios2::ADIOS adios(MPI_COMM_WORLD);
#else
  adios2::ADIOS adios;
#endif

  adios2::IO io = adios.DeclareIO("StaticIO");
  io.SetEngine("BP4");
  
  adios2::Engine writer = io.Open("test_static.bp", adios2::Mode::Write);
  
  const int nx = 40, ny = 40;
  std::vector<float> coordinates(nx * ny);
  
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      coordinates[i + j * nx] = i * 1.0f + j * 100.0f;
    }
  }
  
  auto var = io.DefineVariable<float>("coordinates", {(size_t)ny, (size_t)nx});
  
  writer.BeginStep();
  writer.Put(var, coordinates.data());
  writer.EndStep();
  
  writer.Close();
  
  std::cout << "    Created static BP file" << std::endl;
}

// Helper: Create YAML config for time-varying ADIOS2 stream
void create_adios2_yaml(const std::string& filename, int num_timesteps) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: adios2_test_stream\n";
  f << "  substreams:\n";
  f << "    - name: bp_data\n";
  f << "      format: adios2\n";
  f << "      filenames:\n";
  for (int t = 0; t < num_timesteps; t++) {
    f << "        - test_stream_" << t << ".bp\n";
  }
  f << "      vars:\n";
  f << "        - name: temp\n";
  f << "          possible_names: [temperature, Temperature, TEMP]\n";
  f << "        - name: pres\n";
  f << "          possible_names: [pressure, Pressure]\n";
  f << "        - name: vel\n";
  f << "          possible_names: [velocity]\n";
  f.close();
}

// Helper: Create YAML config with optional variable
void create_adios2_yaml_optional(const std::string& filename) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: adios2_optional\n";
  f << "  substreams:\n";
  f << "    - name: bp_data\n";
  f << "      format: adios2\n";
  f << "      filenames:\n";
  f << "        - test_stream_0.bp\n";
  f << "      vars:\n";
  f << "        - name: temp\n";
  f << "          possible_names: [temperature]\n";
  f << "        - name: missing_var\n";
  f << "          possible_names: [does_not_exist]\n";
  f << "          optional: true\n";
  f.close();
}

// Helper: Create YAML config for static ADIOS2 stream
void create_static_adios2_yaml(const std::string& filename) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: adios2_static\n";
  f << "  substreams:\n";
  f << "    - name: static_data\n";
  f << "      format: adios2\n";
  f << "      static: true\n";
  f << "      filenames:\n";
  f << "        - test_static.bp\n";
  f << "      vars:\n";
  f << "        - name: coords\n";
  f << "          possible_names: [coordinates]\n";
  f.close();
}

// Helper: Create YAML with mixed static and time-varying streams
void create_mixed_yaml(const std::string& filename, int num_timesteps) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: mixed_stream\n";
  f << "  substreams:\n";
  f << "    - name: static_coords\n";
  f << "      format: adios2\n";
  f << "      static: true\n";
  f << "      filenames:\n";
  f << "        - test_static.bp\n";
  f << "      vars:\n";
  f << "        - name: coords\n";
  f << "          possible_names: [coordinates]\n";
  f << "    - name: time_varying\n";
  f << "      format: adios2\n";
  f << "      filenames:\n";
  for (int t = 0; t < num_timesteps; t++) {
    f << "        - test_stream_" << t << ".bp\n";
  }
  f << "      vars:\n";
  f << "        - name: temp\n";
  f << "          possible_names: [temperature]\n";
  f.close();
}

#endif // NDARRAY_HAVE_ADIOS2

int main(int argc, char** argv) {
#if NDARRAY_HAVE_YAML && NDARRAY_HAVE_ADIOS2

#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
#endif
    std::cout << "=== Running ADIOS2 YAML Stream Tests ===" << std::endl << std::endl;
#if NDARRAY_HAVE_MPI
  }
#endif

  const int NUM_TIMESTEPS = 5;

  // Setup: Create test BP files
  {
#if NDARRAY_HAVE_MPI
    if (rank == 0) {
#endif
      create_test_bp_files(NUM_TIMESTEPS);
      create_static_bp_file();
#if NDARRAY_HAVE_MPI
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  // Test 1: Basic ADIOS2 stream with time-varying data
  {
    TEST_SECTION("Time-varying ADIOS2 stream");

    create_adios2_yaml("test_adios2_stream.yaml", NUM_TIMESTEPS);

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_adios2_stream.yaml");

    TEST_ASSERT(stream.n_timesteps() == NUM_TIMESTEPS, 
                "Should have " + std::to_string(NUM_TIMESTEPS) + " timesteps");

    // Read first timestep
    auto g0 = stream.get_timestep(0);
    TEST_ASSERT(g0 != nullptr, "Timestep 0 should be valid");
    TEST_ASSERT(g0->has("temp"), "Should have 'temp' variable");
    TEST_ASSERT(g0->has("pres"), "Should have 'pres' variable");
    TEST_ASSERT(g0->has("vel"), "Should have 'vel' variable");

    auto temp = g0->get("temp");
    TEST_ASSERT(temp->nelem() == 20 * 30, "Temperature should have 600 elements");

    // Verify data from timestep 0
    auto temp_data = std::dynamic_pointer_cast<ftk::ndarray<float>>(temp);
    TEST_ASSERT(temp_data != nullptr, "Should cast to float array");
    float first_val = (*temp_data)[0];
    TEST_ASSERT(std::abs(first_val - 273.15f) < 0.01f, 
                "First value should be ~273.15");

    // Read last timestep
    auto g_last = stream.get_timestep(NUM_TIMESTEPS - 1);
    auto temp_last = std::dynamic_pointer_cast<ftk::ndarray<float>>(g_last->get("temp"));
    float last_first_val = (*temp_last)[0];
    TEST_ASSERT(last_first_val > first_val, 
                "Temperature should increase over time");

    std::cout << "    ✓ Read " << NUM_TIMESTEPS << " timesteps successfully" << std::endl;
  }

  // Test 2: Variable name aliasing
  {
    TEST_SECTION("Variable name aliasing");

    // Create YAML that looks for 'Temperature' but file has 'temperature'
    std::ofstream f("test_alias.yaml");
    f << "stream:\n";
    f << "  name: alias_test\n";
    f << "  substreams:\n";
    f << "    - name: bp_data\n";
    f << "      format: adios2\n";
    f << "      filenames:\n";
    f << "        - test_stream_0.bp\n";
    f << "      vars:\n";
    f << "        - name: t\n";
    f << "          possible_names: [Temperature, TEMP, temperature]\n";
    f.close();

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_alias.yaml");

    auto g = stream.get_timestep(0);
    TEST_ASSERT(g->has("t"), "Should find 't' via alias");

    std::cout << "    ✓ Variable aliasing works" << std::endl;
  }

  // Test 3: Optional variables
  {
    TEST_SECTION("Optional variables");

    create_adios2_yaml_optional("test_optional.yaml");

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_optional.yaml");

    auto g = stream.get_timestep(0);
    TEST_ASSERT(g->has("temp"), "Should have required 'temp' variable");
    TEST_ASSERT(!g->has("missing_var"), "Should not have optional missing variable");

    std::cout << "    ✓ Optional variables handled correctly" << std::endl;
  }

  // Test 4: Static ADIOS2 substream
  {
    TEST_SECTION("Static ADIOS2 substream");

    create_static_adios2_yaml("test_static_stream.yaml");

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_static_stream.yaml");

    // Static data should be available immediately
    stream.read_static();
    
    // Check that we can access static data at any timestep
    auto g0 = stream.get_timestep(0);
    TEST_ASSERT(g0->has("coords"), "Should have static 'coords'");
    
    auto coords = std::dynamic_pointer_cast<ftk::ndarray<float>>(g0->get("coords"));
    TEST_ASSERT(coords->nelem() == 40 * 40, "Coordinates should have 1600 elements");

    std::cout << "    ✓ Static substream works" << std::endl;
  }

  // Test 5: Mixed static and time-varying substreams
  {
    TEST_SECTION("Mixed static and time-varying substreams");

    create_mixed_yaml("test_mixed.yaml", NUM_TIMESTEPS);

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_mixed.yaml");
    stream.read_static();

    TEST_ASSERT(stream.n_timesteps() == NUM_TIMESTEPS, 
                "Should have time-varying timesteps");

    // Check that each timestep has both static and time-varying data
    for (int t = 0; t < NUM_TIMESTEPS; t++) {
      auto g = stream.get_timestep(t);
      TEST_ASSERT(g->has("coords"), "Should have static coords");
      TEST_ASSERT(g->has("temp"), "Should have time-varying temp");
    }

    // Verify static data is the same across timesteps
    auto g0 = stream.get_timestep(0);
    auto g1 = stream.get_timestep(1);
    
    auto coords0 = std::dynamic_pointer_cast<ftk::ndarray<float>>(g0->get("coords"));
    auto coords1 = std::dynamic_pointer_cast<ftk::ndarray<float>>(g1->get("coords"));
    
    TEST_ASSERT((*coords0)[0] == (*coords1)[0], 
                "Static data should be identical across timesteps");

    // Verify time-varying data changes
    auto temp0 = std::dynamic_pointer_cast<ftk::ndarray<float>>(g0->get("temp"));
    auto temp1 = std::dynamic_pointer_cast<ftk::ndarray<float>>(g1->get("temp"));
    
    TEST_ASSERT((*temp0)[0] != (*temp1)[0], 
                "Time-varying data should differ across timesteps");

    std::cout << "    ✓ Mixed substreams work correctly" << std::endl;
  }

  // Test 6: Multiple timestep iteration
  {
    TEST_SECTION("Timestep iteration");

    create_adios2_yaml("test_iteration.yaml", NUM_TIMESTEPS);

    ftk::stream<> stream;
    stream.set_input_source_yaml_file("test_iteration.yaml");

    // Iterate through all timesteps
    std::vector<float> first_temps;
    for (int t = 0; t < stream.n_timesteps(); t++) {
      auto g = stream.get_timestep(t);
      auto temp = std::dynamic_pointer_cast<ftk::ndarray<float>>(g->get("temp"));
      first_temps.push_back((*temp)[0]);
    }

    // Verify monotonic increase (our test data increases with time)
    for (size_t i = 1; i < first_temps.size(); i++) {
      TEST_ASSERT(first_temps[i] > first_temps[i-1], 
                  "Temperature should increase monotonically");
    }

    std::cout << "    ✓ Timestep iteration works" << std::endl;
  }

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  std::cout << std::endl << "=== All ADIOS2 YAML Stream Tests Passed ===" << std::endl;
  return 0;

#else
  std::cout << "ADIOS2 and/or YAML not available - tests skipped" << std::endl;
  std::cout << "Build with -DNDARRAY_USE_ADIOS2=TRUE -DNDARRAY_USE_YAML=TRUE to enable" << std::endl;
  return 0;
#endif
}
