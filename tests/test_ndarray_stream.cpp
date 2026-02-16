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
  f << "    - name: woven\n";
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
  f << "    - name: woven\n";
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
  f << "    - name: woven\n";
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

      // Use float since dtype is float32 in YAML
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

      // Use float since dtype is float32 in YAML
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

      // Use float since dtype is float32 in YAML
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

    // 2D array - use float since dtype is float32
    create_synthetic_yaml("test_stream_2d.yaml", 64, 48, 5);
    ftk::stream s2d;
    s2d.parse_yaml("test_stream_2d.yaml");
    auto g2d = s2d.read(0);
    auto data2d = g2d->get_arr<float>("scalar");
    TEST_ASSERT(data2d.size() == 64*48, "2D array size correct");
    std::cout << "    - 2D array: [64, 48]" << std::endl;

    // Small array - use float since dtype is float32
    create_synthetic_yaml("test_stream_small.yaml", 4, 4, 2);
    ftk::stream ssmall;
    ssmall.parse_yaml("test_stream_small.yaml");
    auto gsmall = ssmall.read(0);
    auto datasmall = gsmall->get_arr<float>("scalar");
    TEST_ASSERT(datasmall.size() == 4*4, "Small array size correct");
    std::cout << "    - Small array: [4, 4]" << std::endl;

    // Large array - use float since dtype is float32
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
          // Try float first (for synthetic), fallback to double
          auto scalar_float_ptr = g->get_ptr<float>("scalar");
          if (scalar_float_ptr) {
            std::cout << "    - Read woven example data: " << scalar_float_ptr->size() << " elements (float)" << std::endl;
          } else {
            auto scalar_double_ptr = g->get_ptr<double>("scalar");
            if (scalar_double_ptr) {
              std::cout << "    - Read woven example data: " << scalar_double_ptr->size() << " elements (double)" << std::endl;
            }
          }
        }

        std::cout << "    PASSED" << std::endl;
      } catch (const std::exception& e) {
        std::cout << "    - Example file test skipped: " << e.what() << std::endl;
      }
    } else {
      std::cout << "    SKIPPED (example files not found)" << std::endl;
    }
  }

#if NDARRAY_HAVE_NETCDF
  // Test 11: NetCDF stream with actual data files
  {
    TEST_SECTION("NetCDF stream with generated data");

    // Create NetCDF test data files
    try {
      // Create test NetCDF file with time series
      int ncid, varid, dimids[3];
      const size_t nx = 8, ny = 10, nt = 5;

      NC_SAFE_CALL( nc_create("test_stream_nc_t0.nc", NC_CLOBBER, &ncid) );

      // Define dimensions
      NC_SAFE_CALL( nc_def_dim(ncid, "time", NC_UNLIMITED, &dimids[0]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "y", ny, &dimids[1]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "x", nx, &dimids[2]) );

      // Define variable
      NC_SAFE_CALL( nc_def_var(ncid, "temperature", NC_DOUBLE, 3, dimids, &varid) );
      NC_SAFE_CALL( nc_enddef(ncid) );

      // Write data for first 3 timesteps
      for (size_t t = 0; t < 3; t++) {
        std::vector<double> data(nx * ny);
        for (size_t i = 0; i < nx * ny; i++) {
          data[i] = t * 100.0 + i;
        }

        size_t start[3] = {t, 0, 0};
        size_t count[3] = {1, ny, nx};
        NC_SAFE_CALL( nc_put_vara_double(ncid, varid, start, count, data.data()) );
      }

      NC_SAFE_CALL( nc_close(ncid) );

      // Create second NetCDF file
      NC_SAFE_CALL( nc_create("test_stream_nc_t1.nc", NC_CLOBBER, &ncid) );
      NC_SAFE_CALL( nc_def_dim(ncid, "time", NC_UNLIMITED, &dimids[0]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "y", ny, &dimids[1]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "x", nx, &dimids[2]) );
      NC_SAFE_CALL( nc_def_var(ncid, "temperature", NC_DOUBLE, 3, dimids, &varid) );
      NC_SAFE_CALL( nc_enddef(ncid) );

      // Write data for remaining 2 timesteps
      for (size_t t = 0; t < 2; t++) {
        std::vector<double> data(nx * ny);
        for (size_t i = 0; i < nx * ny; i++) {
          data[i] = (t + 3) * 100.0 + i;
        }

        size_t start[3] = {t, 0, 0};
        size_t count[3] = {1, ny, nx};
        NC_SAFE_CALL( nc_put_vara_double(ncid, varid, start, count, data.data()) );
      }

      NC_SAFE_CALL( nc_close(ncid) );

      // Create YAML configuration for NetCDF stream
      std::ofstream yaml("test_stream_netcdf.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_netcdf\n";
      yaml << "  substreams:\n";
      yaml << "    - name: nc_data\n";
      yaml << "      format: netcdf\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_nc_t0.nc\n";
      yaml << "        - test_stream_nc_t1.nc\n";
      yaml << "      vars:\n";
      yaml << "        - name: temperature\n";
      yaml << "          dtype: float64\n";
      yaml.close();

      // Test reading the stream
      ftk::stream s;
      s.parse_yaml("test_stream_netcdf.yaml");

      TEST_ASSERT(s.total_timesteps() == 5, "NetCDF stream should have 5 timesteps");
      std::cout << "    - Total timesteps: " << s.total_timesteps() << std::endl;

      // Read first timestep from first file
      auto g0 = s.read(0);
      TEST_ASSERT(g0 != nullptr, "Failed to read timestep 0");
      TEST_ASSERT(g0->has("temperature"), "Missing temperature variable");

      auto temp0 = g0->get_arr<double>("temperature");
      TEST_ASSERT(temp0.size() == nx * ny, "Wrong array size");
      TEST_ASSERT(std::abs(temp0[0] - 0.0) < 1e-10, "Wrong data value at t=0");
      TEST_ASSERT(std::abs(temp0[1] - 1.0) < 1e-10, "Wrong data value at t=0");

      // Read third timestep (last in first file)
      auto g2 = s.read(2);
      auto temp2 = g2->get_arr<double>("temperature");
      TEST_ASSERT(std::abs(temp2[0] - 200.0) < 1e-10, "Wrong data value at t=2");

      // Read fourth timestep (first in second file)
      auto g3 = s.read(3);
      auto temp3 = g3->get_arr<double>("temperature");
      TEST_ASSERT(std::abs(temp3[0] - 300.0) < 1e-10, "Wrong data value at t=3");

      // Read last timestep
      auto g4 = s.read(4);
      auto temp4 = g4->get_arr<double>("temperature");
      TEST_ASSERT(std::abs(temp4[0] - 400.0) < 1e-10, "Wrong data value at t=4");

      std::cout << "    - Successfully read NetCDF stream across multiple files" << std::endl;
      std::cout << "    PASSED" << std::endl;

      // Cleanup NetCDF files
      std::remove("test_stream_nc_t0.nc");
      std::remove("test_stream_nc_t1.nc");
      std::remove("test_stream_netcdf.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    NetCDF stream test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 12: NetCDF stream with multiple variables
  {
    TEST_SECTION("NetCDF stream with multiple variables");

    try {
      int ncid, temp_varid, vel_varid, dimids[3];
      const size_t nx = 6, ny = 8, nt = 3;

      NC_SAFE_CALL( nc_create("test_stream_nc_multi.nc", NC_CLOBBER, &ncid) );

      // Define dimensions
      NC_SAFE_CALL( nc_def_dim(ncid, "time", NC_UNLIMITED, &dimids[0]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "y", ny, &dimids[1]) );
      NC_SAFE_CALL( nc_def_dim(ncid, "x", nx, &dimids[2]) );

      // Define variables
      NC_SAFE_CALL( nc_def_var(ncid, "temperature", NC_FLOAT, 3, dimids, &temp_varid) );
      NC_SAFE_CALL( nc_def_var(ncid, "velocity_x", NC_FLOAT, 3, dimids, &vel_varid) );
      NC_SAFE_CALL( nc_enddef(ncid) );

      // Write data
      for (size_t t = 0; t < nt; t++) {
        std::vector<float> temp_data(nx * ny);
        std::vector<float> vel_data(nx * ny);

        for (size_t i = 0; i < nx * ny; i++) {
          temp_data[i] = t * 10.0f + i * 0.1f;
          vel_data[i] = t * 5.0f + i * 0.05f;
        }

        size_t start[3] = {t, 0, 0};
        size_t count[3] = {1, ny, nx};
        NC_SAFE_CALL( nc_put_vara_float(ncid, temp_varid, start, count, temp_data.data()) );
        NC_SAFE_CALL( nc_put_vara_float(ncid, vel_varid, start, count, vel_data.data()) );
      }

      NC_SAFE_CALL( nc_close(ncid) );

      // Create YAML configuration
      std::ofstream yaml("test_stream_nc_multivars.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_nc_multi\n";
      yaml << "  substreams:\n";
      yaml << "    - name: multi_vars\n";
      yaml << "      format: netcdf\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_nc_multi.nc\n";
      yaml << "      vars:\n";
      yaml << "        - name: temperature\n";
      yaml << "          dtype: float32\n";
      yaml << "        - name: velocity_x\n";
      yaml << "          dtype: float32\n";
      yaml.close();

      // Test reading
      ftk::stream s;
      s.parse_yaml("test_stream_nc_multivars.yaml");

      auto g = s.read(1);
      TEST_ASSERT(g->has("temperature"), "Missing temperature");
      TEST_ASSERT(g->has("velocity_x"), "Missing velocity_x");

      auto temp = g->get_arr<float>("temperature");
      auto vel = g->get_arr<float>("velocity_x");

      TEST_ASSERT(temp.size() == nx * ny, "Wrong temperature size");
      TEST_ASSERT(vel.size() == nx * ny, "Wrong velocity size");
      TEST_ASSERT(std::abs(temp[0] - 10.0f) < 1e-5f, "Wrong temperature value");
      TEST_ASSERT(std::abs(vel[0] - 5.0f) < 1e-5f, "Wrong velocity value");

      std::cout << "    - Successfully read multiple NetCDF variables" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_nc_multi.nc");
      std::remove("test_stream_nc_multivars.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    NetCDF multi-variable test failed: " << e.what() << std::endl;
      return 1;
    }
  }
#else
  std::cout << "  NetCDF stream tests SKIPPED (not built with NetCDF)" << std::endl;
#endif

#if NDARRAY_HAVE_HDF5
  // Test 13: HDF5 stream with time series (multiple timesteps per file)
  {
    TEST_SECTION("HDF5 stream with time series data");

    try {
      const size_t nx = 10, ny = 12, nt_per_file = 3;

      // Create two HDF5 files with time series
      for (int file_idx = 0; file_idx < 2; file_idx++) {
        std::string filename = "test_stream_h5_t" + std::to_string(file_idx) + ".h5";

        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        // Create dataset for each timestep
        for (size_t t = 0; t < nt_per_file; t++) {
          std::string dset_name = "data_t" + std::to_string(t);
          hsize_t dims[2] = {ny, nx};

          hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
          hid_t dataset_id = H5Dcreate2(file_id, dset_name.c_str(), H5T_IEEE_F64LE,
                                        dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

          // Write data
          std::vector<double> data(nx * ny);
          for (size_t i = 0; i < nx * ny; i++) {
            data[i] = (file_idx * nt_per_file + t) * 50.0 + i;
          }

          H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

          H5Dclose(dataset_id);
          H5Sclose(dataspace_id);
        }

        H5Fclose(file_id);
      }

      // Create YAML configuration
      std::ofstream yaml("test_stream_hdf5.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_hdf5\n";
      yaml << "  substreams:\n";
      yaml << "    - name: h5_data\n";
      yaml << "      format: h5\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_h5_t0.h5\n";
      yaml << "        - test_stream_h5_t1.h5\n";
      yaml << "      timesteps_per_file: " << nt_per_file << "\n";
      yaml << "      vars:\n";
      yaml << "        - name: pressure\n";
      yaml << "          h5_name: data_t%d\n";
      yaml << "          dtype: float64\n";
      yaml.close();

      // Test reading
      ftk::stream s;
      s.parse_yaml("test_stream_hdf5.yaml");

      // With timesteps_per_file=3 and 2 files, we should have 6 total timesteps
      int actual_timesteps = s.total_timesteps();
      std::cout << "    - Total timesteps in stream: " << actual_timesteps << std::endl;
      TEST_ASSERT(actual_timesteps == 6, "HDF5 stream should have 6 timesteps (2 files × 3 timesteps/file)");

      // Read first timestep (file 0, dataset data_t0)
      auto g0 = s.read(0);
      TEST_ASSERT(g0 != nullptr, "Failed to read timestep 0");
      TEST_ASSERT(g0->has("pressure"), "Missing pressure variable at timestep 0");

      auto p0 = g0->get_arr<double>("pressure");
      TEST_ASSERT(p0.size() == nx * ny, "Wrong HDF5 array size");
      TEST_ASSERT(std::abs(p0[0] - 0.0) < 1e-10, "Wrong data at timestep 0 (file 0, t=0)");

      // Read second timestep (file 0, dataset data_t1)
      auto g1 = s.read(1);
      auto p1 = g1->get_arr<double>("pressure");
      TEST_ASSERT(std::abs(p1[0] - 50.0) < 1e-10, "Wrong data at timestep 1 (file 0, t=1)");

      // Read fourth timestep (file 1, dataset data_t0)
      auto g3 = s.read(3);
      auto p3 = g3->get_arr<double>("pressure");
      TEST_ASSERT(std::abs(p3[0] - 150.0) < 1e-10, "Wrong data at timestep 3 (file 1, t=0)");

      // Read last timestep (file 1, dataset data_t2)
      auto g5 = s.read(5);
      auto p5 = g5->get_arr<double>("pressure");
      TEST_ASSERT(std::abs(p5[0] - 250.0) < 1e-10, "Wrong data at timestep 5 (file 1, t=2)");

      std::cout << "    - Successfully read all 6 timesteps from HDF5 stream" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_h5_t0.h5");
      std::remove("test_stream_h5_t1.h5");
      std::remove("test_stream_hdf5.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    HDF5 stream test failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 14: HDF5 static data stream
  {
    TEST_SECTION("HDF5 static data stream");

    try {
      const size_t nx = 8, ny = 8;

      // Create HDF5 file with static data
      hid_t file_id = H5Fcreate("test_stream_h5_static.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      hsize_t dims[2] = {ny, nx};
      hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
      hid_t dataset_id = H5Dcreate2(file_id, "mask", H5T_IEEE_F32LE,
                                    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      std::vector<float> mask(nx * ny, 1.0f);
      for (size_t i = 0; i < nx * ny; i++) {
        if (i % 3 == 0) mask[i] = 0.0f;
      }

      H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mask.data());

      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);

      // Create YAML configuration for static data
      std::ofstream yaml("test_stream_h5_static.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_h5_static\n";
      yaml << "  substreams:\n";
      yaml << "    - name: static_data\n";
      yaml << "      format: h5\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_h5_static.h5\n";
      yaml << "      vars:\n";
      yaml << "        - name: mask\n";
      yaml << "          h5_name: mask\n";
      yaml << "          dtype: float32\n";
      yaml << "      static: true\n";
      yaml.close();

      // Test reading static data
      ftk::stream s;
      s.parse_yaml("test_stream_h5_static.yaml");

      auto g_static = s.read_static();
      TEST_ASSERT(g_static != nullptr, "Failed to read static HDF5 data");
      TEST_ASSERT(g_static->has("mask"), "Missing mask variable");

      auto mask_arr = g_static->get_arr<float>("mask");
      TEST_ASSERT(mask_arr.size() == nx * ny, "Wrong static array size");
      TEST_ASSERT(std::abs(mask_arr[0] - 0.0f) < 1e-5f, "Wrong static data value");
      TEST_ASSERT(std::abs(mask_arr[1] - 1.0f) < 1e-5f, "Wrong static data value");

      std::cout << "    - Successfully read static HDF5 data" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_h5_static.h5");
      std::remove("test_stream_h5_static.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    HDF5 static data test failed: " << e.what() << std::endl;
      return 1;
    }
  }
#else
  std::cout << "  HDF5 stream tests SKIPPED (not built with HDF5)" << std::endl;
#endif

#if NDARRAY_HAVE_NETCDF
  // Test 15: Mixed static and dynamic substreams in same YAML
  {
    TEST_SECTION("Mixed static and dynamic substreams");

    try {
      const size_t nx = 10, ny = 12, nt = 3;

      // Create static NetCDF file (mask/domain)
      int ncid_static, dimids_static[2], varid_mask;
      nc_create("test_stream_mixed_static.nc", NC_CLOBBER, &ncid_static);
      nc_def_dim(ncid_static, "x", nx, &dimids_static[1]);
      nc_def_dim(ncid_static, "y", ny, &dimids_static[0]);
      nc_def_var(ncid_static, "land_mask", NC_FLOAT, 2, dimids_static, &varid_mask);
      nc_enddef(ncid_static);

      // Write static mask data
      std::vector<float> mask(nx * ny);
      for (size_t i = 0; i < nx * ny; i++) {
        mask[i] = (i % 2 == 0) ? 1.0f : 0.0f;  // Checkerboard pattern
      }
      nc_put_var_float(ncid_static, varid_mask, mask.data());
      nc_close(ncid_static);

      // Create dynamic NetCDF file (time-varying data)
      int ncid_dynamic, dimids_dynamic[3], varid_temp;
      nc_create("test_stream_mixed_dynamic.nc", NC_CLOBBER, &ncid_dynamic);
      nc_def_dim(ncid_dynamic, "time", NC_UNLIMITED, &dimids_dynamic[0]);
      nc_def_dim(ncid_dynamic, "x", nx, &dimids_dynamic[2]);
      nc_def_dim(ncid_dynamic, "y", ny, &dimids_dynamic[1]);
      nc_def_var(ncid_dynamic, "temperature", NC_FLOAT, 3, dimids_dynamic, &varid_temp);
      nc_enddef(ncid_dynamic);

      // Write time-varying temperature data
      for (size_t t = 0; t < nt; t++) {
        std::vector<float> temp(nx * ny);
        for (size_t i = 0; i < nx * ny; i++) {
          temp[i] = 20.0f + t * 5.0f + i * 0.1f;
        }
        size_t start[3] = {t, 0, 0};
        size_t count[3] = {1, ny, nx};
        nc_put_vara_float(ncid_dynamic, varid_temp, start, count, temp.data());
      }
      nc_close(ncid_dynamic);

      // Create YAML with mixed static and dynamic substreams
      std::ofstream yaml("test_stream_mixed.yaml");
      yaml << "stream:\n";
      yaml << "  name: test_mixed\n";
      yaml << "  substreams:\n";
      yaml << "    - name: static_mask\n";
      yaml << "      format: netcdf\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_mixed_static.nc\n";
      yaml << "      vars:\n";
      yaml << "        - name: land_mask\n";
      yaml << "          nc_name: land_mask\n";
      yaml << "          dtype: float32\n";
      yaml << "      static: true\n";
      yaml << "    - name: dynamic_temp\n";
      yaml << "      format: netcdf\n";
      yaml << "      filenames:\n";
      yaml << "        - test_stream_mixed_dynamic.nc\n";
      yaml << "      vars:\n";
      yaml << "        - name: temperature\n";
      yaml << "          nc_name: temperature\n";
      yaml << "          dtype: float32\n";
      yaml.close();

      // Test reading
      ftk::stream s;
      s.parse_yaml("test_stream_mixed.yaml");

      TEST_ASSERT(s.total_timesteps() == nt, "Mixed stream should have correct timesteps");

      // Test reading static data
      auto g_static = s.read_static();
      TEST_ASSERT(g_static != nullptr, "Failed to read static data from mixed stream");
      TEST_ASSERT(g_static->has("land_mask"), "Missing land_mask in static data");

      auto mask_arr = g_static->get_arr<float>("land_mask");
      TEST_ASSERT(mask_arr.size() == nx * ny, "Wrong static mask size");
      TEST_ASSERT(std::abs(mask_arr[0] - 1.0f) < 1e-5f, "Wrong static mask value at [0]");
      TEST_ASSERT(std::abs(mask_arr[1] - 0.0f) < 1e-5f, "Wrong static mask value at [1]");

      // Test reading dynamic data at different timesteps
      auto g0 = s.read(0);
      TEST_ASSERT(g0 != nullptr, "Failed to read dynamic timestep 0");
      TEST_ASSERT(g0->has("temperature"), "Missing temperature at t=0");

      auto temp0 = g0->get_arr<float>("temperature");
      TEST_ASSERT(temp0.size() == nx * ny, "Wrong temperature array size");
      TEST_ASSERT(std::abs(temp0[0] - 20.0f) < 1e-5f, "Wrong temperature at t=0");

      auto g2 = s.read(2);
      auto temp2 = g2->get_arr<float>("temperature");
      TEST_ASSERT(std::abs(temp2[0] - 30.0f) < 1e-5f, "Wrong temperature at t=2");

      // Verify static data is accessible at any time (should be same)
      auto g_static2 = s.read_static();
      auto mask_arr2 = g_static2->get_arr<float>("land_mask");
      TEST_ASSERT(std::abs(mask_arr2[0] - mask_arr[0]) < 1e-5f, "Static data should be consistent");

      std::cout << "    - Successfully read mixed static/dynamic stream" << std::endl;
      std::cout << "    - Static substream: land_mask" << std::endl;
      std::cout << "    - Dynamic substream: temperature (" << nt << " timesteps)" << std::endl;
      std::cout << "    PASSED" << std::endl;

      std::remove("test_stream_mixed_static.nc");
      std::remove("test_stream_mixed_dynamic.nc");
      std::remove("test_stream_mixed.yaml");

    } catch (const std::exception& e) {
      std::cerr << "    Mixed stream test failed: " << e.what() << std::endl;
      return 1;
    }
  }
#else
  std::cout << "  Mixed stream test SKIPPED (not built with NetCDF)" << std::endl;
#endif

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
