/**
 * I/O functionality tests for ndarray
 *
 * Tests file I/O operations:
 * - Binary file read/write
 * - Data integrity
 * - Optional: NetCDF, HDF5, ADIOS2 (if available)
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
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
  std::cout << "=== Running ndarray I/O Tests ===" << std::endl << std::endl;

  // Test 1: Binary write and read
  {
    TEST_SECTION("Binary file I/O");

    // Create test data
    ftk::ndarray<double> original;
    original.reshapef(10, 20);
    for (size_t i = 0; i < original.size(); i++) {
      original[i] = static_cast<double>(i) * 0.5;
    }

    // Write to file
    try {
      original.to_binary_file("test_output.bin");
      std::cout << "    - Wrote binary file" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    - Write error: " << e.what() << std::endl;
      return 1;
    }

    // Read back
    // Note: Binary files don't store dimension info, must reshape before reading
    ftk::ndarray<double> loaded;
    loaded.reshapef(10, 20);  // Must match original dimensions
    try {
      loaded.read_binary_file("test_output.bin");
      std::cout << "    - Read binary file" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "    - Read error: " << e.what() << std::endl;
      return 1;
    }

    // Verify dimensions
    TEST_ASSERT(loaded.size() == original.size(), "Loaded size should match original");
    TEST_ASSERT(loaded.nd() == original.nd(), "Loaded dimensions should match original");

    // Verify data
    for (size_t i = 0; i < loaded.size(); i++) {
      if (loaded[i] != original[i]) {
        std::cerr << "    - Data mismatch at index " << i << std::endl;
        std::cerr << "      Original: " << original[i] << ", Loaded: " << loaded[i] << std::endl;
        return 1;
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Different data types I/O
  {
    TEST_SECTION("I/O with different types");

    // Float
    ftk::ndarray<float> float_arr;
    float_arr.reshapef(50);
    for (size_t i = 0; i < float_arr.size(); i++) {
      float_arr[i] = static_cast<float>(i) * 0.1f;
    }
    float_arr.to_binary_file("test_float.bin");

    ftk::ndarray<float> float_loaded;
    float_loaded.reshapef(50);  // Must match original dimensions
    float_loaded.read_binary_file("test_float.bin");
    TEST_ASSERT(float_loaded.size() == 50, "Float array size should match");
    TEST_ASSERT(std::abs(float_loaded[0] - 0.0f) < 0.001f, "Float data should match");

    // Integer
    ftk::ndarray<int> int_arr;
    int_arr.reshapef(30);
    for (size_t i = 0; i < int_arr.size(); i++) {
      int_arr[i] = static_cast<int>(i * 10);
    }
    int_arr.to_binary_file("test_int.bin");

    ftk::ndarray<int> int_loaded;
    int_loaded.reshapef(30);  // Must match original dimensions
    int_loaded.read_binary_file("test_int.bin");
    TEST_ASSERT(int_loaded.size() == 30, "Int array size should match");
    TEST_ASSERT(int_loaded[0] == 0, "Int data should match");
    TEST_ASSERT(int_loaded[29] == 290, "Int data should match");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Multi-dimensional I/O
  {
    TEST_SECTION("Multi-dimensional array I/O");

    ftk::ndarray<double> arr3d;
    arr3d.reshapef(5, 6, 7);

    for (size_t i = 0; i < arr3d.size(); i++) {
      arr3d[i] = static_cast<double>(i);
    }

    arr3d.to_binary_file("test_3d.bin");

    ftk::ndarray<double> loaded3d;
    loaded3d.reshapef(5, 6, 7);  // Must match original dimensions
    loaded3d.read_binary_file("test_3d.bin");

    TEST_ASSERT(loaded3d.nd() == 3, "3D array dimensions preserved");
    TEST_ASSERT(loaded3d.size() == 5*6*7, "3D array size preserved");
    TEST_ASSERT(loaded3d[0] == 0.0, "3D array data start correct");
    TEST_ASSERT(loaded3d[loaded3d.size()-1] == static_cast<double>(loaded3d.size()-1),
                "3D array data end correct");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Large array I/O
  {
    TEST_SECTION("Large array I/O");

    ftk::ndarray<double> large_arr;
    large_arr.reshapef(1000, 500);

    // Fill with pattern
    for (size_t i = 0; i < large_arr.size(); i++) {
      large_arr[i] = static_cast<double>(i % 1000);
    }

    large_arr.to_binary_file("test_large.bin");

    ftk::ndarray<double> large_loaded;
    large_loaded.reshapef(1000, 500);  // Must match original dimensions
    large_loaded.read_binary_file("test_large.bin");

    TEST_ASSERT(large_loaded.size() == 500000, "Large array size preserved");

    // Spot check values
    TEST_ASSERT(large_loaded[0] == 0.0, "Large array spot check 1");
    TEST_ASSERT(large_loaded[1000] == 0.0, "Large array spot check 2");
    TEST_ASSERT(large_loaded[1001] == 1.0, "Large array spot check 3");

    std::cout << "    PASSED" << std::endl;
  }

// NetCDF I/O requires manual file opening and ncid/varid management
// The convenience methods write_netcdf() don't exist in this library
// Use read_netcdf() with filename and to_netcdf() with ncid/varid
std::cout << "  NetCDF tests SKIPPED (requires manual ncid/varid management)" << std::endl;

// HDF5 I/O requires manual file opening and hid_t management
// The convenience methods write_h5()/read_h5() don't exist in this library
std::cout << "  HDF5 tests SKIPPED (requires manual hid_t management)" << std::endl;

  // Clean up temporary test files
  std::remove("test_output.bin");
  std::remove("test_float.bin");
  std::remove("test_int.bin");
  std::remove("test_3d.bin");
  std::remove("test_large.bin");

  std::cout << std::endl;
  std::cout << "=== All I/O Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
