/**
 * Test HDF5 exception handling
 *
 * Verifies that HDF5 functions throw appropriate exceptions instead of
 * returning false or calling fatal().
 */

#include <ndarray/ndarray.hh>
#include <ndarray/error.hh>
#include <iostream>
#include <cassert>

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "=== Testing HDF5 Exception Handling ===" << std::endl << std::endl;

#if NDARRAY_HAVE_HDF5

  // Test 1: read_h5() with non-existent file throws exception
  {
    std::cout << "Test 1: read_h5() with non-existent file" << std::endl;
    ftk::ndarray<float> arr;

    try {
      arr.read_h5("nonexistent_file_12345.h5", "data");
      std::cerr << "  FAILED: Expected hdf5_error to be thrown" << std::endl;
      return 1;
    } catch (const ftk::hdf5_error& e) {
      std::cout << "  PASSED: Caught hdf5_error as expected" << std::endl;
      std::cout << "    Message: " << e.what() << std::endl;
      std::cout << "    Error code: " << e.error_code() << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "  FAILED: Caught wrong exception type: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 2: Successful write and read
  {
    std::cout << "\nTest 2: Successful write and read" << std::endl;

    ftk::ndarray<double> original({5, 5});
    for (size_t i = 0; i < original.size(); i++) {
      original[i] = static_cast<double>(i * 2);
    }

    try {
      // Write
      hid_t file_id = H5Fcreate("test_exceptions.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      if (file_id < 0) {
        std::cerr << "  FAILED: Could not create test file" << std::endl;
        return 1;
      }

      hsize_t dims[2] = {5, 5};
      hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
      hid_t dataset_id = H5Dcreate2(file_id, "data", H5T_NATIVE_DOUBLE, dataspace_id,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      // Write data
      H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, original.data());

      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);

      // Read back
      ftk::ndarray<double> loaded;
      loaded.read_h5("test_exceptions.h5", "data");  // Should not throw

      // Verify
      if (loaded.size() != original.size()) {
        std::cerr << "  FAILED: Size mismatch" << std::endl;
        return 1;
      }

      std::cout << "  PASSED: Write/read succeeded without exceptions" << std::endl;

    } catch (const std::exception& e) {
      std::cerr << "  FAILED: Unexpected exception: " << e.what() << std::endl;
      return 1;
    }
  }

  // Test 3: read_h5() with valid file but invalid dataset
  {
    std::cout << "\nTest 3: read_h5() with invalid dataset name" << std::endl;

    ftk::ndarray<double> arr;
    try {
      arr.read_h5("test_exceptions.h5", "nonexistent_dataset");
      std::cerr << "  FAILED: Expected hdf5_error to be thrown" << std::endl;
      return 1;
    } catch (const ftk::hdf5_error& e) {
      std::cout << "  PASSED: Caught hdf5_error as expected" << std::endl;
      std::cout << "    Message: " << e.what() << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "  FAILED: Caught wrong exception type: " << e.what() << std::endl;
      return 1;
    }
  }

  std::cout << "\n=== All HDF5 exception tests passed ===" << std::endl;

#else
  std::cout << "SKIPPED: HDF5 not available" << std::endl;
#endif

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
