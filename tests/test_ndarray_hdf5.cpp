/**
 * HDF5 functionality tests for ndarray
 *
 * Tests HDF5 file I/O operations:
 * - Writing arrays to HDF5 files
 * - Reading arrays from HDF5 files
 * - Different data types
 * - Multi-dimensional arrays
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>

#if NDARRAY_HAVE_HDF5
#include <hdf5.h>
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

int main() {
  std::cout << "=== Running ndarray HDF5 Tests ===" << std::endl << std::endl;

#if NDARRAY_HAVE_HDF5

  // Test 1: Basic HDF5 write and read
  {
    TEST_SECTION("Basic HDF5 write/read");

    ftk::ndarray<double> original;
    original.reshapef(10, 20);
    for (size_t i = 0; i < original.size(); i++) {
      original[i] = static_cast<double>(i) * 0.5;
    }

    // Write to HDF5 file
    hid_t file_id = H5Fcreate("test_hdf5_output.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    TEST_ASSERT(file_id >= 0, "Failed to create HDF5 file");

    // Create dataspace
    hsize_t dims[2] = {10, 20};
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    TEST_ASSERT(dataspace_id >= 0, "Failed to create dataspace");

    // Create dataset
    hid_t dataset_id = H5Dcreate2(file_id, "data", H5T_NATIVE_DOUBLE, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    TEST_ASSERT(dataset_id >= 0, "Failed to create dataset");

    // Write data using HDF5 API
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, original.data());

    // Close dataset and file
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    std::cout << "    - Wrote HDF5 file" << std::endl;

    // Read back - read_h5_did() automatically reshapes based on dataset dimensions
    ftk::ndarray<double> loaded;

    file_id = H5Fopen("test_hdf5_output.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    TEST_ASSERT(file_id >= 0, "Failed to open HDF5 file");

    dataset_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    TEST_ASSERT(dataset_id >= 0, "Failed to open dataset");

    loaded.read_h5_did(dataset_id);

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    std::cout << "    - Read HDF5 file" << std::endl;

    // Verify data
    TEST_ASSERT(loaded.size() == original.size(), "Loaded size should match");
    for (size_t i = 0; i < loaded.size(); i++) {
      if (std::abs(loaded[i] - original[i]) > 1e-10) {
        std::cerr << "    - Data mismatch at index " << i << std::endl;
        return 1;
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Different data types
  {
    TEST_SECTION("HDF5 with different data types");

    // Float array
    ftk::ndarray<float> float_arr;
    float_arr.reshapef(30);
    for (size_t i = 0; i < float_arr.size(); i++) {
      float_arr[i] = static_cast<float>(i) * 0.1f;
    }

    hid_t file_id = H5Fcreate("test_hdf5_float.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[1] = {30};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "float_data", H5T_NATIVE_FLOAT, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, float_arr.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    // Read back
    ftk::ndarray<float> float_loaded;

    file_id = H5Fopen("test_hdf5_float.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "float_data", H5P_DEFAULT);
    float_loaded.read_h5_did(dataset_id);

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    TEST_ASSERT(float_loaded.size() == 30, "Float array size should match");
    TEST_ASSERT(std::abs(float_loaded[0] - 0.0f) < 0.001f, "Float data should match");
    TEST_ASSERT(std::abs(float_loaded[29] - 2.9f) < 0.001f, "Float data should match");

    // Integer array
    ftk::ndarray<int> int_arr;
    int_arr.reshapef(20);
    for (size_t i = 0; i < int_arr.size(); i++) {
      int_arr[i] = static_cast<int>(i * 10);
    }

    file_id = H5Fcreate("test_hdf5_int.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dims[0] = 20;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "int_data", H5T_NATIVE_INT, dataspace_id,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, int_arr.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    // Read back
    ftk::ndarray<int> int_loaded;

    file_id = H5Fopen("test_hdf5_int.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "int_data", H5P_DEFAULT);
    int_loaded.read_h5_did(dataset_id);

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    TEST_ASSERT(int_loaded.size() == 20, "Int array size should match");
    TEST_ASSERT(int_loaded[0] == 0, "Int data should match");
    TEST_ASSERT(int_loaded[19] == 190, "Int data should match");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Multi-dimensional arrays
  {
    TEST_SECTION("Multi-dimensional HDF5 I/O");

    ftk::ndarray<double> arr3d;
    arr3d.reshapef(4, 5, 6);

    for (size_t i = 0; i < arr3d.size(); i++) {
      arr3d[i] = static_cast<double>(i);
    }

    hid_t file_id = H5Fcreate("test_hdf5_3d.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {4, 5, 6};
    hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "data_3d", H5T_NATIVE_DOUBLE, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr3d.data());

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    // Read back
    ftk::ndarray<double> loaded3d;

    file_id = H5Fopen("test_hdf5_3d.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen2(file_id, "data_3d", H5P_DEFAULT);
    loaded3d.read_h5_did(dataset_id);

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    TEST_ASSERT(loaded3d.nd() == 3, "3D array dimensions preserved");
    TEST_ASSERT(loaded3d.size() == 4*5*6, "3D array size preserved");
    TEST_ASSERT(loaded3d[0] == 0.0, "3D array data start correct");
    TEST_ASSERT(loaded3d[loaded3d.size()-1] == static_cast<double>(loaded3d.size()-1),
                "3D array data end correct");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Multiple datasets in one file
  {
    TEST_SECTION("Multiple datasets in one HDF5 file");

    ftk::ndarray<double> arr1;
    ftk::ndarray<double> arr2;
    arr1.reshapef(10);
    arr2.reshapef(20);

    for (size_t i = 0; i < arr1.size(); i++) arr1[i] = static_cast<double>(i);
    for (size_t i = 0; i < arr2.size(); i++) arr2[i] = static_cast<double>(i) * 2.0;

    // Write both datasets
    hid_t file_id = H5Fcreate("test_hdf5_multi.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t dims1[1] = {10};
    hid_t dataspace1_id = H5Screate_simple(1, dims1, NULL);
    hid_t dataset1_id = H5Dcreate2(file_id, "dataset1", H5T_NATIVE_DOUBLE, dataspace1_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset1_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr1.data());
    H5Dclose(dataset1_id);
    H5Sclose(dataspace1_id);

    hsize_t dims2[1] = {20};
    hid_t dataspace2_id = H5Screate_simple(1, dims2, NULL);
    hid_t dataset2_id = H5Dcreate2(file_id, "dataset2", H5T_NATIVE_DOUBLE, dataspace2_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset2_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, arr2.data());
    H5Dclose(dataset2_id);
    H5Sclose(dataspace2_id);

    H5Fclose(file_id);

    // Read both back
    ftk::ndarray<double> loaded1, loaded2;

    file_id = H5Fopen("test_hdf5_multi.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

    dataset1_id = H5Dopen2(file_id, "dataset1", H5P_DEFAULT);
    loaded1.read_h5_did(dataset1_id);
    H5Dclose(dataset1_id);

    dataset2_id = H5Dopen2(file_id, "dataset2", H5P_DEFAULT);
    loaded2.read_h5_did(dataset2_id);
    H5Dclose(dataset2_id);

    H5Fclose(file_id);

    TEST_ASSERT(loaded1.size() == 10, "Dataset1 size should match");
    TEST_ASSERT(loaded2.size() == 20, "Dataset2 size should match");
    TEST_ASSERT(loaded1[5] == 5.0, "Dataset1 data should match");
    TEST_ASSERT(loaded2[10] == 20.0, "Dataset2 data should match");

    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All HDF5 Tests Passed ===" << std::endl;

#else
  std::cout << "HDF5 support not enabled!" << std::endl;
  std::cout << "Compile with -DNDARRAY_USE_HDF5=ON to run these tests" << std::endl;
  return 1;
#endif

  return 0;
}
