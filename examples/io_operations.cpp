#include <ndarray/ndarray.hh>
#include <iostream>

/**
 * I/O operations example for ndarray
 *
 * This example demonstrates:
 * - Reading and writing binary data
 * - File I/O with different formats (when available)
 * - Serialization and deserialization
 *
 * Note: NetCDF, HDF5, and ADIOS2 examples require
 * building with respective flags enabled
 */

int main() {
  std::cout << "=== ndarray I/O Operations Example ===" << std::endl << std::endl;

  // 1. Create sample data
  std::cout << "1. Creating sample data:" << std::endl;
  ftk::ndarray<double> original_data;
  original_data.reshapef(10, 10, 5);

  // Fill with sample values
  for (size_t i = 0; i < original_data.size(); i++) {
    original_data[i] = static_cast<double>(i) * 0.01;
  }
  std::cout << "   - Created array: " << original_data.dimf(0) << " x "
            << original_data.dimf(1) << " x " << original_data.dimf(2) << std::endl;
  std::cout << "   - Total elements: " << original_data.size() << std::endl << std::endl;

  // 2. Write to binary file
  std::cout << "2. Writing to binary file:" << std::endl;
  try {
    original_data.to_binary_file("output_data.bin");
    std::cout << "   - Successfully wrote data to output_data.bin" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "   - Error writing binary: " << e.what() << std::endl;
  }
  std::cout << std::endl;

  // 3. Read from binary file
  std::cout << "3. Reading from binary file:" << std::endl;
  std::cout << "   - Note: Binary format doesn't store dimensions - must reshape first!" << std::endl;
  ftk::ndarray<double> loaded_data;
  loaded_data.reshapef(10, 10, 5);  // Must know and set dimensions before reading
  try {
    loaded_data.read_binary_file("output_data.bin");
    std::cout << "   - Successfully read data from output_data.bin" << std::endl;
    std::cout << "   - Loaded array dimensions: " << loaded_data.nd() << std::endl;
    std::cout << "   - Loaded array size: " << loaded_data.size() << std::endl;

    // Verify data integrity
    bool data_matches = true;
    if (loaded_data.size() == original_data.size()) {
      for (size_t i = 0; i < loaded_data.size(); i++) {
        if (loaded_data[i] != original_data[i]) {
          data_matches = false;
          break;
        }
      }
    } else {
      data_matches = false;
    }
    std::cout << "   - Data integrity check: "
              << (data_matches ? "PASSED" : "FAILED") << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "   - Error reading binary: " << e.what() << std::endl;
  }
  std::cout << std::endl;

  // Note: NetCDF and HDF5 I/O require manual file opening with ncid/varid or hid_t
  // The convenience methods write_netcdf()/write_h5() don't exist in this library
  // Use read_netcdf(filename, varname) and to_netcdf(ncid, varid) instead
  std::cout << "4. NetCDF/HDF5 I/O: Requires manual file handle management" << std::endl;
  std::cout << "   - Use read_netcdf(filename, varname) for reading" << std::endl;
  std::cout << "   - Use to_netcdf(ncid, varid) for writing (requires opened file)" << std::endl;
  std::cout << std::endl;

  // 5. Array metadata
  std::cout << "5. Array metadata and properties:" << std::endl;
  std::cout << "   - Element size: " << original_data.elem_size() << " bytes" << std::endl;
  std::cout << "   - Total memory: " << (original_data.size() * original_data.elem_size())
            << " bytes" << std::endl;
  std::cout << "   - Dimensions: " << original_data.nd() << std::endl;
  std::cout << "   - Empty: " << (original_data.empty() ? "yes" : "no") << std::endl;
  std::cout << std::endl;

  // 6. Working with different data types
  std::cout << "6. I/O with different data types:" << std::endl;

  // Float array
  ftk::ndarray<float> float_arr;
  float_arr.reshapef(20, 30);
  for (size_t i = 0; i < float_arr.size(); i++) {
    float_arr[i] = static_cast<float>(i) * 0.5f;
  }
  float_arr.to_binary_file("float_data.bin");
  std::cout << "   - Saved float array (20x30)" << std::endl;

  // Integer array
  ftk::ndarray<int> int_arr;
  int_arr.reshapef(15, 15);
  for (size_t i = 0; i < int_arr.size(); i++) {
    int_arr[i] = static_cast<int>(i);
  }
  int_arr.to_binary_file("int_data.bin");
  std::cout << "   - Saved int array (15x15)" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Example completed successfully ===" << std::endl;
  std::cout << "Generated files:" << std::endl;
  std::cout << "  - output_data.bin" << std::endl;
  std::cout << "  - float_data.bin" << std::endl;
  std::cout << "  - int_data.bin" << std::endl;

  return 0;
}
