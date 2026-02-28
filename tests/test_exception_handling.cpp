/**
 * Test that error-handling macros throw exceptions instead of calling exit().
 * Exercises NC_SAFE_CALL, PNC_SAFE_CALL, and HDF5 error paths through
 * actual ndarray I/O methods with non-existent files.
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>

void test_netcdf_exception() {
  std::cout << "Testing: NetCDF exception on non-existent file" << std::endl;
#if NDARRAY_HAVE_NETCDF
  try {
    ftk::ndarray<float> arr;
    arr.read_netcdf("nonexistent_file_that_does_not_exist.nc", "var");
    std::cerr << "  FAILED: Expected exception but none was thrown" << std::endl;
    assert(false);
  } catch (const ftk::netcdf_error& e) {
    std::cout << "  - Caught netcdf_error: " << e.what() << std::endl;
    std::cout << "  PASSED" << std::endl;
  }
#else
  std::cout << "  SKIPPED (NetCDF not enabled)" << std::endl;
#endif
}

void test_hdf5_exception() {
  std::cout << "Testing: HDF5 exception on non-existent file" << std::endl;
#if NDARRAY_HAVE_HDF5
  try {
    ftk::ndarray<float> arr;
    arr.read_h5("nonexistent_file_that_does_not_exist.h5", "dataset");
    std::cerr << "  FAILED: Expected exception but none was thrown" << std::endl;
    assert(false);
  } catch (const ftk::hdf5_error& e) {
    std::cout << "  - Caught hdf5_error: " << e.what() << std::endl;
    std::cout << "  PASSED" << std::endl;
  }
#else
  std::cout << "  SKIPPED (HDF5 not enabled)" << std::endl;
#endif
}

int main(int argc, char **argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "=== Testing Exception Handling ===" << std::endl;
  test_netcdf_exception();
  test_hdf5_exception();
  std::cout << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
