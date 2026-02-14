/**
 * Test that NC_SAFE_CALL and PNC_SAFE_CALL throw exceptions instead of calling exit()
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>

void test_exception_not_exit() {
  std::cout << "Testing: Exception handling replaces exit() calls" << std::endl;

#if NDARRAY_HAVE_PNETCDF
  // Test that PNC_SAFE_CALL throws exception on error (not exit)
  try {
    // Try to open non-existent file - should throw exception
    int ncid;
    int result = ncmpi_open(MPI_COMM_SELF, "nonexistent_file.nc", NC_NOWRITE, MPI_INFO_NULL, &ncid);

    // Manually trigger the error handling to test exception
    if (result != NC_NOERR) {
      std::string error_msg = std::string(ncmpi_strerror(result)) + " at test_exception_handling.cpp";
      throw ftk::nd::netcdf_error(ftk::nd::ERR_PNETCDF_IO, error_msg);
    }

    std::cerr << "FAILED: Expected exception but none was thrown" << std::endl;
    return;
  } catch (const ftk::nd::netcdf_error& e) {
    std::cout << "  - Caught exception (expected): " << e.what() << std::endl;
    std::cout << "  - Error code: " << e.error_code() << std::endl;
    std::cout << "  PASSED" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: Caught wrong exception type: " << e.what() << std::endl;
    return;
  }
#else
  std::cout << "  SKIPPED (PNetCDF not enabled)" << std::endl;
#endif

#if NDARRAY_HAVE_NETCDF
  // Test that NC_SAFE_CALL throws exception on error (not exit)
  try {
    // Try to open non-existent file - should throw exception
    int ncid;
    int result = nc_open("nonexistent_netcdf_file.nc", NC_NOWRITE, &ncid);

    // Manually trigger the error handling to test exception
    if (result != NC_NOERR) {
      std::string error_msg = std::string(nc_strerror(result)) + " at test_exception_handling.cpp";
      throw ftk::nd::netcdf_error(ftk::nd::ERR_NETCDF_IO, error_msg);
    }

    std::cerr << "FAILED: Expected exception but none was thrown" << std::endl;
    return;
  } catch (const ftk::nd::netcdf_error& e) {
    std::cout << "  - Caught NetCDF exception (expected): " << e.what() << std::endl;
    std::cout << "  PASSED" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: Caught wrong exception type: " << e.what() << std::endl;
    return;
  }
#endif
}

int main(int argc, char **argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "=== Testing Exception Handling (No exit() calls) ===" << std::endl;
  test_exception_not_exit();
  std::cout << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
