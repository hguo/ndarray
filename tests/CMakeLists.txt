# Unit tests for ndarray library

# Core functionality tests (no external dependencies)
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME ndarray_core COMMAND test_ndarray_core)

# I/O tests (may use optional backends)
add_executable(test_ndarray_io test_ndarray_io.cpp)
target_link_libraries(test_ndarray_io ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_ndarray_io netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_ndarray_io ${HDF5_LIBRARIES})
endif()

add_test(NAME ndarray_io COMMAND test_ndarray_io)

# Test summary
message("")
message("Unit tests configured:")
message("  - test_ndarray_core: Core array operations")
message("  - test_ndarray_io: File I/O operations")
if (NDARRAY_HAVE_NETCDF)
  message("    + NetCDF support enabled")
endif()
if (NDARRAY_HAVE_HDF5)
  message("    + HDF5 support enabled")
endif()
message("")
