# Unit tests for ndarray library

# Core functionality tests (may need optional backends for template instantiation)
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_ndarray_core netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_ndarray_core ${HDF5_LIBRARIES})
endif()

add_test(NAME ndarray_core COMMAND test_ndarray_core)

# I/O tests (may use optional backends)
add_executable(test_ndarray_io test_ndarray_io.cpp)
target_link_libraries(test_ndarray_io ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_ndarray_io netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_ndarray_io ${HDF5_LIBRARIES})
endif()

add_test(NAME ndarray_io COMMAND test_ndarray_io)

# HDF5 functionality tests (requires HDF5 support)
if (NDARRAY_HAVE_HDF5)
  add_executable(test_ndarray_hdf5 test_ndarray_hdf5.cpp)
  target_link_libraries(test_ndarray_hdf5 ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_ndarray_hdf5 ${HDF5_LIBRARIES})
  add_test(NAME ndarray_hdf5 COMMAND test_ndarray_hdf5)
endif()

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
  message("  - test_ndarray_hdf5: HDF5 I/O operations")
endif()
message("")
