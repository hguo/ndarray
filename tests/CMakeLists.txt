# Unit tests for ndarray library

# Helper macro to add VTK libraries to all tests when VTK is enabled
# These are needed because ndarray.hh includes VTK headers when NDARRAY_HAVE_VTK is ON
macro(add_vtk_if_enabled target)
  if (NDARRAY_HAVE_VTK)
    target_link_libraries(${target} VTK::CommonCore VTK::CommonDataModel VTK::IOLegacy VTK::IOXML)
  endif()
endmacro()

# Core functionality tests (may need optional backends for template instantiation)
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_ndarray_core)

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
add_vtk_if_enabled(test_ndarray_io)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_ndarray_io netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_ndarray_io ${HDF5_LIBRARIES})
endif()

add_test(NAME ndarray_io COMMAND test_ndarray_io)

# Stream functionality tests (requires YAML support)
if (NDARRAY_HAVE_YAML)
  add_executable(test_ndarray_stream test_ndarray_stream.cpp)
  target_link_libraries(test_ndarray_stream ${CMAKE_THREAD_LIBS_INIT} yaml-cpp::yaml-cpp)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(test_ndarray_stream netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(test_ndarray_stream ${HDF5_LIBRARIES})
  endif()

  add_test(NAME ndarray_stream COMMAND test_ndarray_stream)
endif()

# Zero-copy optimization test
add_executable(test_zero_copy test_zero_copy.cpp)
target_link_libraries(test_zero_copy ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_zero_copy)
add_test(NAME zero_copy COMMAND test_zero_copy)

# Variable name matching test
add_executable(test_variable_names test_variable_names.cpp)
target_link_libraries(test_variable_names ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_variable_names)
add_test(NAME variable_names COMMAND test_variable_names)

# Vector conversion test
add_executable(test_vector_conversion test_vector_conversion.cpp)
target_link_libraries(test_vector_conversion ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_vector_conversion)
add_test(NAME vector_conversion COMMAND test_vector_conversion)

# F/C ordering demonstration
add_executable(test_fc_ordering test_fc_ordering.cpp)
target_link_libraries(test_fc_ordering ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_fc_ordering)
add_test(NAME fc_ordering COMMAND test_fc_ordering)

# HDF5 functionality tests (requires HDF5 support)
if (NDARRAY_HAVE_HDF5)
  add_executable(test_ndarray_hdf5 test_ndarray_hdf5.cpp)
  target_link_libraries(test_ndarray_hdf5 ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_ndarray_hdf5 ${HDF5_LIBRARIES})
  add_test(NAME ndarray_hdf5 COMMAND test_ndarray_hdf5)
endif()

# VTK functionality tests (requires VTK support)
if (NDARRAY_HAVE_VTK)
  add_executable(test_vtk test_vtk.cpp)
  target_link_libraries(test_vtk ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_vtk VTK::CommonCore VTK::CommonDataModel VTK::IOLegacy VTK::IOXML)
  add_test(NAME vtk COMMAND test_vtk)
endif()

# Parallel NetCDF functionality tests (requires PNetCDF and MPI support)
if (NDARRAY_HAVE_PNETCDF AND NDARRAY_HAVE_MPI)
  add_executable(test_pnetcdf test_pnetcdf.cpp)
  target_link_libraries(test_pnetcdf ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_pnetcdf ${PNETCDF_LIBRARIES})
  target_link_libraries(test_pnetcdf MPI::MPI_CXX)
  add_vtk_if_enabled(test_pnetcdf)

  # PNetCDF tests should be run with mpirun
  # add_test(NAME pnetcdf COMMAND mpirun -np 4 test_pnetcdf)
  # Note: MPI tests require mpirun which may not be available in all test environments
  # Users should run manually: mpirun -np 4 ./bin/test_pnetcdf
endif()

# ADIOS2 functionality tests (requires ADIOS2 support)
if (NDARRAY_HAVE_ADIOS2)
  add_executable(test_adios2 test_adios2.cpp)
  target_link_libraries(test_adios2 ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_adios2 adios2::adios2)
  add_vtk_if_enabled(test_adios2)

  if (NDARRAY_HAVE_MPI)
    target_link_libraries(test_adios2 MPI::MPI_CXX)
  endif()

  add_test(NAME adios2 COMMAND test_adios2)
endif()

# GPU functionality tests (requires CUDA or SYCL support)
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  add_executable(test_gpu test_gpu.cpp)
  target_link_libraries(test_gpu ${CMAKE_THREAD_LIBS_INIT})
  add_vtk_if_enabled(test_gpu)

  if (NDARRAY_HAVE_CUDA)
    target_link_libraries(test_gpu CUDA::cudart)
  endif()

  if (NDARRAY_HAVE_SYCL)
    # Add SYCL-specific flags and libraries
    target_compile_options(test_gpu PRIVATE ${SYCL_FLAGS})
  endif()

  add_test(NAME gpu COMMAND test_gpu)
endif()

# PNG functionality tests (requires PNG support)
if (NDARRAY_HAVE_PNG)
  add_executable(test_png test_png.cpp)
  target_link_libraries(test_png ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_png PNG::PNG)
  add_vtk_if_enabled(test_png)
  add_test(NAME png COMMAND test_png)
endif()

# Test summary
message("")
message("Unit tests configured:")
message("  - test_ndarray_core: Core array operations")
message("  - test_ndarray_io: File I/O operations")
message("  - test_ndarray_stream: YAML stream functionality")
message("  - test_zero_copy: Zero-copy optimization")
message("  - test_variable_names: Variable name matching")
message("  - test_vector_conversion: Vector conversion")
message("  - test_fc_ordering: F/C ordering demonstration")
if (NDARRAY_HAVE_NETCDF)
  message("    + NetCDF support enabled")
endif()
if (NDARRAY_HAVE_HDF5)
  message("    + HDF5 support enabled")
  message("  - test_ndarray_hdf5: HDF5 I/O operations")
endif()
if (NDARRAY_HAVE_VTK)
  message("    + VTK support enabled")
  message("  - test_vtk: VTK I/O operations")
endif()
if (NDARRAY_HAVE_PNETCDF AND NDARRAY_HAVE_MPI)
  message("    + PNetCDF support enabled")
  message("  - test_pnetcdf: Parallel NetCDF I/O operations (run with: mpirun -np 4 ./bin/test_pnetcdf)")
endif()
if (NDARRAY_HAVE_ADIOS2)
  message("    + ADIOS2 support enabled")
  message("  - test_adios2: ADIOS2 BP file I/O operations")
endif()
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  if (NDARRAY_HAVE_CUDA)
    message("    + CUDA support enabled")
  endif()
  if (NDARRAY_HAVE_SYCL)
    message("    + SYCL support enabled")
  endif()
  message("  - test_gpu: GPU device memory management and transfers")
endif()
if (NDARRAY_HAVE_PNG)
  message("    + PNG support enabled")
  message("  - test_png: PNG image I/O operations")
endif()

# Exception handling test (NetCDF or PNetCDF)
if (NDARRAY_HAVE_NETCDF OR NDARRAY_HAVE_PNETCDF)
  add_executable(test_exception_handling test_exception_handling.cpp)
  target_link_libraries(test_exception_handling ${CMAKE_THREAD_LIBS_INIT})

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(test_exception_handling netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    target_link_libraries(test_exception_handling ${PNETCDF_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_MPI)
    target_link_libraries(test_exception_handling MPI::MPI_CXX)
  endif()

  add_vtk_if_enabled(test_exception_handling)
  add_test(NAME exception_handling COMMAND test_exception_handling)
  message("  - test_exception_handling: Verify exceptions instead of exit()")
endif()

# Storage backend tests
add_executable(test_storage_backends test_storage_backends.cpp)
target_link_libraries(test_storage_backends ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_storage_backends)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_storage_backends netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_storage_backends ${HDF5_LIBRARIES})
endif()

add_test(NAME storage_backends COMMAND test_storage_backends)
message("  - test_storage_backends: Storage backend functionality")

# Storage backend stream tests (requires YAML)
add_executable(test_storage_streams test_storage_streams.cpp)
target_link_libraries(test_storage_streams ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_storage_streams)

if (NDARRAY_HAVE_YAML)
  target_link_libraries(test_storage_streams yaml-cpp::yaml-cpp)
endif()

add_test(NAME storage_streams COMMAND test_storage_streams)
message("  - test_storage_streams: Storage backend stream functionality")

# Storage backend memory management tests
add_executable(test_storage_memory test_storage_memory.cpp)
target_link_libraries(test_storage_memory ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_storage_memory)
add_test(NAME storage_memory COMMAND test_storage_memory)
message("  - test_storage_memory: Storage backend memory management")

# Storage backend performance benchmarks
add_executable(benchmark_storage benchmark_storage.cpp)
target_link_libraries(benchmark_storage ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(benchmark_storage)
# Note: benchmarks are not added as tests, they are run manually
message("  - benchmark_storage: Storage backend performance (run manually)")

# Distributed ndarray tests (require MPI)
if (NDARRAY_HAVE_MPI)
  add_executable(test_distributed_ndarray test_distributed_ndarray.cpp)
  target_link_libraries(test_distributed_ndarray ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(test_distributed_ndarray MPI::MPI_CXX)
  add_vtk_if_enabled(test_distributed_ndarray)

  # Note: Distributed tests should be run with mpirun
  # add_test(NAME distributed_ndarray COMMAND mpirun -np 4 test_distributed_ndarray)
  # Note: MPI tests require mpirun which may not be available in all test environments
  # Users should run manually: mpirun -np 4 ./bin/test_distributed_ndarray
  message("  - test_distributed_ndarray: Distributed memory decomposition (run with: mpirun -np 4 ./bin/test_distributed_ndarray)")
endif()

message("")
