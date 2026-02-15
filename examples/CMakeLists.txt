# Examples for ndarray library

cmake_minimum_required(VERSION 3.10)

# Include parent project's include directories
include_directories(${NDARRAY_INCLUDE_DIR})

# Helper macro to add VTK libraries when VTK is enabled
# These are needed because ndarray.hh includes VTK headers when NDARRAY_HAVE_VTK is ON
macro(add_vtk_if_enabled target)
  if (NDARRAY_HAVE_VTK)
    target_link_libraries(${target} VTK::CommonCore VTK::CommonDataModel VTK::IOLegacy VTK::IOXML)
  endif()
endmacro()

# Basic usage example (may need optional backends if enabled)
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(basic_usage)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(basic_usage netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(basic_usage ${HDF5_LIBRARIES})
endif()

# I/O operations example (may use optional backends)
add_executable(io_operations io_operations.cpp)
target_link_libraries(io_operations ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(io_operations)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(io_operations netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(io_operations ${HDF5_LIBRARIES})
endif()

# Convolution example (may need optional backends if enabled)
add_executable(convolution convolution.cpp)
target_link_libraries(convolution ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(convolution)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(convolution netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(convolution ${HDF5_LIBRARIES})
endif()

# Vector conversion example (may need optional backends if enabled)
add_executable(vector_conversion_example vector_conversion_example.cpp)
target_link_libraries(vector_conversion_example ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(vector_conversion_example)

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(vector_conversion_example ${HDF5_LIBRARIES})
endif()

# MPI parallel example (requires MPI, may need optional backends)
if (NDARRAY_HAVE_MPI)
  add_executable(parallel_mpi parallel_mpi.cpp)
  target_link_libraries(parallel_mpi ${CMAKE_THREAD_LIBS_INIT})
  add_vtk_if_enabled(parallel_mpi)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(parallel_mpi netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(parallel_mpi ${HDF5_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    target_link_libraries(parallel_mpi ${PNETCDF_LIBRARY})
  endif()
endif()

# SYCL acceleration example (requires SYCL, may need optional backends)
if (NDARRAY_HAVE_SYCL)
  add_executable(sycl_acceleration sycl_acceleration.cpp)
  target_link_libraries(sycl_acceleration ${CMAKE_THREAD_LIBS_INIT})
  add_vtk_if_enabled(sycl_acceleration)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(sycl_acceleration netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(sycl_acceleration ${HDF5_LIBRARIES})
  endif()

  add_sycl_to_target(TARGET sycl_acceleration SOURCES sycl_acceleration.cpp)
endif()

# Device memory management example (requires CUDA or SYCL, may need optional backends)
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  add_executable(device_memory device_memory.cpp)
  target_link_libraries(device_memory ${CMAKE_THREAD_LIBS_INIT})
  add_vtk_if_enabled(device_memory)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(device_memory netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(device_memory ${HDF5_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_CUDA)
    target_link_libraries(device_memory ${CUDA_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_SYCL)
    add_sycl_to_target(TARGET device_memory SOURCES device_memory.cpp)
  endif()
endif()

# Distributed stream examples (requires MPI and YAML)
if (NDARRAY_HAVE_MPI AND NDARRAY_HAVE_YAML)
  # Basic distributed stream example
  add_executable(distributed_stream distributed_stream.cpp)
  target_link_libraries(distributed_stream ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(distributed_stream MPI::MPI_CXX)
  target_link_libraries(distributed_stream yaml-cpp::yaml-cpp)
  add_vtk_if_enabled(distributed_stream)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(distributed_stream netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(distributed_stream ${HDF5_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    target_link_libraries(distributed_stream ${PNETCDF_LIBRARIES})
  endif()

  # YAML-based distributed stream example
  add_executable(distributed_stream_yaml distributed_stream_yaml.cpp)
  target_link_libraries(distributed_stream_yaml ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(distributed_stream_yaml MPI::MPI_CXX)
  target_link_libraries(distributed_stream_yaml yaml-cpp::yaml-cpp)
  add_vtk_if_enabled(distributed_stream_yaml)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(distributed_stream_yaml netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(distributed_stream_yaml ${HDF5_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    target_link_libraries(distributed_stream_yaml ${PNETCDF_LIBRARIES})
  endif()

  # Dry-run example
  add_executable(distributed_stream_dryrun distributed_stream_dryrun.cpp)
  target_link_libraries(distributed_stream_dryrun ${CMAKE_THREAD_LIBS_INIT})
  target_link_libraries(distributed_stream_dryrun MPI::MPI_CXX)
  target_link_libraries(distributed_stream_dryrun yaml-cpp::yaml-cpp)
  add_vtk_if_enabled(distributed_stream_dryrun)

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(distributed_stream_dryrun netCDF::netcdf)
  endif()

  if (NDARRAY_HAVE_HDF5)
    target_link_libraries(distributed_stream_dryrun ${HDF5_LIBRARIES})
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    target_link_libraries(distributed_stream_dryrun ${PNETCDF_LIBRARIES})
  endif()
endif()

# Installation
install(TARGETS basic_usage io_operations convolution
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)

if (NDARRAY_HAVE_MPI)
  install(TARGETS parallel_mpi
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)

  if (NDARRAY_HAVE_YAML)
    install(TARGETS distributed_stream distributed_stream_yaml distributed_stream_dryrun
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
  endif()
endif()

if (NDARRAY_HAVE_SYCL)
  install(TARGETS sycl_acceleration
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
endif()

if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  install(TARGETS device_memory
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
endif()

# Install source files as documentation
install(FILES
  basic_usage.cpp
  io_operations.cpp
  convolution.cpp
  parallel_mpi.cpp
  sycl_acceleration.cpp
  device_memory.cpp
  README.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples)
