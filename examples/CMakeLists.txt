# Examples for ndarray library

cmake_minimum_required(VERSION 3.10)

# Include parent project's include directories
include_directories(${NDARRAY_INCLUDE_DIR})

# Basic usage example (may need optional backends if enabled)
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(basic_usage netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(basic_usage ${HDF5_LIBRARIES})
endif()

# I/O operations example (may use optional backends)
add_executable(io_operations io_operations.cpp)
target_link_libraries(io_operations ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(io_operations netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(io_operations ${HDF5_LIBRARIES})
endif()

# Convolution example (may need optional backends if enabled)
add_executable(convolution convolution.cpp)
target_link_libraries(convolution ${CMAKE_THREAD_LIBS_INIT})

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(convolution netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(convolution ${HDF5_LIBRARIES})
endif()

# MPI parallel example (requires MPI, may need optional backends)
if (NDARRAY_HAVE_MPI)
  add_executable(parallel_mpi parallel_mpi.cpp)
  target_link_libraries(parallel_mpi ${CMAKE_THREAD_LIBS_INIT})

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(parallel_mpi netcdf)
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

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(sycl_acceleration netcdf)
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

  if (NDARRAY_HAVE_NETCDF)
    target_link_libraries(device_memory netcdf)
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

# Installation
install(TARGETS basic_usage io_operations convolution
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)

if (NDARRAY_HAVE_MPI)
  install(TARGETS parallel_mpi
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
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
