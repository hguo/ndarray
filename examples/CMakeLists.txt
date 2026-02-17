# Examples for ndarray library

cmake_minimum_required(VERSION 3.10)

# Include parent project's include directories
include_directories(${NDARRAY_INCLUDE_DIR})

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage ndarray ${CMAKE_THREAD_LIBS_INIT})

# I/O operations example
add_executable(io_operations io_operations.cpp)
target_link_libraries(io_operations ndarray ${CMAKE_THREAD_LIBS_INIT})

# Convolution example
add_executable(convolution convolution.cpp)
target_link_libraries(convolution ndarray ${CMAKE_THREAD_LIBS_INIT})

# Vector conversion example
add_executable(vector_conversion_example vector_conversion_example.cpp)
target_link_libraries(vector_conversion_example ndarray ${CMAKE_THREAD_LIBS_INIT})

# MPI parallel example (requires MPI)
if (NDARRAY_HAVE_MPI)
  add_executable(parallel_mpi parallel_mpi.cpp)
  target_link_libraries(parallel_mpi ndarray MPI::MPI_CXX ${CMAKE_THREAD_LIBS_INIT})
endif()

# SYCL acceleration example (requires SYCL)
if (NDARRAY_HAVE_SYCL)
  add_executable(sycl_acceleration sycl_acceleration.cpp)
  target_link_libraries(sycl_acceleration ndarray ${CMAKE_THREAD_LIBS_INIT})
  add_sycl_to_target(TARGET sycl_acceleration SOURCES sycl_acceleration.cpp)
endif()

# Device memory management example (requires CUDA or SYCL)
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  add_executable(device_memory device_memory.cpp)
  target_link_libraries(device_memory ndarray ${CMAKE_THREAD_LIBS_INIT})

  if (NDARRAY_HAVE_SYCL)
    add_sycl_to_target(TARGET device_memory SOURCES device_memory.cpp)
  endif()
endif()

# NOTE: Distributed stream functionality is built into the regular stream class.
# Use ftk::stream with MPI communicator and decompose() on arrays for distributed I/O.
# See parallel_mpi.cpp for distributed array examples.

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
