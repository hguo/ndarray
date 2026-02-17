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

# Distributed stream examples (requires MPI and YAML)
# TODO: Disabled until distributed_stream class is implemented
# The distributed_stream class was planned but not completed
# For now, use regular stream with manual decompose() calls
#
# if (NDARRAY_HAVE_MPI AND NDARRAY_HAVE_YAML)
#   # Basic distributed stream example
#   add_executable(distributed_stream distributed_stream.cpp)
#   target_link_libraries(distributed_stream ndarray MPI::MPI_CXX ${CMAKE_THREAD_LIBS_INIT})
#
#   # YAML-based distributed stream example
#   add_executable(distributed_stream_yaml distributed_stream_yaml.cpp)
#   target_link_libraries(distributed_stream_yaml ndarray MPI::MPI_CXX ${CMAKE_THREAD_LIBS_INIT})
#
#   # Dry-run example
#   add_executable(distributed_stream_dryrun distributed_stream_dryrun.cpp)
#   target_link_libraries(distributed_stream_dryrun ndarray MPI::MPI_CXX ${CMAKE_THREAD_LIBS_INIT})
# endif()

# Installation
install(TARGETS basic_usage io_operations convolution
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)

if (NDARRAY_HAVE_MPI)
  install(TARGETS parallel_mpi
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)

  # TODO: Re-enable when distributed_stream is implemented
  # if (NDARRAY_HAVE_YAML)
  #   install(TARGETS distributed_stream distributed_stream_yaml distributed_stream_dryrun
  #     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples)
  # endif()
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
