# Unit tests for ndarray library

# Core functionality tests
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ndarray)
add_test(NAME ndarray_core COMMAND test_ndarray_core)

# I/O tests
add_executable(test_ndarray_io test_ndarray_io.cpp)
target_link_libraries(test_ndarray_io ndarray)
add_test(NAME ndarray_io COMMAND test_ndarray_io)

# Stream functionality tests (requires YAML support)
if (NDARRAY_HAVE_YAML)
  add_executable(test_ndarray_stream test_ndarray_stream.cpp)
  target_link_libraries(test_ndarray_stream ndarray)
  add_test(NAME ndarray_stream COMMAND test_ndarray_stream)
endif()

# Zero-copy optimization test
add_executable(test_zero_copy test_zero_copy.cpp)
target_link_libraries(test_zero_copy ndarray)
add_test(NAME zero_copy COMMAND test_zero_copy)

# Variable name matching test
add_executable(test_variable_names test_variable_names.cpp)
target_link_libraries(test_variable_names ndarray)
add_test(NAME variable_names COMMAND test_variable_names)

# Vector conversion test
add_executable(test_vector_conversion test_vector_conversion.cpp)
target_link_libraries(test_vector_conversion ndarray)
add_test(NAME vector_conversion COMMAND test_vector_conversion)

# F/C ordering demonstration
add_executable(test_fc_ordering test_fc_ordering.cpp)
target_link_libraries(test_fc_ordering ndarray)
add_test(NAME fc_ordering COMMAND test_fc_ordering)

# HDF5 functionality tests (requires HDF5 support)
if (NDARRAY_HAVE_HDF5)
  add_executable(test_ndarray_hdf5 test_ndarray_hdf5.cpp)
  target_link_libraries(test_ndarray_hdf5 ndarray)
  add_test(NAME ndarray_hdf5 COMMAND test_ndarray_hdf5)

  add_executable(test_hdf5_exceptions test_hdf5_exceptions.cpp)
  target_link_libraries(test_hdf5_exceptions ndarray)
  add_test(NAME hdf5_exceptions COMMAND test_hdf5_exceptions)
endif()

# VTK functionality tests (requires VTK support)
if (NDARRAY_HAVE_VTK)
  add_executable(test_vtk test_vtk.cpp)
  target_link_libraries(test_vtk ndarray)
  add_test(NAME vtk COMMAND test_vtk)
endif()

# Parallel NetCDF functionality tests (requires PNetCDF and MPI support)
if (NDARRAY_HAVE_PNETCDF AND NDARRAY_HAVE_MPI)
  add_executable(test_pnetcdf test_pnetcdf.cpp)
  target_link_libraries(test_pnetcdf ndarray)
  # PNetCDF tests should be run with mpirun
  # add_test(NAME pnetcdf COMMAND mpirun -np 4 test_pnetcdf)
  # Note: MPI tests require mpirun which may not be available in all test environments
  # Users should run manually: mpirun -np 4 ./bin/test_pnetcdf
endif()

# ADIOS2 functionality tests (requires ADIOS2 support)
if (NDARRAY_HAVE_ADIOS2)
  add_executable(test_adios2 test_adios2.cpp)
  target_link_libraries(test_adios2 ndarray)
  add_test(NAME adios2 COMMAND test_adios2)

  # ADIOS2 stream tests (requires ADIOS2 and YAML)
  if (NDARRAY_HAVE_YAML)
    add_executable(test_adios2_stream test_adios2_stream.cpp)
    target_link_libraries(test_adios2_stream ndarray)
    add_test(NAME adios2_stream COMMAND test_adios2_stream)
  endif()
endif()

# GPU functionality tests (requires CUDA or IntelSYCL support)
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  add_executable(test_gpu test_gpu.cpp)
  target_link_libraries(test_gpu ndarray)

  if (NDARRAY_HAVE_SYCL)
    # Add IntelSYCL-specific flags and libraries
    target_compile_options(test_gpu PRIVATE ${SYCL_FLAGS})
  endif()

  add_test(NAME gpu COMMAND test_gpu)

  # GPU kernel and data management tests
  add_executable(test_gpu_kernels test_gpu_kernels.cpp)
  target_link_libraries(test_gpu_kernels ndarray)

  if (NDARRAY_HAVE_SYCL)
    target_compile_options(test_gpu_kernels PRIVATE ${SYCL_FLAGS})
  endif()

  add_test(NAME gpu_kernels COMMAND test_gpu_kernels)
endif()

# PNG functionality tests (requires PNG support)
if (NDARRAY_HAVE_PNG)
  add_executable(test_png test_png.cpp)
  target_link_libraries(test_png ndarray)
  add_test(NAME png COMMAND test_png)
endif()

# Arithmetic operator tests
add_executable(test_arithmetic test_arithmetic.cpp)
target_link_libraries(test_arithmetic ndarray)
add_test(NAME arithmetic COMMAND test_arithmetic)

# Statistical function tests
add_executable(test_statistics test_statistics.cpp)
target_link_libraries(test_statistics ndarray)
add_test(NAME statistics COMMAND test_statistics)

# Data manipulation tests
add_executable(test_data_ops test_data_ops.cpp)
target_link_libraries(test_data_ops ndarray)
add_test(NAME data_ops COMMAND test_data_ops)

# Gradient computation tests
add_executable(test_gradient test_gradient.cpp)
target_link_libraries(test_gradient ndarray)
add_test(NAME gradient COMMAND test_gradient)

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
message("  - test_arithmetic: Arithmetic operators and scale/add")
message("  - test_statistics: Statistical functions (min_max, maxabs, resolution)")
message("  - test_data_ops: Data manipulation (perturb, flip, slice, concat, stack)")
message("  - test_gradient: Gradient computations (gradient2D, gradient3D)")
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
  if (NDARRAY_HAVE_YAML)
    message("  - test_adios2_stream: ADIOS2 YAML stream functionality")
  endif()
endif()
if (NDARRAY_HAVE_CUDA OR NDARRAY_HAVE_SYCL)
  if (NDARRAY_HAVE_CUDA)
    message("    + CUDA support enabled")
  endif()
  if (NDARRAY_HAVE_SYCL)
    message("    + IntelSYCL support enabled")
  endif()
  message("  - test_gpu: GPU device memory management and transfers")
  message("  - test_gpu_kernels: GPU data management and kernel tests (fill, scale, add)")
endif()
if (NDARRAY_HAVE_PNG)
  message("    + PNG support enabled")
  message("  - test_png: PNG image I/O operations")
endif()

# Exception handling test (NetCDF or PNetCDF)
if (NDARRAY_HAVE_NETCDF OR NDARRAY_HAVE_PNETCDF)
  add_executable(test_exception_handling test_exception_handling.cpp)
  target_link_libraries(test_exception_handling ndarray)
  add_test(NAME exception_handling COMMAND test_exception_handling)
  message("  - test_exception_handling: Verify exceptions instead of exit()")
endif()

# Storage backend tests
add_executable(test_storage_backends test_storage_backends.cpp)
target_link_libraries(test_storage_backends ndarray)
add_test(NAME storage_backends COMMAND test_storage_backends)
message("  - test_storage_backends: Storage backend functionality")

# Storage backend stream tests (requires YAML)
add_executable(test_storage_streams test_storage_streams.cpp)
target_link_libraries(test_storage_streams ndarray)
add_test(NAME storage_streams COMMAND test_storage_streams)
message("  - test_storage_streams: Storage backend stream functionality")

# Storage backend memory management tests
add_executable(test_storage_memory test_storage_memory.cpp)
target_link_libraries(test_storage_memory ndarray)
add_test(NAME storage_memory COMMAND test_storage_memory)
message("  - test_storage_memory: Storage backend memory management")

# Storage backend performance benchmarks
add_executable(benchmark_storage benchmark_storage.cpp)
target_link_libraries(benchmark_storage ndarray)
# Note: benchmarks are not added as tests, they are run manually
message("  - benchmark_storage: Storage backend performance (run manually)")

# Distributed ndarray tests (require MPI)
if (NDARRAY_HAVE_MPI)
  add_executable(test_distributed_ndarray test_distributed_ndarray.cpp)
  target_link_libraries(test_distributed_ndarray ndarray)
  # Note: Distributed tests should be run with mpirun
  # add_test(NAME distributed_ndarray COMMAND mpirun -np 4 test_distributed_ndarray)
  # Note: MPI tests require mpirun which may not be available in all test environments
  # Users should run manually: mpirun -np 4 ./bin/test_distributed_ndarray
  message("  - test_distributed_ndarray: Distributed memory decomposition (run with: mpirun -np 4 ./bin/test_distributed_ndarray)")

  # Ghost exchange tests (require MPI)
  add_executable(test_ghost_exchange test_ghost_exchange.cpp)
  target_link_libraries(test_ghost_exchange ndarray)
  message("  - test_ghost_exchange: Ghost layer exchange between MPI ranks (run with: mpirun -np 4 ./bin/test_ghost_exchange)")

  # Arbitrary ghost widths tests (require MPI)
  add_executable(test_arbitrary_ghosts test_arbitrary_ghosts.cpp)
  target_link_libraries(test_arbitrary_ghosts ndarray)
  message("  - test_arbitrary_ghosts: Ghost exchange with non-uniform widths (run with: mpirun -np 4 ./bin/test_arbitrary_ghosts)")

  # Distributed YAML streams (require MPI and YAML)
  if (NDARRAY_HAVE_YAML)
    add_executable(test_distributed_yaml_stream test_distributed_yaml_stream.cpp)
    target_link_libraries(test_distributed_yaml_stream ndarray)
    message("  - test_distributed_yaml_stream: Distributed synthetic streams with domain decomposition (run with: mpirun -np 4 ./bin/test_distributed_yaml_stream)")
  endif()

  if (NDARRAY_HAVE_PNETCDF)
    add_executable(test_pnetcdf_auto test_pnetcdf_auto.cpp)
    target_link_libraries(test_pnetcdf_auto ndarray)
    message("  - test_pnetcdf_auto: Distribution-aware PNetCDF I/O (run with: mpirun -np 4 ./bin/test_pnetcdf_auto)")
  endif()

  if (NDARRAY_HAVE_HDF5)
    add_executable(test_hdf5_auto test_hdf5_auto.cpp)
    target_link_libraries(test_hdf5_auto ndarray)
    message("  - test_hdf5_auto: Distribution-aware HDF5 I/O (run with: mpirun -np 4 ./bin/test_hdf5_auto)")
  endif()

  if (NDARRAY_HAVE_ADIOS2)
    add_executable(test_adios2_parallel test_adios2_parallel.cpp)
    target_link_libraries(test_adios2_parallel ndarray)
    message("  - test_adios2_parallel: Parallel ADIOS2 I/O with MPI (run with: mpirun -np 4 ./bin/test_adios2_parallel)")
  endif()

  # Distributed transpose tests (require MPI)
  add_executable(test_transpose_distributed test_transpose_distributed.cpp)
  target_link_libraries(test_transpose_distributed ndarray)
  # Disable optimizations to avoid Heisenbug in distributed transpose
  target_compile_options(test_transpose_distributed PRIVATE -O0 -g)
  message("  - test_transpose_distributed: Distributed array transpose with MPI (run with: mpirun -np 4 ./bin/test_transpose_distributed)")

  # Distributed GPU transpose tests (require MPI + CUDA)
  if (NDARRAY_HAVE_CUDA)
    add_executable(test_transpose_distributed_gpu test_transpose_distributed_gpu.cpp)
    target_link_libraries(test_transpose_distributed_gpu ndarray)
    message("  - test_transpose_distributed_gpu: GPU-accelerated distributed transpose (run with: mpirun -np 4 ./bin/test_transpose_distributed_gpu)")
  endif()
endif()

# Transpose tests (basic functionality)
add_executable(test_transpose test_transpose.cpp)
target_link_libraries(test_transpose ndarray)
add_test(NAME transpose COMMAND test_transpose)
message("  - test_transpose: Core transpose operations")

add_executable(test_transpose_metadata test_transpose_metadata.cpp)
target_link_libraries(test_transpose_metadata ndarray)
add_test(NAME transpose_metadata COMMAND test_transpose_metadata)
message("  - test_transpose_metadata: Transpose with multicomponent/time arrays")

add_executable(test_transpose_performance test_transpose_performance.cpp)
target_link_libraries(test_transpose_performance ndarray)
add_test(NAME transpose_performance COMMAND test_transpose_performance)
message("  - test_transpose_performance: Transpose performance benchmarks")

# GPU transpose tests (require CUDA)
if (NDARRAY_HAVE_CUDA)
  add_executable(test_transpose_cuda test_transpose_cuda.cpp)
  set_source_files_properties(test_transpose_cuda.cpp PROPERTIES LANGUAGE CUDA)
  target_link_libraries(test_transpose_cuda ndarray)
  add_test(NAME transpose_cuda COMMAND test_transpose_cuda)
  message("  - test_transpose_cuda: GPU-accelerated transpose with CUDA")
endif()

message("")

# Minimal distributed transpose test (requires MPI)
if (NDARRAY_HAVE_MPI)
  add_executable(test_transpose_distributed_minimal test_transpose_distributed_minimal.cpp)
  target_link_libraries(test_transpose_distributed_minimal ndarray)
  # Disable optimizations to avoid Heisenbug in distributed transpose
  target_compile_options(test_transpose_distributed_minimal PRIVATE -O0 -g)
  message("  - test_transpose_distributed_minimal: Minimal distributed transpose debugging test (run with: mpirun -np 2 ./bin/test_transpose_distributed_minimal)")
endif()

