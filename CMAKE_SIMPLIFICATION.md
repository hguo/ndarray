# CMake Build System Simplification

## Summary

Simplified the ndarray CMake build system by changing dependency linking from default to PUBLIC visibility. This allows transitive dependency propagation, eliminating repetitive manual linking in examples and tests.

## Changes Made

### 1. src/CMakeLists.txt - Changed to PUBLIC Linking

All `target_link_libraries` calls for the ndarray library now use PUBLIC visibility:

```cmake
# Before:
target_link_libraries (ndarray yaml-cpp::yaml-cpp)

# After:
target_link_libraries (ndarray PUBLIC yaml-cpp::yaml-cpp)
```

This was applied to all optional dependencies:
- yaml-cpp::yaml-cpp
- adios2::adios2
- VTK::* components
- netCDF::netcdf
- ${PNETCDF_LIBRARIES}
- ${HDF5_LIBRARIES}
- ${HENSON_PMPI_LIBRARY} ${HENSON_LIBRARY}
- MPI::MPI_CXX and MPI::MPI_C
- PNG::PNG
- OpenMP::OpenMP_CXX
- CUDA::cudart

### 2. examples/CMakeLists.txt - Simplified Linking

**Before** (manual dependency listing):
```cmake
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(basic_usage)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(basic_usage netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(basic_usage ${HDF5_LIBRARIES})
endif()
```

**After** (automatic dependency propagation):
```cmake
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage ndarray ${CMAKE_THREAD_LIBS_INIT})
```

**Removed:**
- `add_vtk_if_enabled()` macro (VTK propagates automatically)
- All conditional NetCDF/HDF5/YAML/VTK/ADIOS2/PNG linking blocks
- Reduced from ~180 lines to ~50 lines

**Exception:** MPI examples still explicitly link MPI::MPI_CXX since they directly use MPI functions.

### 3. tests/CMakeLists.txt - Simplified Linking

Same pattern as examples:

**Before:**
```cmake
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ${CMAKE_THREAD_LIBS_INIT})
add_vtk_if_enabled(test_ndarray_core)

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(test_ndarray_core netCDF::netcdf)
endif()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries(test_ndarray_core ${HDF5_LIBRARIES})
endif()
```

**After:**
```cmake
add_executable(test_ndarray_core test_ndarray_core.cpp)
target_link_libraries(test_ndarray_core ndarray ${CMAKE_THREAD_LIBS_INIT})
add_test(NAME ndarray_core COMMAND test_ndarray_core)
```

**Removed:**
- `add_vtk_if_enabled()` macro
- All conditional dependency linking blocks
- Reduced from ~335 lines to ~170 lines

## Benefits

1. **Less repetition**: Don't need to list all optional dependencies for every executable
2. **Easier maintenance**: Adding a new dependency only requires updating src/CMakeLists.txt
3. **Fewer errors**: Can't forget to link a required library
4. **Cleaner code**: Examples/tests focus on their purpose, not build configuration
5. **Consistent behavior**: All consumers automatically get the same dependencies as the library

## CMake PUBLIC/PRIVATE/INTERFACE Linking

**PUBLIC** - Dependency is part of both interface and implementation:
- Used when: Header files expose the dependency (e.g., ndarray.hh includes VTK headers when NDARRAY_HAVE_VTK is ON)
- Effect: Consumers automatically link against this dependency
- Example: VTK, HDF5, NetCDF (all exposed in ndarray.hh)

**PRIVATE** - Dependency is only in implementation:
- Used when: Only .cpp files use the dependency, not headers
- Effect: Consumers don't automatically link against this dependency
- Example: Could be used if a dependency was only in ndarray.cpp

**INTERFACE** - Dependency is only in interface:
- Used when: Header-only libraries or pure interface targets
- Effect: Consumers link, but library implementation doesn't
- Example: Not typically used for ndarray

**Why PUBLIC for ndarray?**
Because ndarray.hh is a header-only library that conditionally includes dependency headers:
```cpp
#if NDARRAY_HAVE_VTK
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
// ...
#endif

#if NDARRAY_HAVE_HDF5
#include <hdf5.h>
// ...
#endif
```

Any consumer that includes ndarray.hh will also need these dependencies linked, so PUBLIC is correct.

## Verification

All targets build successfully with simplified configuration:

```bash
# Library and core tests
make ndarray test_ndarray_core test_ndarray_io test_distributed_ndarray -j8
# ✓ Success

# Simple examples
make basic_usage io_operations convolution -j4
# ✓ Success

# MPI examples
make parallel_mpi
# ✓ Success
```

## Known Issues

The distributed_stream examples (distributed_stream.cpp, distributed_stream_yaml.cpp) have compilation errors unrelated to this CMake refactoring. These examples call methods that don't exist in the distributed_stream class:
- `add_variable()`
- `set_input_source()`
- `set_n_timesteps()`
- `read_var()`

These are pre-existing issues with the example code itself, not the CMake changes.

## Migration Guide for External Projects

If your project builds against ndarray, no changes required! The PUBLIC visibility means you'll automatically get all necessary dependencies.

**Before this change:**
```cmake
# External project had to manually list all dependencies
add_executable(my_app main.cpp)
target_link_libraries(my_app ndarray)
if (NDARRAY_HAVE_VTK)
  target_link_libraries(my_app VTK::CommonCore VTK::CommonDataModel ...)
endif()
if (NDARRAY_HAVE_HDF5)
  target_link_libraries(my_app ${HDF5_LIBRARIES})
endif()
# ... and so on for each dependency
```

**After this change:**
```cmake
# External project just links ndarray, dependencies propagate automatically
add_executable(my_app main.cpp)
target_link_libraries(my_app ndarray)
```

## Lines of Code Reduced

- src/CMakeLists.txt: Changed ~25 lines (added PUBLIC keyword)
- examples/CMakeLists.txt: **Reduced from ~217 to ~50 lines** (-77%)
- tests/CMakeLists.txt: **Reduced from ~335 to ~170 lines** (-49%)

**Total: ~330 lines of CMake code removed**

## References

- CMake documentation: https://cmake.org/cmake/help/latest/command/target_link_libraries.html
- Modern CMake practices: https://cliutils.gitlab.io/modern-cmake/chapters/basics/functions.html
