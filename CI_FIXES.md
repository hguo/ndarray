# CI Fixes Applied

This document summarizes the fixes applied to ensure GitHub CI passes consistently.

## Issues Fixed

### 1. Missing CHANGELOG.md
**Problem**: CI requires `CHANGELOG.md` to exist.
**Solution**: Created `CHANGELOG.md` following the Keep a Changelog format with version history.

### 2. Missing VTK Library Linking in Examples
**Problem**: When VTK support is enabled (`NDARRAY_HAVE_VTK=ON`), the ndarray headers include VTK headers. All executables that include these headers must link against VTK libraries, or the build fails with undefined symbol errors.

**Solution**:
- Added `add_vtk_if_enabled()` helper macro to `examples/CMakeLists.txt`
- Called this macro for all example executables
- This matches the pattern already used in `tests/CMakeLists.txt`

**Affected files**:
- `examples/CMakeLists.txt` - Added VTK linking to all 7 example executables

### 3. Missing Library Linking for Distributed Tests
**Problem**: `test_distributed_ndarray` and `test_ghost_exchange` use PNetCDF, NetCDF, and HDF5 functionality but weren't linking against these libraries.

**Solution**:
- Added conditional library linking for NetCDF, PNetCDF, and HDF5 to both distributed tests in `tests/CMakeLists.txt`

**Affected files**:
- `tests/CMakeLists.txt` - Added library linking for distributed tests

### 4. Invalid NetCDF Write API Call
**Problem**: `test_distributed_ndarray.cpp` called non-existent `to_netcdf(filename, varname)` method.

**Solution**:
- Replaced with proper NetCDF C API calls: `nc_create()`, `nc_def_dim()`, `nc_def_var()`, `nc_enddef()`, `to_netcdf(ncid, varid)`, `nc_close()`
- Added `#if NDARRAY_HAVE_NETCDF` guard

**Affected files**:
- `tests/test_distributed_ndarray.cpp`

### 5. lattice_partitioner Initialization
**Problem**: `lattice_partitioner` has no default constructor, causing compilation errors when used as a direct member variable.

**Solution**:
- Changed from `lattice_partitioner partitioner_` to `std::unique_ptr<lattice_partitioner> partitioner_`
- Updated all member access from `.` to `->`
- Initialize with `std::make_unique<lattice_partitioner>(global_lattice_)`

**Affected files**:
- `include/ndarray/distributed_ndarray.hh`

## CI Configuration Summary

The current CI workflow (`.github/workflows/ci.yml`) tests:

### Build Matrices
1. **build-and-test**: Ubuntu + macOS × gcc + clang with AUTO dependencies
2. **build-minimal**: Ubuntu with all optional dependencies disabled
3. **code-style**: Checks for trailing whitespace and tabs
4. **documentation**: Verifies required documentation files exist

### Key Dependencies Tested
- NetCDF (AUTO mode)
- HDF5 (AUTO mode)
- YAML-cpp (required)
- MPI (not tested in CI - requires mpirun)
- VTK (AUTO mode, not typically available)
- ADIOS2 (not tested)
- PNG (not tested)

### Tests Excluded from CI
MPI tests are compiled but not run in CI because they require `mpirun`:
- `test_distributed_ndarray` - Requires: `mpirun -np 4`
- `test_ghost_exchange` - Requires: `mpirun -np 4`
- `test_pnetcdf` - Requires: `mpirun -np 4`

These tests should be run manually before pushing.

## Build Commands for Local Testing

### Test CI Configuration Locally
```bash
# Minimal build (no optional dependencies)
cmake -B build_minimal \
  -DCMAKE_BUILD_TYPE=Release \
  -DNDARRAY_BUILD_TESTS=ON \
  -DNDARRAY_BUILD_EXAMPLES=ON \
  -DNDARRAY_USE_NETCDF=OFF \
  -DNDARRAY_USE_HDF5=OFF \
  -DNDARRAY_USE_MPI=OFF
cmake --build build_minimal -j$(nproc)
cd build_minimal && ctest --output-on-failure

# AUTO dependencies (simulates Ubuntu CI)
cmake -B build_ci \
  -DCMAKE_BUILD_TYPE=Release \
  -DNDARRAY_BUILD_TESTS=ON \
  -DNDARRAY_BUILD_EXAMPLES=ON \
  -DNDARRAY_USE_NETCDF=AUTO \
  -DNDARRAY_USE_HDF5=AUTO
cmake --build build_ci -j$(nproc)
cd build_ci && ctest --output-on-failure
```

### Test MPI Functionality (Not in CI)
```bash
cmake -B build_mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DNDARRAY_BUILD_TESTS=ON \
  -DNDARRAY_USE_MPI=ON \
  -DMPI_CXX_COMPILER=$(which mpicxx)

cmake --build build_mpi -j$(nproc)

# Run MPI tests manually
cd build_mpi
mpirun -np 4 ./bin/test_distributed_ndarray
mpirun -np 4 ./bin/test_ghost_exchange
```

## Checklist Before Pushing

- [ ] Build succeeds with minimal dependencies (`NDARRAY_USE_*=OFF`)
- [ ] Build succeeds with AUTO dependencies
- [ ] All non-MPI tests pass (`ctest`)
- [ ] MPI tests compile (even if not run)
- [ ] No trailing whitespace in modified files
- [ ] No tabs in source files (use spaces)
- [ ] CHANGELOG.md updated if adding features
- [ ] Documentation files present (README.md, CONTRIBUTING.md, LICENSE, CHANGELOG.md)

## Common Pitfalls

### 1. Forgetting to Link Libraries
**Symptom**: Undefined symbol errors during linking
**Solution**: If a file includes `ndarray.hh` and optional backends are enabled, link the corresponding libraries:
```cmake
if (NDARRAY_HAVE_HDF5)
  target_link_libraries(target_name ${HDF5_LIBRARIES})
endif()
if (NDARRAY_HAVE_NETCDF)
  target_link_libraries(target_name netCDF::netcdf)
endif()
add_vtk_if_enabled(target_name)  # For VTK
```

### 2. Using Non-Existent Convenience Methods
**Symptom**: "No matching member function" errors
**Solution**: Check the actual API in header files. Many "convenience" methods don't exist and you need to use the C API directly.

### 3. Template Instantiation Issues
**Symptom**: "Expected class name" or template errors
**Solution**: Make sure all template parameters are specified, especially `<StoragePolicy>` for stream classes.

### 4. MPI Tests in CI
**Symptom**: MPI tests fail or hang in CI
**Solution**: Don't run MPI tests in CI. Compile them to verify they build, but exclude from `ctest` with `-E "(distributed|ghost|pnetcdf)"`.

## Future Improvements

1. **Add MPI Testing**: Configure CI runner with MPI support to run distributed tests
2. **Code Coverage**: Add code coverage reporting (e.g., codecov)
3. **Static Analysis**: Add clang-tidy or cppcheck
4. **Documentation Building**: Test that documentation can be generated
5. **Performance Regression**: Track benchmark results over time
