# ndarray Unit Tests

This directory contains unit tests for the ndarray library.

## Test Structure

### Core Tests (`test_ndarray_core.cpp`)
Tests fundamental array operations without requiring external dependencies:
- Array construction and initialization
- Reshaping operations
- Element access and modification
- Multi-dimensional indexing
- Filling arrays
- Array slicing
- Copy and assignment operations
- Different data types (int, float, double)
- Memory management

### I/O Tests (`test_ndarray_io.cpp`)
Tests file input/output operations:
- Binary file read/write
- Data integrity verification
- Multi-dimensional array I/O
- Large array handling
- Optional: NetCDF I/O (if compiled with NetCDF support)
- Optional: HDF5 I/O (if compiled with HDF5 support)

### Stream Tests (`test_ndarray_stream.cpp`)
Tests YAML stream functionality for reading time-series data:
- YAML configuration parsing
- Synthetic data streams (no external data files needed)
- Static vs time-varying data
- Multiple variables per stream
- Sequential and random timestep access
- Different array dimensions and data types
- Error handling for invalid timesteps
- Optional: NetCDF stream reading (if data files available)
- Optional: MPI communicator usage (if compiled with MPI)

## Building and Running Tests

### Build Tests

```bash
cd build
cmake .. -DNDARRAY_BUILD_TESTS=ON
make
```

### Run All Tests

```bash
ctest
```

Or with verbose output:
```bash
ctest --output-on-failure
```

### Run Individual Tests

```bash
./test_ndarray_core
./test_ndarray_io
./test_ndarray_stream
```

Or using ctest:
```bash
ctest -R ndarray_core
ctest -R ndarray_io
ctest -R ndarray_stream
```

## Test Configuration

Tests automatically adapt to available features:
- **NetCDF**: If `NDARRAY_USE_NETCDF` was enabled during build, NetCDF I/O tests will run
- **HDF5**: If `NDARRAY_USE_HDF5` was enabled, HDF5 I/O tests will run
- Tests gracefully skip unavailable features

## Adding New Tests

To add a new test file:

1. Create `test_new_feature.cpp` in this directory
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(test_new_feature test_new_feature.cpp)
   target_link_libraries(test_new_feature ${CMAKE_THREAD_LIBS_INIT})
   add_test(NAME new_feature COMMAND test_new_feature)
   ```
3. Follow the existing test structure using `TEST_ASSERT` and `TEST_SECTION`

## Test Output

Successful test runs show:
```
=== Running ndarray Core Tests ===
  Testing: Basic construction
    PASSED
  Testing: Vector constructor
    PASSED
...
=== All Core Tests Passed ===
```

Failed tests show detailed error messages including:
- Failed assertion message
- File and line number
- Expected vs actual values

## Continuous Integration

Tests are automatically run on every push via GitHub Actions.
See [.github/workflows/ci.yml](../.github/workflows/ci.yml) for CI configuration.

## Coverage

Current test coverage focuses on:
- ✓ Core array operations (construction, access, manipulation)
- ✓ Binary file I/O
- ✓ YAML stream functionality (synthetic and NetCDF streams)
- ✓ Time-series data reading
- ✓ Static and time-varying data
- ✓ Basic NetCDF/HDF5 operations (when available)
- Future: Convolution operations
- Future: MPI parallel operations
- Future: CUDA operations

## Troubleshooting

**Tests fail to build:**
- Ensure you have required dependencies installed
- Check CMake configuration output for missing features

**I/O tests fail:**
- Check file system permissions
- Verify sufficient disk space
- Ensure test output directory is writable

**Optional tests skipped:**
- This is normal if dependencies (NetCDF, HDF5) aren't installed
- Tests will note which features are skipped

## Legacy Tests

The old test suite (for a different project) has been backed up to `CMakeLists_old.txt`.
The new test suite focuses specifically on ndarray functionality.
