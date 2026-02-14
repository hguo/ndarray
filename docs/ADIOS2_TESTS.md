## ADIOS2 Tests

This document describes the ADIOS2 test suite for ndarray.

## Overview

The ADIOS2 tests (`test_adios2.cpp`) verify I/O functionality using the ADIOS2 (Adaptable I/O System) library. ADIOS2 is a high-performance I/O framework supporting:

- **BP (Binary Pack) format**: BP4 and BP5 engines
- **Streaming**: Real-time data streaming (SST, InSituMPI)
- **Time-series**: Multi-step data for temporal datasets
- **Parallel I/O**: MPI-based parallel file operations
- **Compression**: Multiple compression methods (zlib, bzip2, blosc, etc.)

## Prerequisites

### Required Dependencies

1. **ADIOS2**: The Adaptable I/O System library
   - Version 2.7.0 or later recommended
   - Enable with: `-DNDARRAY_USE_ADIOS2=ON`
   - Install via: `spack install adios2`, `apt-get install adios2-dev`, or build from source

2. **MPI** (Optional but recommended):
   - Required for parallel I/O features
   - Enable with: `-DNDARRAY_USE_MPI=ON`
   - ADIOS2 must be compiled with MPI support

### Building with ADIOS2 Support

```bash
# Serial mode (no MPI)
cmake -DNDARRAY_USE_ADIOS2=ON ..
make test_adios2

# With MPI support
cmake -DNDARRAY_USE_ADIOS2=ON -DNDARRAY_USE_MPI=ON ..
make test_adios2
```

## Running the Tests

### Serial Mode

```bash
./bin/test_adios2
```

### Parallel Mode (with MPI)

```bash
# Run with 4 MPI processes
mpirun -np 4 ./bin/test_adios2

# Run with 1 process (same as serial)
mpirun -np 1 ./bin/test_adios2
```

## Test Coverage

### Test 1: Write and Read Single-Step BP File

- **Purpose**: Basic BP file I/O operations
- **Operations**:
  - Create 2D float array (10 x 20)
  - Write to BP4 format file
  - Read using `from_bp()` static factory method
  - Verify data integrity
- **APIs Tested**: `from_bp(filename, varname, step)`

**Example**:
```cpp
ftk::ndarray<float> data;
data.reshapef(10, 20);
// ... fill data ...

// Write (using raw ADIOS2 API)
adios2::ADIOS adios;
adios2::IO io = adios.DeclareIO("TestIO");
adios2::Engine writer = io.Open("data.bp", adios2::Mode::Write);
auto var = io.DefineVariable<float>("temperature", {20, 10}, {0, 0}, {20, 10});
writer.BeginStep();
writer.Put(var, data.data());
writer.EndStep();
writer.Close();

// Read (using ndarray API)
ftk::ndarray<float> loaded = ftk::ndarray<float>::from_bp("data.bp", "temperature", 0);
```

### Test 2: Multi-Step Time-Series Data

- **Purpose**: Verify time-series (multi-step) data handling
- **Operations**:
  - Write 5 timesteps to a single BP file
  - Read individual timesteps (step 0, 2, 4)
  - Read all timesteps at once (returns 3D array)
- **Key Feature**: Time dimension becomes the last dimension in the array
- **APIs Tested**:
  - `from_bp(filename, varname, step)` for single step
  - `from_bp(filename, varname, -1)` for all steps

**Time Series Structure**:
```
Single step:  [nx, ny]           (2D spatial)
All steps:    [nx, ny, nsteps]   (3D: spatial + time)
```

### Test 3: Multiple Data Types

- **Purpose**: Verify support for different numeric types
- **Data Types**:
  - `float`: Single-precision floating point
  - `double`: Double-precision floating point
  - `int`: 32-bit integers
- **Operations**:
  - Write all three types to a single BP file
  - Read each type independently
  - Verify correct type handling and values

### Test 4: Low-Level ADIOS2 API

- **Purpose**: Test ndarray integration with ADIOS2 Engine/IO objects
- **Operations**:
  - Create ADIOS2 engine directly
  - Use `read_bp(IO&, Engine&, varname, step)` function
  - Demonstrates advanced control over ADIOS2 operations
- **Use Case**: Custom ADIOS2 configurations (compression, engine selection)

**Example**:
```cpp
adios2::ADIOS adios;
adios2::IO io = adios.DeclareIO("CustomIO");
io.SetEngine("BP5");  // Use BP5 engine
io.SetParameter("CompressionMethod", "zlib");

adios2::Engine reader = io.Open("data.bp", adios2::Mode::Read);
ftk::ndarray<float> data;
data.read_bp(io, reader, "temperature", 0);
reader.Close();
```

### Test 5: 3D Array I/O

- **Purpose**: Verify multi-dimensional array support
- **Dimensions**: 8 x 10 x 6 (nx × ny × nz)
- **Operations**:
  - Write 3D double array
  - Read and verify all dimensions
  - Test specific element access
- **Note**: Dimensions are stored in Fortran order (column-major)

### Test 6: Convenience read_bp Function

- **Purpose**: Test simplified read_bp API
- **Function**: `read_bp(filename, varname, step, comm)`
- **Feature**: Automatically creates ADIOS2 objects internally
- **Use Case**: Quick file reading without ADIOS2 boilerplate

### Test 7: Parallel I/O with MPI (MPI only)

- **Purpose**: Test distributed parallel I/O
- **Requirements**: Multiple MPI ranks (skipped if nprocs = 1)
- **Operations**:
  - Each rank writes its portion of the array
  - Domain decomposition along Y-axis
  - Rank 0 reads the full array
  - Verifies parallel composition
- **Pattern**: SPMD (Single Program Multiple Data)

**Domain Decomposition**:
```
Global Array: [100, 80]
Rank 0: [100, 20]  (rows 0-19)
Rank 1: [100, 20]  (rows 20-39)
Rank 2: [100, 20]  (rows 40-59)
Rank 3: [100, 20]  (rows 60-79)
```

## Expected Output

### Serial Mode

```
=== Running ndarray ADIOS2 Tests ===
Running in serial mode (MPI not available)

  Testing: Write and read single-step BP file
    - Wrote BP file: 10 x 20 array
    - Read BP file using from_bp()
    - Data verification passed
    PASSED
  Testing: Multi-step time-series data
    - Wrote 5 timesteps
    - Read individual timesteps successfully
    - Read all steps: 8 x 12 x 5
    PASSED
  Testing: Multiple data types
    - Wrote and read float, double, and int arrays
    PASSED
  Testing: Low-level ADIOS2 API
    - Low-level API read_bp() works correctly
    PASSED
  Testing: 3D array I/O
    - Wrote 3D array: 8 x 10 x 6
    - Read 3D array successfully
    PASSED
  Testing: Convenience read_bp(filename, varname, step)
    - Convenience function read_bp() works
    PASSED

=== All ADIOS2 Tests Passed ===
```

### Parallel Mode (4 processes)

```
=== Running ndarray ADIOS2 Tests ===
Running with 4 MPI process(es)

  Testing: Write and read single-step BP file
    - Wrote BP file: 10 x 20 array
    - Read BP file using from_bp()
    - Data verification passed
    PASSED
  ... (same as serial for tests 1-6) ...
  Testing: Parallel I/O with MPI
    - 4 ranks wrote 100 x 80 array
    - Rank 0 read full array successfully
    PASSED

=== All ADIOS2 Tests Passed ===
```

## ADIOS2 API Reference (ndarray)

### High-Level API

```cpp
// Static factory method - read from BP file
static ndarray<T> from_bp(
  const std::string& filename,
  const std::string& varname,
  int step = -1,
  MPI_Comm comm = MPI_COMM_WORLD
);

// Convenience read function
void read_bp(
  const std::string& filename,
  const std::string& varname,
  int step = -1,
  MPI_Comm comm = MPI_COMM_WORLD
);
```

### Low-Level API (requires ADIOS2)

```cpp
#if NDARRAY_HAVE_ADIOS2
void read_bp(
  adios2::IO& io,
  adios2::Engine& reader,
  const std::string& varname,
  int step = -1
);

void read_bp(
  adios2::IO& io,
  adios2::Engine& reader,
  adios2::Variable<T>& var,
  int step = -1
);
#endif
```

### Step Parameter

- `step >= 0`: Read specific timestep
- `step = -1` (or `NDARRAY_ADIOS2_STEPS_ALL`): Read all timesteps
  - Returns 3D array: `[spatial_dims..., nsteps]`
- `step = -2` (or `NDARRAY_ADIOS2_STEPS_UNSPECIFIED`): Auto-detect (default)

## BP File Format

ADIOS2 BP (Binary Pack) files:

- **BP4**: Default, widely compatible
- **BP5**: New format, better performance for large-scale simulations
- **File Structure**: Metadata + binary data
- **Portability**: Cross-platform, endian-safe
- **Compression**: Optional (zlib, bzip2, blosc, MGARD, etc.)

### File Inspection

```bash
# Use bpls (part of ADIOS2)
bpls test_adios2_single.bp
bpls -la test_adios2_single.bp  # Long format with attributes
bpls -d test_adios2_single.bp   # Show data values

# Dump to text
bpls -d temperature test_adios2_single.bp > data.txt
```

## Engine Selection

ADIOS2 supports multiple engines:

```cpp
adios2::IO io = adios.DeclareIO("MyIO");

// File-based engines
io.SetEngine("BP4");          // Binary Pack v4 (default)
io.SetEngine("BP5");          // Binary Pack v5 (newer)
io.SetEngine("HDF5");         // HDF5 format

// Streaming engines
io.SetEngine("SST");          // Sustainable Staging Transport
io.SetEngine("InSituMPI");    // In-situ MPI
io.SetEngine("DataMan");      // WAN streaming

// Special engines
io.SetEngine("Null");         // No output (testing)
```

## Compression

Enable compression for smaller files:

```cpp
adios2::IO io = adios.DeclareIO("CompressedIO");

// Per-variable compression
auto var = io.DefineVariable<float>("data", shape, start, count);
io.SetParameter("variable", "data");
io.SetParameter("compression", "zlib");
io.SetParameter("compression-level", "9");

// Available compressors (if built with support):
// - zlib, bzip2, blosc, sz, zfp, mgard, png
```

## Performance Considerations

### Single File vs Multiple Files

**Single File (Recommended)**:
- All steps in one `.bp` file
- Efficient metadata management
- Better for analysis workflows

**Multiple Files** (Step-file format):
- Separate file per step: `data.bp.dir/data.bp.0`, `data.bp.1`, ...
- Use when appending over long runs
- Set with: `io.SetParameter("StepsPerFile", "1")`

### Buffering

```cpp
// Adjust buffer sizes
io.SetParameter("BufferGrowthFactor", "1.5");
io.SetParameter("InitialBufferSize", "100Mb");
io.SetParameter("MaxBufferSize", "2Gb");
```

### Parallel I/O

For best parallel I/O performance:
- Use BP5 engine
- Enable aggregation: `io.SetParameter("NumAggregators", "16")`
- Use collective writes: `writer.PerformPuts()` or `EndStep()`

## Troubleshooting

### ADIOS2 Not Found

```
CMake Error: Could not find ADIOS2
```

**Solution**:
```bash
# Install ADIOS2
spack install adios2+hdf5+mpi  # Using Spack
# or
git clone https://github.com/ornladios/ADIOS2
cd ADIOS2 && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make install

# Then tell CMake where to find it
cmake -DADIOS2_DIR=/usr/local/lib/cmake/adios2 ..
```

### MPI Version Mismatch

```
Error: ADIOS2 was built with different MPI
```

**Solution**: Ensure ndarray and ADIOS2 use the same MPI:
```bash
cmake -DNDARRAY_USE_MPI=ON \
      -DMPI_C_COMPILER=$(which mpicc) \
      -DMPI_CXX_COMPILER=$(which mpicxx) \
      -DADIOS2_DIR=/path/to/adios2 ..
```

### File Not Found Errors

ADIOS2 BP files are actually directories with `.bp` extension:
```bash
ls -la test_adios2_single.bp/
# Contains: data.0, metadata.0, etc.
```

Always reference as `filename.bp`, not `filename.bp/data.0`.

### Reading Old BP3 Files

BP3 format (ADIOS1) requires special handling:
```cpp
// Use legacy API for BP3 files
bool success = array.read_bp_legacy(filename, varname, MPI_COMM_WORLD);
```

## Integration with ndarray_group_stream

ADIOS2 can be used in streaming workflows:

```yaml
stream:
  name: adios2_stream
  substreams:
    - name: simulation_data
      format: adios2   # Future feature
      filenames:
        - data.bp
      vars:
        - name: temperature
        - name: velocity
```

**Note**: ADIOS2 stream support may require additional implementation.

## Comparison: ADIOS2 vs NetCDF vs HDF5

| Feature | ADIOS2 | NetCDF | HDF5 |
|---------|--------|--------|------|
| Parallel I/O | ✓✓✓ (Excellent) | ✓✓ (PNetCDF) | ✓✓ (MPI-IO) |
| Streaming | ✓✓✓ | ✗ | ✗ |
| Compression | ✓✓✓ (Many) | ✓ (Limited) | ✓✓ |
| Time-series | ✓✓✓ (Native) | ✓✓ (Unlimited dim) | ✓ (Manual) |
| File size | Smallest | Medium | Largest |
| Portability | Good | Excellent | Excellent |
| Tools | bpls, Python | ncdump, Python | h5dump, Python |

**Use ADIOS2 when**:
- You need highest I/O performance
- Working with time-series simulation data
- Streaming or in-situ analysis required
- Large-scale HPC applications

**Use NetCDF when**:
- Standards compliance required
- Climate/atmospheric science domains
- Long-term archival (CF conventions)

**Use HDF5 when**:
- Complex hierarchical data structures
- Many small datasets in one file
- Extensive metadata requirements

## Example: Complete Workflow

```cpp
#include <ndarray/ndarray.hh>
#include <adios2.h>

int main() {
  // Create simulation data
  ftk::ndarray<double> temperature;
  temperature.reshapef(256, 256, 128);

  // Fill with initial conditions
  for (size_t i = 0; i < temperature.size(); i++) {
    temperature[i] = 300.0 + rand() % 100;
  }

  // Write time series
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("SimulationIO");
  io.SetEngine("BP5");

  adios2::Engine writer = io.Open("simulation.bp", adios2::Mode::Write);
  auto var = io.DefineVariable<double>("T",
    {128, 256, 256},  // Global shape (reversed)
    {0, 0, 0},        // Offset
    {128, 256, 256}   // Local shape
  );

  // Time stepping
  for (int step = 0; step < 100; step++) {
    // ... update temperature ...

    writer.BeginStep();
    writer.Put(var, temperature.data());
    writer.EndStep();
  }
  writer.Close();

  // Analysis: Read specific timestep
  ftk::ndarray<double> snapshot = ftk::ndarray<double>::from_bp(
    "simulation.bp", "T", 50);

  // Compute statistics
  auto [min_val, max_val] = snapshot.min_max();
  std::cout << "Temperature range at t=50: ["
            << min_val << ", " << max_val << "]" << std::endl;

  return 0;
}
```

## Resources

- [ADIOS2 Documentation](https://adios2.readthedocs.io/)
- [ADIOS2 GitHub](https://github.com/ornladios/ADIOS2)
- [ADIOS2 User Guide](https://adios2.readthedocs.io/en/latest/users_guide/index.html)
- [ADIOS2 Examples](https://github.com/ornladios/ADIOS2-Examples)

## Related Documentation

- [HDF5 Tests](./VTK_TESTS.md) - HDF5 I/O tests
- [PNetCDF Tests](./PNETCDF_TESTS.md) - Parallel NetCDF tests
- [Stream Tests](../tests/test_ndarray_stream.cpp) - YAML-based streams

## See Also

- [ndarray.hh](../include/ndarray/ndarray.hh) - ADIOS2 API declarations
- [ndarray_base.hh](../include/ndarray/ndarray_base.hh) - Base ADIOS2 functionality
- [CMakeLists.txt](../CMakeLists.txt) - ADIOS2 configuration
