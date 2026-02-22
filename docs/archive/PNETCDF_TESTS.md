# Parallel NetCDF (PNetCDF) Tests

This document describes the PNetCDF test suite for ndarray.

## Overview

The PNetCDF tests (`test_pnetcdf.cpp`) verify parallel I/O functionality using the Parallel NetCDF library. These tests require both MPI and PNetCDF support to be enabled.

**Important**: PNetCDF support in ndarray is marked as **experimental** and some functions may be declared but not yet fully implemented.

## Prerequisites

### Required Dependencies

1. **MPI**: Message Passing Interface for parallel execution
   - OpenMPI, MPICH, or compatible MPI implementation
   - Enable with: `-DNDARRAY_USE_MPI=ON`

2. **PNetCDF**: Parallel NetCDF library
   - Version 1.12.0 or later recommended
   - Enable with: `-DNDARRAY_USE_PNETCDF=ON`

### Building with PNetCDF Support

```bash
cmake -DNDARRAY_USE_MPI=ON -DNDARRAY_USE_PNETCDF=ON ..
make test_pnetcdf
```

## Running the Tests

PNetCDF tests must be run with `mpirun` or `mpiexec`:

```bash
# Run with 4 MPI processes
mpirun -np 4 ./bin/test_pnetcdf

# Run with 2 processes
mpirun -np 2 ./bin/test_pnetcdf

# Run with 8 processes
mpirun -np 8 ./bin/test_pnetcdf
```

The tests are designed to work with any number of MPI processes.

## Test Coverage

### Test 1: Create Parallel NetCDF File

- **Purpose**: Verify collective file creation
- **Operations**:
  - Multiple MPI ranks create a single NetCDF file collectively
  - Define dimensions and variables
  - Each rank writes its portion of the data
- **Variables**:
  - `temperature`: 2D float array
  - `pressure`: 2D double array
- **I/O Mode**: Collective write using `ncmpi_put_vara_*_all()`

### Test 2: Parallel Read with read_pnetcdf_all

- **Purpose**: Test distributed reading with ndarray API
- **Operations**:
  - Each rank reads its portion of the array
  - Uses `ndarray::read_pnetcdf_all()` function
  - Verifies data integrity
- **Domain Decomposition**: Y-axis partitioning
- **Note**: This test may be skipped if `read_pnetcdf_all()` is not yet implemented

**Example**:
```cpp
ftk::ndarray<float> temp;
temp.reshapef(nx, local_ny);  // Local portion

MPI_Offset start[2] = {start_y, 0};
MPI_Offset count[2] = {local_ny, nx};

temp.read_pnetcdf_all(ncid, varid, start, count);
```

### Test 3: Read Entire Dataset Collectively

- **Purpose**: Verify all ranks can read the full dataset
- **Operations**:
  - All MPI ranks read the entire array
  - Each rank gets a complete copy
- **Use Case**: Small datasets or broadcast patterns

### Test 4: 3D Array Parallel I/O

- **Purpose**: Test multi-dimensional parallel I/O
- **Operations**:
  - Create and write 3D array (`[nz][ny][nx]`)
  - Each rank handles a subset of Z-slices
  - Domain decomposition along Z-axis
- **Dimensions**: 50 x 40 x 30

**Example**:
```cpp
// Rank 0: handles z = [0, nz/nprocs)
// Rank 1: handles z = [nz/nprocs, 2*nz/nprocs)
// ...
```

### Test 5: Independent I/O Mode

- **Purpose**: Test non-collective I/O operations
- **Operations**:
  - Switch between collective and independent modes
  - Uses `ncmpi_begin_indep_data()` and `ncmpi_end_indep_data()`
  - Rank 0 writes independently
- **Use Case**: Metadata writes, irregular access patterns

## Domain Decomposition Patterns

The tests demonstrate common parallel I/O patterns:

### 1D Decomposition (Y-axis)
```
Global Array: [nx, ny]
Rank 0: [nx, ny/nprocs]
Rank 1: [nx, ny/nprocs]
...
Rank n-1: [nx, remaining rows]
```

### 1D Decomposition (Z-axis, 3D)
```
Global Array: [nx, ny, nz]
Rank 0: [nx, ny, nz/nprocs]
Rank 1: [nx, ny, nz/nprocs]
...
```

## PNetCDF API Functions Used

### File Operations
- `ncmpi_create()`: Create file collectively
- `ncmpi_open()`: Open file for parallel access
- `ncmpi_close()`: Close file collectively

### Define Mode
- `ncmpi_def_dim()`: Define dimensions
- `ncmpi_def_var()`: Define variables
- `ncmpi_enddef()`: Exit define mode

### Data Mode
- `ncmpi_put_vara_*_all()`: Collective write
- `ncmpi_get_vara_*_all()`: Collective read
- `ncmpi_begin_indep_data()`: Begin independent I/O
- `ncmpi_end_indep_data()`: End independent I/O

### ndarray API (Experimental)
- `read_pnetcdf_all(ncid, varid, start, count)`: Read array portion

## Expected Output

When running with 4 processes:

```
=== Running ndarray PNetCDF Tests ===
Running with 4 MPI processes

  Testing: Create parallel NetCDF file
    - Created parallel NetCDF file: 100 x 80
    - Variables: temperature (float), pressure (double)
    PASSED
  Testing: Parallel read with read_pnetcdf_all
    - Each rank read 100 x 20 array
    - Data verification passed
    PASSED
  Testing: Read entire dataset collectively
    - All ranks read full 100 x 80 array
    PASSED
  Testing: 3D array parallel I/O
    - Created 3D array: 50 x 40 x 30
    - Each rank wrote 7 z-slices
    PASSED
  Testing: Independent I/O operations
    - Independent I/O mode tested
    PASSED

=== PNetCDF Tests Completed ===
```

## Skipped Tests

If MPI or PNetCDF is not enabled:

```
=== PNetCDF Tests ===
SKIPPED: MPI support not enabled
  Enable with: -DNDARRAY_USE_MPI=ON
SKIPPED: PNetCDF support not enabled
  Enable with: -DNDARRAY_USE_PNETCDF=ON
  NOTE: PNetCDF is marked as experimental
```

If `read_pnetcdf_all()` is not implemented:

```
  Testing: Parallel read with read_pnetcdf_all
    - NOTE: read_pnetcdf_all not yet implemented
    - This is expected (marked experimental)
    SKIPPED
```

## Implementation Status

### Implemented

- PNetCDF header inclusion (`#include <pnetcdf.h>`)
- CMake detection and configuration
- Test file creation with parallel writes
- Raw PNetCDF API usage (collective and independent modes)

### Experimental / Pending

- `ndarray::read_pnetcdf_all()`: Declared but may not be fully implemented
- `ndarray::write_pnetcdf()`: Mentioned in examples but not declared in header
- Automatic domain decomposition utilities

## Performance Considerations

### Collective vs Independent I/O

**Collective I/O** (`_all` functions):
- All ranks participate in the operation
- Enables MPI-IO optimizations
- Better performance for large datasets
- Use for regular, structured data access

**Independent I/O**:
- Ranks operate independently
- Useful for metadata or irregular access
- Less efficient for large data volumes
- Use sparingly, between `begin_indep_data()` / `end_indep_data()` calls

### File Formats

PNetCDF supports multiple NetCDF formats:
- **NC_64BIT_DATA** (CDF-5): Recommended for parallel I/O
  - Supports large variables (>4GB)
  - Enables 8-byte integers
- **NC_64BIT_OFFSET** (CDF-2): Classic format extension
- **NC_CLOBBER**: Overwrite existing file

## Debugging Tips

### MPI Errors

If you encounter MPI errors:
```bash
# Check MPI installation
mpirun --version

# Use verbose mode
mpirun -np 4 --verbose ./bin/test_pnetcdf

# Check process binding
mpirun -np 4 --report-bindings ./bin/test_pnetcdf
```

### PNetCDF Errors

Enable PNetCDF debug output:
```cpp
// In code:
ncmpi_set_default_format(NC_FORMAT_CDF5, NULL);
```

Check error codes:
```cpp
int retval;
if ((retval = ncmpi_open(...))) {
  std::cerr << "Error: " << ncmpi_strerror(retval) << std::endl;
}
```

### File Inspection

Inspect generated files:
```bash
# Use ncdump (serial NetCDF tool works on PNetCDF files)
ncdump test_pnetcdf_basic.nc

# Check dimensions
ncdump -h test_pnetcdf_basic.nc

# View data
ncdump -v temperature test_pnetcdf_basic.nc
```

## Related Documentation

- [NetCDF Tests](./VTK_TESTS.md) - Serial NetCDF tests
- [Parallel MPI Example](../examples/parallel_mpi.cpp) - MPI usage patterns
- [Stream Tests](../tests/test_ndarray_stream.cpp) - NetCDF streaming
- [PNetCDF Documentation](https://parallel-netcdf.github.io/)

## Future Work

Potential enhancements for PNetCDF support:

1. **Complete Implementation**:
   - Finish `read_pnetcdf_all()` implementation
   - Add `write_pnetcdf()` and `write_pnetcdf_all()` functions
   - Add `read_pnetcdf()` for independent reads

2. **Advanced Features**:
   - Non-blocking I/O (`_iput`, `_iget` functions)
   - Varm (strided) access patterns
   - Flexible API with automatic type conversion

3. **Integration**:
   - Add PNetCDF backend to stream functionality
   - Support for PNetCDF in ndarray_group
   - Automatic domain decomposition utilities

4. **Testing**:
   - Add benchmarks for scalability testing
   - Test with large-scale data (>100GB)
   - Weak and strong scaling tests
   - Comparison with HDF5 parallel I/O

## Example: Complete Parallel I/O Workflow

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>
#include <pnetcdf.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Domain decomposition
  const size_t global_nx = 1000, global_ny = 800;
  size_t local_ny = global_ny / nprocs;
  size_t start_y = rank * local_ny;

  // Create local data
  ftk::ndarray<double> local_data;
  local_data.reshapef(global_nx, local_ny);

  // Fill with rank-specific data
  for (size_t j = 0; j < local_ny; j++) {
    for (size_t i = 0; i < global_nx; i++) {
      local_data.f(i, j) = rank * 1000.0 + (start_y + j) * global_nx + i;
    }
  }

  // Write parallel NetCDF file
  int ncid, dimids[2], varid;
  ncmpi_create(MPI_COMM_WORLD, "output.nc", NC_CLOBBER | NC_64BIT_DATA,
               MPI_INFO_NULL, &ncid);
  ncmpi_def_dim(ncid, "x", global_nx, &dimids[1]);
  ncmpi_def_dim(ncid, "y", global_ny, &dimids[0]);
  ncmpi_def_var(ncid, "data", NC_DOUBLE, 2, dimids, &varid);
  ncmpi_enddef(ncid);

  MPI_Offset start[2] = {(MPI_Offset)start_y, 0};
  MPI_Offset count[2] = {(MPI_Offset)local_ny, (MPI_Offset)global_nx};
  ncmpi_put_vara_double_all(ncid, varid, start, count, local_data.data());

  ncmpi_close(ncid);

  MPI_Finalize();
  return 0;
}
```

## See Also

- [CMakeLists.txt](../CMakeLists.txt) - PNetCDF configuration
- [config.hh.in](../include/ndarray/config.hh.in) - Build configuration
- [ndarray.hh](../include/ndarray/ndarray.hh) - PNetCDF API declarations
