# ndarray Examples

This directory contains example programs demonstrating various features of the ndarray library.

## Building Examples

Examples are built automatically when you build the main project:

```bash
cd build
cmake .. -DNDARRAY_BUILD_EXAMPLES=ON
make
```

The example executables will be located in `build/examples/`.

## Running Examples

### Basic Usage
Demonstrates core array operations without any optional dependencies:
```bash
./basic_usage
```

**Covers:**
- Creating 1D, 2D, and 3D arrays
- Reshaping arrays
- Filling arrays with data
- Accessing elements
- Array slicing

### I/O Operations
Shows reading and writing data in various formats:
```bash
./io_operations
```

**Covers:**
- Binary file I/O
- NetCDF I/O (if enabled)
- HDF5 I/O (if enabled)
- Data integrity verification
- Working with different data types

**Requirements:**
- Optional: NetCDF library (compile with `-DNDARRAY_USE_NETCDF=ON`)
- Optional: HDF5 library (compile with `-DNDARRAY_USE_HDF5=ON`)

### Convolution
Demonstrates image processing and convolution operations:
```bash
./convolution
```

**Covers:**
- Creating convolution kernels
- Box blur (smoothing)
- Edge detection (Sobel, Laplacian)
- Gaussian blur
- Manual convolution computation

### Parallel MPI
Shows distributed computing with MPI:
```bash
mpirun -np 4 ./parallel_mpi
```

**Covers:**
- Distributed array processing
- Local computations on each rank
- Global reductions
- Parallel I/O
- Domain decomposition

**Requirements:**
- MPI library (compile with `-DNDARRAY_USE_MPI=ON`)
- Optional: Parallel-NetCDF (compile with `-DNDARRAY_USE_PNETCDF=ON`)

### SYCL Acceleration
Demonstrates cross-platform heterogeneous acceleration:
```bash
./sycl_acceleration
```

**Covers:**
- SYCL for portable GPU/accelerator programming
- Vector addition on accelerators
- Vector scaling operations
- Performance comparison (CPU vs accelerator)
- Device information querying

**Requirements:**
- SYCL implementation (compile with `-DNDARRAY_USE_SYCL=ON`)
- Supported implementations:
  - Intel DPC++ (for Intel GPUs and CPUs)
  - hipSYCL (for AMD and NVIDIA GPUs)
  - ComputeCpp (Codeplay)

**Note:** SYCL provides portable performance across Intel, AMD, and NVIDIA GPUs using the same code.

## Example Output Files

Running these examples will create several output files:

From `io_operations`:
- `output_data.bin` - Binary array data
- `float_data.bin` - Float array data
- `int_data.bin` - Integer array data
- `output_data.nc` - NetCDF file (if enabled)
- `output_data.h5` - HDF5 file (if enabled)

From `parallel_mpi`:
- `rank_0_data.bin`, `rank_1_data.bin`, etc. - Per-rank output files
- `parallel_output.nc` - Parallel NetCDF file (if enabled)

## Modifying Examples

Feel free to modify these examples to experiment with the library:

1. Change array dimensions
2. Try different data types
3. Adjust convolution kernels
4. Experiment with different MPI process counts

## Compiling Individual Examples

You can compile examples individually:

```bash
# Basic example (no dependencies)
g++ -std=c++17 -I../include basic_usage.cpp -o basic_usage

# With NetCDF
g++ -std=c++17 -I../include io_operations.cpp -o io_operations -lnetcdf

# With MPI
mpic++ -std=c++17 -I../include parallel_mpi.cpp -o parallel_mpi
```

## Next Steps

After exploring these examples:

1. Check the [API documentation](../README.md#api-reference)
2. Review the [header files](../include/ndarray/) for more functions
3. Look at the [tests](../tests/) for additional usage patterns
4. Read [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute your own examples

## Questions or Issues?

If you encounter problems with any examples:
- Check that you have the required dependencies installed
- Verify your CMake configuration enables the necessary features
- Open an issue on GitHub with details about your setup and the problem
