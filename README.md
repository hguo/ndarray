# ndarray

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Version](https://img.shields.io/badge/version-0.0.1--alpha-orange.svg)](https://github.com/hguo/ndarray/releases)

> **⚠️ Alpha Release**: This is an early pre-release version (v0.0.1). Core features are functional but the API may change. Not recommended for production use.

> **🤖 AI-Assisted Development**: Significant portions of this project's code, documentation, examples, and tests have been generated or enhanced with AI assistance (starting 2026). While functional, the code should be thoroughly reviewed and tested before use in critical applications.

NDArray is a versatile multidimensional array C++ library that provides a unified interface for working with scientific data across multiple I/O formats including NetCDF, HDF5, ADIOS2, and binary data. The library is header-only when used without external dependencies, but requires linking against third-party libraries when using features like NetCDF, HDF5, or ADIOS2.

## Features

- **Header-only core** - Easy integration for basic array operations without external dependencies
- **Modern C++17** - Clean, type-safe template-based design
- **Multiple I/O backends**:
  - NetCDF (with parallel-netcdf support)
  - HDF5
  - ADIOS2 for high-performance I/O
  - Binary data streams
  - PNG images
- **Parallel computing support**:
  - MPI for distributed-memory parallelism
  - OpenMP for shared-memory parallelism
  - CUDA support (experimental)
  - SYCL support (experimental, cross-platform acceleration)
- **Rich array operations**:
  - Flexible reshaping and slicing
  - Convolution operations
  - Gradient computation
  - Lattice partitioning
- **Scientific workflows**:
  - VTK integration for visualization
  - Henson support for in-situ analysis (experimental)
  - Synthetic data generation utilities

## Requirements

### Core Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- yaml-cpp

### Optional Dependencies
All optional dependencies can be enabled with the `NDARRAY_USE_<NAME>` CMake options:
- **ADIOS2** - High-performance I/O framework
- **HDF5** - Hierarchical data format
- **NetCDF** - Network Common Data Form
- **MPI** - Message Passing Interface
- **OpenMP** - Shared-memory parallelism
- **VTK** - Visualization toolkit
- **PNG** - Image I/O
- **CUDA** - GPU acceleration (experimental)
- **SYCL** - Cross-platform heterogeneous acceleration (experimental)

**Note**: The library is header-only when no optional dependencies are used. Basic array operations, binary I/O, and core functionality work without linking. When using optional dependencies, you must link against the corresponding external libraries.

## Installation

### Basic Installation

```bash
git clone https://github.com/hguo/ndarray.git
cd ndarray
mkdir build && cd build
cmake ..
make install
```

### Custom Installation with Optional Features

Use the flexible `NDARRAY_USE_<NAME>` options with three possible values:
- `TRUE` - Require the dependency (build fails if not found)
- `FALSE` - Disable the dependency
- `AUTO` - Use if available, continue if not found

```bash
cmake .. \
  -DNDARRAY_USE_NETCDF=AUTO \
  -DNDARRAY_USE_HDF5=AUTO \
  -DNDARRAY_USE_ADIOS2=TRUE \
  -DNDARRAY_USE_MPI=AUTO \
  -DNDARRAY_BUILD_TESTS=ON
make
make test
make install
```

### Using ndarray in Your Project

#### CMake Integration

```cmake
find_package(ndarray REQUIRED)
target_link_libraries(your_target ndarray::ndarray)
```

#### Header-Only Usage (No External Dependencies)

For basic array operations without NetCDF, HDF5, ADIOS2, etc., simply include the headers:

```cmake
include_directories(/path/to/ndarray/include)
```

```cpp
#include <ndarray/ndarray.hh>
// Use basic array operations, binary I/O, etc.
```

#### With External Dependencies

When using features that require external libraries (NetCDF, HDF5, ADIOS2, MPI, VTK, etc.), you must:

1. Build ndarray with the required dependencies enabled
2. Link against both ndarray and the external libraries:

```cmake
find_package(ndarray REQUIRED)
find_package(NetCDF REQUIRED)
find_package(HDF5 REQUIRED)

target_link_libraries(your_target
  ndarray::ndarray
  netcdf
  ${HDF5_LIBRARIES}
)
```

## Quick Start

### Basic Array Operations (Header-Only)

These operations work without external dependencies:

```cpp
#include <ndarray/ndarray.hh>

// Create a 3D array (10x20x30)
ftk::ndarray<double> arr;
arr.reshapef(10, 20, 30);

// Fill with a constant value
arr.fill(0.0);

// Access elements
arr[{5, 10, 15}] = 3.14;

// Reshape to different dimensions
arr.reshapef({100, 60});

// Get array properties
std::cout << "Dimensions: " << arr.nd() << std::endl;
std::cout << "Total size: " << arr.size() << std::endl;
```

### Reading and Writing Data (Requires External Libraries)

These operations require linking against NetCDF, HDF5, etc.:

```cpp
#include <ndarray/ndarray.hh>

// Read from NetCDF (requires linking with NetCDF library)
ftk::ndarray<float> data;
data.read_netcdf("input.nc", "temperature");

// Process data
for (size_t i = 0; i < data.size(); i++) {
    data.data()[i] *= 2.0;  // Scale values
}

// Write to HDF5 (requires linking with HDF5 library)
data.write_h5("output.h5", "scaled_temperature");
```

### Array Slicing (Header-Only)

```cpp
ftk::ndarray<double> arr;
arr.reshapef(100, 100, 50);
arr.fill(1.0);

// Extract a slice
std::vector<size_t> start = {10, 20, 0};
std::vector<size_t> size = {30, 40, 50};
auto slice = arr.slice(start, size);
```

### MPI Parallel I/O (Requires MPI and Parallel-NetCDF)

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    ftk::ndarray<double> local_data;
    // ... process local data ...

    // Parallel write using parallel-netcdf (requires MPI and PNetCDF libraries)
    local_data.write_pnetcdf("output.nc", "data", MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
```

## Examples

See the [examples](examples/) directory for complete working examples:
- `basic_usage.cpp` - Creating and manipulating arrays
- `io_operations.cpp` - Reading and writing various formats
- `parallel_mpi.cpp` - MPI parallel operations
- `convolution.cpp` - Image processing with convolution
- `sycl_acceleration.cpp` - Cross-platform GPU acceleration with SYCL
- `device_memory.cpp` - Managing data transfers between host and device

## Building Tests

```bash
cmake .. -DNDARRAY_BUILD_TESTS=ON
make
ctest
```

## Documentation

### API Reference

Key classes and functions:

- `ftk::ndarray<T>` - Main templated array class
  - `reshapef(dims...)` - Reshape array with new dimensions
  - `fill(value)` - Fill array with constant value
  - `slice(start, size)` - Extract sub-array
  - `read_netcdf()`, `write_netcdf()` - NetCDF I/O
  - `read_h5()`, `write_h5()` - HDF5 I/O
  - `read_adios2()`, `write_adios2()` - ADIOS2 I/O

- `ftk::ndarray_group` - Manage groups of related arrays
- `ftk::lattice` - Define regular grid structures
- `ftk::conv` - Convolution operations

### Configuration Options

The library automatically detects available dependencies based on CMake configuration. Check your build configuration:

```bash
cmake .. -DNDARRAY_USE_NETCDF=AUTO
# Configuration summary will show which features are enabled
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2018 Hanqi Guo

## Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/hguo/ndarray/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/hguo/ndarray/discussions)

## Acknowledgments

This library is part of the larger scientific computing ecosystem and integrates with various established formats and frameworks including NetCDF, HDF5, ADIOS2, VTK, and MPI.
