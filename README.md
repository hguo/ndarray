# ndarray

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![Version](https://img.shields.io/badge/version-0.0.1--alpha-orange.svg)](https://github.com/hguo/ndarray/releases)

> **⚠️ Alpha Release**: This is an early pre-release version (v0.0.1). Core features are functional but the API may change. Not recommended for production use.

> **🤖 AI-Assisted Development**: Significant portions of this project's code, documentation, examples, and tests have been generated or enhanced with AI assistance (starting 2026). While functional, the code should be thoroughly reviewed and tested before use in critical applications.

NDArray is a **unified I/O abstraction library for time-varying scientific data** designed for HPC systems. It provides a consistent C++ interface for reading and writing multidimensional arrays across diverse scientific data formats (NetCDF, HDF5, ADIOS2, VTK), eliminating the need to learn multiple I/O APIs. The library is header-only when used without external dependencies, but requires linking against third-party libraries when using features like NetCDF, HDF5, or ADIOS2.

## Key Features

### Time-Varying Scientific Data Abstraction
- **Unified API** for multiple scientific data formats - write once, support all formats
- **Stream-based interface** for time-series data workflows
- **Format interoperability** - read NetCDF, write HDF5 seamlessly
- **Zero-copy optimization** for efficient memory usage with large datasets

### HPC I/O Backends
- **NetCDF** - Climate, ocean, atmosphere data (with parallel-netcdf support)
- **HDF5** - General-purpose hierarchical data
- **ADIOS2** - Extreme-scale parallel I/O
- **VTK** - Visualization data formats
- **Binary streams** - Custom binary formats
- **PNG** - Image data

### Parallel Computing Integration
- **MPI** - Distributed-memory parallel I/O (collective writes)
- **Parallel-NetCDF (PNetCDF)** - High-performance parallel NetCDF
- **OpenMP** - Shared-memory parallelism
- **CUDA** - GPU acceleration (experimental)
- **SYCL** - Cross-platform heterogeneous acceleration (experimental)

### Backend Interoperability
- **Eigen integration** - Seamless conversion to/from Eigen matrices for linear algebra
- **xtensor integration** - NumPy-like API with zero-copy views
- Both backends optional, enabled via CMake

### Array Operations
- Modern C++17 template-based design
- Flexible reshaping and slicing
- Fortran and C-order indexing support
- Exception-based error handling
- Multi-component array support

## Requirements

### Core Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- yaml-cpp

### Optional Dependencies
All optional dependencies can be enabled with the `NDARRAY_USE_<NAME>` CMake options:

**I/O Libraries**:
- **ADIOS2** - High-performance I/O framework
- **HDF5** - Hierarchical data format
- **NetCDF** - Network Common Data Form
- **PNetCDF** - Parallel-NetCDF
- **VTK** - Visualization toolkit
- **PNG** - Image I/O

**Parallel Computing**:
- **MPI** - Message Passing Interface
- **OpenMP** - Shared-memory parallelism
- **CUDA** - GPU acceleration (experimental)
- **SYCL** - Cross-platform heterogeneous acceleration (experimental)

**Backend Interoperability**:
- **Eigen** - Linear algebra backend for conversion/interop
- **xtensor** - NumPy-like tensor backend for conversion/interop

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

### Time-Varying Scientific Data Workflow (YAML-based Streams)

This is the primary use case - **abstracting time-varying data as streams using YAML configuration files**:

**1. Define stream in YAML** (`config.yaml`):
```yaml
stream:
  path_prefix: /path/to/data
  substreams:
    - name: input
      format: netcdf
      filenames: "simulation_*.nc"
      vars:
        - name: temperature
          possible_names: [temperature, temp, T]
        - name: pressure
          possible_names: [pressure, press, P]
```

**2. Read and process in C++**:
```cpp
#include <ndarray/ndarray_group_stream.hh>

int main() {
    // Parse YAML configuration
    ftk::stream s;
    s.parse_yaml("config.yaml");

    // Process each timestep
    for (int t = 0; t < s.total_timesteps(); t++) {
        auto g = s.read(t);

        // Zero-copy access to variables (no unnecessary allocations)
        const auto& temperature = g->get_ref<float>("temperature");
        const auto& pressure = g->get_ref<float>("pressure");

        // Process data...
        // (your analysis code here)
    }

    return 0;
}
```

**Key benefits**:
- **Configuration-driven** - Change formats, file paths, variable names in YAML without recompiling
- **Format independence** - Switch between NetCDF, HDF5, ADIOS2, VTK by changing one line in YAML
- **Variable aliasing** - Handle different naming conventions (`temperature` vs `temp` vs `T`)
- **Zero-copy access** - `get_ref()` avoids memory allocation overhead
- **Multi-file support** - Automatically handles datasets split across multiple files

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

### Backend Interoperability (Eigen/xtensor)

Seamlessly convert between ndarray and Eigen/xtensor for advanced computations:

```cpp
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_eigen.hh>
#include <Eigen/Dense>

int main() {
    // 1. Read data with ndarray
    ftk::ndarray<double> temperature;
    temperature.read_netcdf("climate.nc", "temperature");

    // 2. Convert to Eigen for linear algebra
    auto mat = ftk::ndarray_to_eigen(temperature);

    // 3. Perform Eigen operations
    Eigen::VectorXd col_means = mat.colwise().mean();

    // 4. Convert back to ndarray
    auto result = ftk::eigen_vector_to_ndarray(col_means);

    // 5. Write to different format
    result.write_h5("means.h5", "column_means");

    return 0;
}
```

See [docs/BACKENDS.md](docs/BACKENDS.md) for xtensor integration and zero-copy views.

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

### User Guides

- **[ARRAY_ACCESS.md](docs/ARRAY_ACCESS.md)** - Dimension queries and element access (`dimf`, `dimc`, `at`, `f`, `c`)
- **[ZERO_COPY_OPTIMIZATION.md](docs/ZERO_COPY_OPTIMIZATION.md)** - Using `get_ref()` for efficient memory access
- **[BACKENDS.md](docs/BACKENDS.md)** - Eigen and xtensor integration for interoperability
- **[EXCEPTION_HANDLING.md](docs/EXCEPTION_HANDLING.md)** - Exception-based error handling guide
- **[FDPOOL.md](docs/FDPOOL.md)** - NetCDF file descriptor pool (prevents double-opening)

### API Reference

Key classes and functions:

- `ftk::ndarray<T>` - Main templated array class
  - `reshapef(dims...)` - Reshape array with new dimensions (Fortran order)
  - `get_ref<T>(key)` - Zero-copy reference access (for `ndarray_group`)
  - `slice(start, size)` - Extract sub-array
  - `read_netcdf()`, `write_netcdf()` - NetCDF I/O
  - `read_h5()`, `write_h5()` - HDF5 I/O
  - `read_bp()`, `write_bp()` - ADIOS2 I/O
  - `read_vtk_image_data_file()` - VTK I/O

- `ftk::ndarray_group` - Manage groups of related arrays (for time-series data)
- `ftk::stream` - Stream interface for time-varying datasets
- `ftk::lattice` - Define regular grid structures

### Configuration Options

The library automatically detects available dependencies based on CMake configuration. Check your build configuration:

```bash
cmake .. \
  -DNDARRAY_USE_NETCDF=AUTO \
  -DNDARRAY_USE_EIGEN=AUTO \
  -DNDARRAY_USE_XTENSOR=AUTO
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
