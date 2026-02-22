# Getting Started with ndarray

**Version**: 0.0.3
**Last Updated**: 2026-02-21

This guide will get you up and running with ndarray in 15 minutes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Your First Program](#your-first-program)
4. [Basic Operations](#basic-operations)
5. [File I/O](#file-io)
6. [Working with Time Series](#working-with-time-series)
7. [Parallel Computing (MPI)](#parallel-computing-mpi)
8. [Next Steps](#next-steps)

---

## Quick Start

### Minimal Example (Header-Only)

```cpp
#include <ndarray/ndarray.hh>
#include <iostream>

int main() {
  // Create a 3D array (100 x 200 x 50)
  ftk::ndarray<float> arr;
  arr.reshapef(100, 200, 50);

  // Fill with data
  for (size_t k = 0; k < 50; k++) {
    for (size_t j = 0; j < 200; j++) {
      for (size_t i = 0; i < 100; i++) {
        arr.f(i, j, k) = i + j * 0.1 + k * 0.01;
      }
    }
  }

  // Access data
  std::cout << "Value at (10, 20, 5): " << arr.f(10, 20, 5) << std::endl;
  std::cout << "Total elements: " << arr.size() << std::endl;

  return 0;
}
```

**Compile**:
```bash
g++ -std=c++17 -I /path/to/ndarray/include example.cpp -o example
./example
```

---

## Installation

### Option 1: Header-Only (No External Dependencies)

For basic array operations and binary I/O:

```bash
git clone https://github.com/hguo/ndarray.git
cd ndarray

# Just copy the headers
sudo cp -r include/ndarray /usr/local/include/
```

Use in your code:
```cpp
#include <ndarray/ndarray.hh>  // Core functionality
```

### Option 2: With I/O Libraries (Recommended)

For NetCDF, HDF5, ADIOS2 support:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install \
  cmake \
  libhdf5-dev \
  libnetcdf-dev \
  libyaml-cpp-dev

# Build and install
git clone https://github.com/hguo/ndarray.git
cd ndarray
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNDARRAY_USE_HDF5=ON \
  -DNDARRAY_USE_NETCDF=ON \
  -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
sudo make install
```

### Option 3: With MPI for Parallel I/O

```bash
# Install MPI and parallel I/O libraries
sudo apt-get install \
  cmake \
  mpich \
  libhdf5-mpi-dev \
  libnetcdf-dev \
  libyaml-cpp-dev

# Build with MPI support
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNDARRAY_USE_MPI=ON \
  -DNDARRAY_USE_HDF5=ON \
  -DNDARRAY_USE_NETCDF=ON

make -j$(nproc)
sudo make install
```

### Using in Your CMake Project

```cmake
find_package(ndarray REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp ndarray)
```

Or with FetchContent:
```cmake
include(FetchContent)
FetchContent_Declare(
  ndarray
  GIT_REPOSITORY https://github.com/hguo/ndarray.git
  GIT_TAG        v0.0.3
)
FetchContent_MakeAvailable(ndarray)

add_executable(myapp main.cpp)
target_link_libraries(myapp ndarray)
```

---

## Your First Program

### Creating and Accessing Arrays

```cpp
#include <ndarray/ndarray.hh>
#include <iostream>

int main() {
  // 1. Create a 2D array (10 x 20)
  ftk::ndarray<double> temperature;
  temperature.reshapef(10, 20);  // Fortran order (column-major)

  // 2. Fill with data
  for (size_t j = 0; j < 20; j++) {
    for (size_t i = 0; i < 10; i++) {
      temperature.f(i, j) = 20.0 + i * 0.5 + j * 0.2;
    }
  }

  // 3. Access individual elements
  std::cout << "Temperature at (5, 10): " << temperature.f(5, 10) << " °C\n";

  // 4. Get array info
  std::cout << "Dimensions: " << temperature.dimf(0) << " x "
            << temperature.dimf(1) << "\n";
  std::cout << "Total elements: " << temperature.size() << "\n";

  // 5. Direct pointer access (for performance)
  double* data = temperature.data();
  std::cout << "First element: " << data[0] << "\n";

  return 0;
}
```

**Key Concepts**:
- `.reshapef()` - Create array in **Fortran order** (first index varies fastest)
- `.f(i, j, k)` - Access with **Fortran indexing**
- `.dimf(n)` - Get dimension size
- `.data()` - Get raw pointer for performance-critical code

---

## Basic Operations

### Reshaping

```cpp
ftk::ndarray<float> arr;

// 1D array (1000 elements)
arr.reshapef(1000);

// 2D array (100 x 200)
arr.reshapef(100, 200);

// 3D array (50 x 60 x 70)
arr.reshapef(50, 60, 70);

// From vector
arr.reshapef({100, 200, 300});
```

### Filling with Data

```cpp
ftk::ndarray<float> arr;
arr.reshapef(100, 200);

// Method 1: Element-by-element
for (size_t j = 0; j < 200; j++) {
  for (size_t i = 0; i < 100; i++) {
    arr.f(i, j) = i * j;
  }
}

// Method 2: Direct pointer access (faster)
float* ptr = arr.data();
for (size_t i = 0; i < arr.size(); i++) {
  ptr[i] = i * 0.5;
}

// Method 3: Fill with constant
std::fill(arr.data(), arr.data() + arr.size(), 42.0f);
```

### C vs Fortran Indexing

```cpp
ftk::ndarray<int> arr;
arr.reshapef(3, 4);  // 3 rows, 4 columns

// Fortran order (column-major): first index varies fastest
arr.f(0, 0) = 1;  arr.f(1, 0) = 2;  arr.f(2, 0) = 3;
arr.f(0, 1) = 4;  arr.f(1, 1) = 5;  // ...

// C order (row-major): last index varies fastest
arr.c(0, 0) = arr.f(0, 0);  // Same element, different indexing
arr.c(0, 1) = arr.f(1, 0);  // Maps to different element!

// Memory layout (Fortran order):
// [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
//  ^^^^^^^  -------  -------  -----------
//  column0  column1  column2    column3
```

**Rule of thumb**: Use **Fortran indexing** (`.f()`) for scientific computing compatibility (MATLAB, Fortran, NumPy 'F' order).

---

## File I/O

### HDF5 Example

```cpp
#include <ndarray/ndarray.hh>

int main() {
  // Write to HDF5
  {
    ftk::ndarray<float> temperature;
    temperature.reshapef(100, 200, 50);

    // Fill with data...
    for (size_t k = 0; k < 50; k++) {
      for (size_t j = 0; j < 200; j++) {
        for (size_t i = 0; i < 100; i++) {
          temperature.f(i, j, k) = 20.0 + i * 0.1;
        }
      }
    }

    // Write to file
    temperature.to_h5("temperature.h5", "temperature");
    std::cout << "Wrote temperature.h5\n";
  }

  // Read from HDF5
  {
    ftk::ndarray<float> loaded;
    loaded.read_h5("temperature.h5", "temperature");

    std::cout << "Loaded array: " << loaded.dimf(0) << " x "
              << loaded.dimf(1) << " x " << loaded.dimf(2) << "\n";
    std::cout << "Value at (10, 20, 5): " << loaded.f(10, 20, 5) << "\n";
  }

  return 0;
}
```

**Compile with HDF5**:
```bash
g++ -std=c++17 -I include example.cpp -lndarray -lhdf5 -o example
```

### NetCDF Example

```cpp
#include <ndarray/ndarray.hh>

int main() {
  ftk::ndarray<double> data;
  data.reshapef(100, 200);

  // Fill with data...
  for (size_t j = 0; j < 200; j++) {
    for (size_t i = 0; i < 100; i++) {
      data.f(i, j) = i + j * 100;
    }
  }

  // Write to NetCDF
  data.to_nc("output.nc", "field");

  // Read from NetCDF
  ftk::ndarray<double> loaded;
  loaded.read_nc("output.nc", "field");

  std::cout << "NetCDF dimensions: " << loaded.dimf(0) << " x "
            << loaded.dimf(1) << "\n";

  return 0;
}
```

### ADIOS2 Example (High-Performance I/O)

```cpp
#include <ndarray/ndarray.hh>

int main() {
  ftk::ndarray<float> velocity;
  velocity.reshapef(1000, 2000, 100);

  // Fill with data...

  // Write to ADIOS2 BP format (fast parallel I/O)
  velocity.to_bp("velocity.bp", "velocity");

  // Read back
  ftk::ndarray<float> loaded = ftk::ndarray<float>::from_bp(
    "velocity.bp", "velocity", 0);  // timestep 0

  return 0;
}
```

### Binary I/O (Portable)

```cpp
#include <ndarray/ndarray.hh>

int main() {
  ftk::ndarray<double> data;
  data.reshapef(100, 200);

  // Fill with data...

  // Write binary
  data.to_binary("data.bin");

  // Read binary
  ftk::ndarray<double> loaded;
  loaded.from_binary("data.bin");

  return 0;
}
```

---

## Working with Time Series

### Reading Time Series from Files

```cpp
#include <ndarray/ndarray_group_stream.hh>

int main() {
  // Setup stream from YAML configuration
  ftk::stream stream;
  stream.parse_yaml("simulation.yaml");

  std::cout << "Total timesteps: " << stream.total_timesteps() << "\n";

  // Read each timestep
  for (int t = 0; t < stream.total_timesteps(); t++) {
    auto group = stream.read(t);

    // Access variables
    auto temp = group->get_arr<float>("temperature");
    auto pres = group->get_arr<float>("pressure");

    std::cout << "Timestep " << t << ": temp[0] = " << temp[0]
              << ", pres[0] = " << pres[0] << "\n";
  }

  return 0;
}
```

**YAML Configuration** (`simulation.yaml`):
```yaml
stream:
  name: simulation
  substreams:
    - name: netcdf_data
      format: nc
      filenames:
        - sim_000.nc
        - sim_001.nc
        - sim_002.nc
      vars:
        - name: temperature
          nc_name: temp
        - name: pressure
          nc_name: pres
```

### Writing Time Series

```cpp
#include <ndarray/ndarray.hh>

int main() {
  const int num_timesteps = 10;

  for (int t = 0; t < num_timesteps; t++) {
    ftk::ndarray<float> data;
    data.reshapef(100, 200);

    // Generate data for this timestep
    for (size_t j = 0; j < 200; j++) {
      for (size_t i = 0; i < 100; i++) {
        data.f(i, j) = std::sin(i * 0.1 + t) + std::cos(j * 0.1);
      }
    }

    // Write timestep
    std::string filename = "timestep_" + std::to_string(t) + ".h5";
    data.to_h5(filename, "field");
  }

  return 0;
}
```

---

## Parallel Computing (MPI)

### Distributed Arrays

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Create distributed array (global size: 1000 x 2000)
  ftk::ndarray<float> arr;
  arr.decompose(MPI_COMM_WORLD, {1000, 2000});

  // Get this rank's portion
  const auto& core = arr.local_core();      // Core region (no ghosts)
  const auto& extent = arr.local_extent();  // Extended region (with ghosts)

  std::cout << "Rank " << rank << " core: "
            << core.size(0) << " x " << core.size(1) << "\n";

  // Fill local data
  size_t off_i = core.start(0) - extent.start(0);
  size_t off_j = core.start(1) - extent.start(1);

  for (size_t j = 0; j < core.size(1); j++) {
    for (size_t i = 0; i < core.size(0); i++) {
      size_t global_i = core.start(0) + i;
      size_t global_j = core.start(1) + j;
      arr.f(off_i + i, off_j + j) = rank * 1000.0 + global_i + global_j * 0.1;
    }
  }

  // Parallel write to HDF5
  arr.write_hdf5_auto("distributed.h5", "data");

  if (rank == 0) {
    std::cout << "Wrote distributed array to distributed.h5\n";
  }

  MPI_Finalize();
  return 0;
}
```

**Compile with MPI**:
```bash
mpic++ -std=c++17 -I include mpi_example.cpp -lndarray -lhdf5 -o mpi_example
mpirun -np 4 ./mpi_example
```

### Ghost Cell Exchange

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Create distributed array with ghost cells
  ftk::ndarray<float> arr;
  arr.decompose(MPI_COMM_WORLD, {1000, 2000},
                /*nprocs*/ 0,
                /*decomp*/ {},
                /*ghost_width*/ {1, 1});  // 1-cell ghost on each side

  // Fill core region...
  // (your computation)

  // Exchange ghost cells with neighbors
  arr.start_exchange_ghosts();
  arr.finish_exchange_ghosts();

  // Now ghosts are synchronized, can compute stencils
  // that span across domain boundaries

  MPI_Finalize();
  return 0;
}
```

---

## Next Steps

### Learn More

- **[Storage Backends](STORAGE_BACKENDS.md)** - Use xtensor or Eigen for computation
- **[Distributed Arrays](DISTRIBUTED_NDARRAY.md)** - Deep dive into MPI parallelism
- **[GPU Support](GPU_SUPPORT.md)** - Move arrays to/from GPU memory
- **[Parallel HDF5](PARALLEL_HDF5.md)** - High-performance parallel I/O
- **[Dimension Ordering](DIMENSION_ORDERING.md)** - Fortran vs C conventions

### Examples

See the `tests/` directory for comprehensive examples:
- `tests/test_ndarray.cpp` - Basic array operations
- `tests/test_distributed_ndarray.cpp` - MPI parallelism
- `tests/test_hdf5_auto.cpp` - Parallel HDF5 I/O
- `tests/test_adios2_stream.cpp` - Time-series with ADIOS2
- `tests/test_storage_backends.cpp` - Using xtensor/Eigen

### Common Issues

**Q: Compilation errors with template instantiation?**
A: Make sure you're using C++17: `-std=c++17`

**Q: Linking errors with HDF5/NetCDF?**
A: Link libraries in order: `-lndarray -lhdf5 -lnetcdf`

**Q: MPI deadlocks?**
A: Ensure all ranks participate in collective operations (decompose, ghost exchange, parallel I/O)

**Q: Dimension ordering confusion?**
A: Use Fortran indexing (`.f()`) consistently for scientific data. See [DIMENSION_ORDERING.md](DIMENSION_ORDERING.md).

### Getting Help

- **Issues**: https://github.com/hguo/ndarray/issues
- **Documentation**: https://github.com/hguo/ndarray/tree/main/docs
- **Examples**: https://github.com/hguo/ndarray/tree/main/tests

---

**Next**: Try the [examples in tests/](../tests/) or dive into specific topics in the [docs/](.) directory.
