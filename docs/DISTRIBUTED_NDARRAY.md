# Distributed Memory Parallel I/O with ndarray

## Overview

The `distributed_ndarray` class provides distributed memory parallel I/O for large-scale scientific datasets using MPI. It enables efficient reading and processing of time-varying data across multiple compute nodes, with automatic domain decomposition and ghost cell exchange for stencil operations.

**Primary Use Case**: Reading and analyzing large time-series datasets (NetCDF, HDF5, ADIOS2, binary) in distributed memory settings for visualization and analysis workflows.

## Key Features

- **Automatic Domain Decomposition**: Prime factorization-based load balancing
- **Manual Decomposition**: User-specified domain splitting patterns
- **Ghost Layer Support**: Configurable ghost cells for stencil operations
- **Parallel I/O**: Format-agnostic reading (NetCDF, HDF5, binary via MPI-IO)
- **Index Conversion**: Seamless global ↔ local index translation
- **Ghost Exchange**: MPI communication for boundary synchronization
- **Storage Backend Support**: Works with all ndarray storage policies

## Quick Start

### Basic Example (Low-Level API)

```cpp
#include <ndarray/distributed_ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Create distributed array
  ftk::distributed_ndarray<float> temperature(MPI_COMM_WORLD);

  // Automatic decomposition: 1000×800 grid with 1-layer ghosts
  temperature.decompose({1000, 800}, 0, {}, {1, 1});

  // Parallel read from NetCDF file
  temperature.read_parallel("simulation.nc", "temperature", 0);

  // Exchange ghost cells with neighbors
  temperature.exchange_ghosts();

  // Access local data
  auto& local = temperature.local_array();
  std::cout << "Rank " << temperature.rank()
            << " owns " << temperature.local_core() << std::endl;

  MPI_Finalize();
  return 0;
}
```

### YAML Stream Example (Recommended)

For time-series processing, use the YAML stream interface for cleaner code:

```cpp
#include <ndarray/ndarray_group_stream.hh>  // Includes distributed support
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  ftk::distributed_stream<> stream(MPI_COMM_WORLD);
  stream.parse_yaml("config.yaml");

  for (int t = 0; t < stream.n_timesteps(); t++) {
    auto group = stream.read(t);
    group->exchange_ghosts_all();

    auto& temperature = (*group)["temperature"];
    auto& pressure = (*group)["pressure"];
    // ... process data ...
  }

  MPI_Finalize();
  return 0;
}
```

**YAML Configuration (config.yaml):**
```yaml
decomposition:
  global_dims: [1000, 800]
  ghost: [1, 1]

streams:
  - name: simulation
    format: netcdf
    filenames: "data_*.nc"
    vars:
      - name: temperature
      - name: pressure
```

### Compile with MPI

```bash
mpicxx -o example example.cpp -I/path/to/ndarray/include -lyaml-cpp
mpirun -np 4 ./example
```

## Domain Decomposition

### Automatic Decomposition

The library automatically determines optimal domain decomposition based on the global dimensions and number of MPI ranks:

```cpp
ftk::distributed_ndarray<float> darray(MPI_COMM_WORLD);

// Automatic: library chooses decomposition
// 1000×800 with 4 ranks might become 2×2 grid: [500×400] per rank
darray.decompose({1000, 800}, 0, {}, {1, 1});
```

The decomposition algorithm uses prime factorization to balance load across dimensions, preferring more even splits.

**Example decompositions:**
- 1000×800 with 4 ranks: 2×2 → each rank gets ~500×400
- 1000×800 with 2 ranks: 2×1 → each rank gets ~500×800
- 1000×800×600 with 8 ranks: 2×2×2 → each rank gets ~500×400×300

### Manual Decomposition

For specific requirements, specify the decomposition pattern explicitly:

```cpp
int nprocs = 4;

// 1D decomposition: split only dimension 0
darray.decompose({1000, 800},
                 nprocs,
                 {nprocs, 0, 0},  // Split dim 0 into 4, don't split others
                 {2, 0, 0});       // 2-layer ghosts only in dim 0

// 2D decomposition: 2×2 grid
darray.decompose({1000, 800},
                 4,
                 {2, 2},          // 2×2 grid
                 {1, 1});         // 1-layer ghosts in both dimensions
```

**When to use manual decomposition:**
- Irregular domains or anisotropic grids
- Performance tuning based on communication patterns
- Specific application requirements (e.g., 1D for parallel I/O efficiency)

### Ghost Layers

Ghost layers (halos) store copies of neighboring rank's boundary data, enabling stencil operations without additional communication:

```cpp
// No ghosts: pure domain decomposition
darray.decompose({1000, 800}, 0, {}, {});

// 1-layer ghosts: for nearest-neighbor stencils
darray.decompose({1000, 800}, 0, {}, {1, 1});

// 2-layer ghosts: for wider stencils
darray.decompose({1000, 800}, 0, {}, {2, 2});

// Asymmetric: ghosts only in specific dimensions
darray.decompose({1000, 800, 600}, 0, {}, {1, 1, 0});  // No ghosts in z
```

**Memory considerations:**
- Ghost layers increase memory per rank
- For 1000×800 with 1-layer ghosts on 4 ranks: ~500×400 core + ~502×402 with ghosts
- Choose ghost width based on stencil requirements

## Parallel I/O

### Supported Formats

The library automatically detects file format from extension:

| Format | Extension | Backend | Requirements |
|--------|-----------|---------|--------------|
| NetCDF | `.nc` | PNetCDF | `NDARRAY_HAVE_PNETCDF` |
| HDF5 | `.h5` | HDF5 Parallel | `NDARRAY_HAVE_HDF5` (parallel) |
| Binary | `.bin`, `.dat` | MPI-IO | MPI |
| ADIOS2 | `.bp` | ADIOS2 | `NDARRAY_HAVE_ADIOS2` |

### Reading Data

```cpp
ftk::distributed_ndarray<float> darray(MPI_COMM_WORLD);
darray.decompose({1000, 800}, 0, {}, {1, 1});

// Read from NetCDF (variable "temperature", timestep 0)
darray.read_parallel("simulation.nc", "temperature", 0);

// Read from binary file (no variable name needed)
darray.read_parallel("data.bin");

// Each rank reads only its local portion automatically
```

**NetCDF/HDF5 Notes:**
- Variable name matches dataset name in file
- Timestep is the record dimension index (typically time)
- Library uses collective I/O operations for efficiency

**Binary I/O Notes:**
- Assumes row-major (C-order) contiguous layout
- File size must match global dimensions exactly
- Uses MPI-IO collective read (MPI_File_read_at_all)

### Performance Tips

1. **Use collective I/O**: Library uses MPI collective operations automatically
2. **File system tuning**: Configure Lustre striping for large files
3. **Format choice**: NetCDF and HDF5 are self-describing and recommended
4. **I/O patterns**: 1D decomposition often performs better for parallel I/O
5. **Buffering**: Libraries like ADIOS2 provide additional buffering layers

## Index Conversion

### Global and Local Indices

- **Global index**: Position in the full dataset (0 to N-1)
- **Local index**: Position in this rank's local array
- **Core region**: Data owned by this rank (no ghosts)
- **Extent region**: Core + ghost layers

```cpp
ftk::distributed_ndarray<float> darray(MPI_COMM_WORLD);
darray.decompose({1000, 800}, 4, {2, 2}, {1, 1});

// Check if global point is on this rank
std::vector<size_t> global_point = {500, 400};
if (darray.is_local(global_point)) {
  // Convert to local index
  auto local_idx = darray.global_to_local(global_point);

  // Access data
  float value = darray.local_array().f(local_idx[0], local_idx[1]);

  std::cout << "Global [500, 400] = Local ["
            << local_idx[0] << ", " << local_idx[1]
            << "] = " << value << std::endl;
}

// Convert local to global
auto local_pt = {10, 20};
auto global_pt = darray.local_to_global({10, 20});
std::cout << "Local [10, 20] = Global ["
          << global_pt[0] << ", " << global_pt[1] << "]" << std::endl;
```

### Lattice Regions

```cpp
// Global domain
auto global = darray.global_lattice();
std::cout << "Global: " << global.sizes() << std::endl;  // [1000, 800]

// Local core (owned data, no ghosts)
auto core = darray.local_core();
std::cout << "Core start: " << core.starts() << std::endl;  // e.g., [500, 400]
std::cout << "Core size: " << core.sizes() << std::endl;    // e.g., [500, 400]

// Local extent (core + ghosts)
auto extent = darray.local_extent();
std::cout << "Extent start: " << extent.starts() << std::endl;  // e.g., [499, 399]
std::cout << "Extent size: " << extent.sizes() << std::endl;    // e.g., [502, 402]

// Local array size matches extent
auto& local = darray.local_array();
std::cout << "Local array: " << local.shape() << std::endl;  // [502, 402]
```

## Ghost Cell Exchange

### Basic Usage

After reading data and before applying stencil operations, exchange ghost cells:

```cpp
// Read data
darray.read_parallel("data.nc", "temperature");

// Update ghosts from neighbors
darray.exchange_ghosts();

// Now safe to apply stencil (can access i±1, j±1)
auto& local = darray.local_array();
for (size_t i = 1; i < local.dim(0) - 1; i++) {
  for (size_t j = 1; j < local.dim(1) - 1; j++) {
    float avg = (local.at(i-1, j) + local.at(i+1, j) +
                 local.at(i, j-1) + local.at(i, j+1)) / 4.0f;
    // ... use avg ...
  }
}
```

### Stencil Operations

**3-point stencil (1D):**
```cpp
darray.decompose({1000}, nprocs, {nprocs}, {1});  // 1-layer ghosts
darray.read_parallel("data.nc", "field");
darray.exchange_ghosts();

// Laplacian: d²f/dx²
for (size_t i = 1; i < local.dim(0) - 1; i++) {
  float laplacian = local.at(i-1) - 2*local.at(i) + local.at(i+1);
}
```

**5-point stencil (2D):**
```cpp
darray.decompose({1000, 800}, 0, {}, {1, 1});  // 1-layer ghosts
darray.read_parallel("data.nc", "field");
darray.exchange_ghosts();

// 5-point Laplacian
for (size_t i = 1; i < local.dim(0) - 1; i++) {
  for (size_t j = 1; j < local.dim(1) - 1; j++) {
    float lap = local.at(i-1, j) + local.at(i+1, j) +
                local.at(i, j-1) + local.at(i, j+1) - 4*local.at(i, j);
  }
}
```

**9-point stencil (2D with corners):**
```cpp
darray.decompose({1000, 800}, 0, {}, {1, 1});  // 1-layer ghosts
darray.read_parallel("data.nc", "field");
darray.exchange_ghosts();

for (size_t i = 1; i < local.dim(0) - 1; i++) {
  for (size_t j = 1; j < local.dim(1) - 1; j++) {
    float sum = 0.0f;
    for (int di = -1; di <= 1; di++) {
      for (int dj = -1; dj <= 1; dj++) {
        sum += local.at(i+di, j+dj);
      }
    }
    float avg = sum / 9.0f;
  }
}
```

### Communication Details

The `exchange_ghosts()` implementation:
1. **Identify neighbors**: Done once during `decompose()`
2. **Post receives**: Non-blocking receives (MPI_Irecv) for all neighbors
3. **Send boundary data**: Blocking sends (MPI_Send) to all neighbors
4. **Wait**: Wait for all receives to complete (MPI_Waitall)
5. **Unpack**: Copy received data into ghost regions

**Performance characteristics:**
- Communication volume: O(surface area) per rank
- Latency: One synchronization point per exchange
- Overlapping: Future optimization could overlap computation and communication

## Time-Series Processing

### Processing Multiple Timesteps

```cpp
ftk::distributed_ndarray<float> temperature(MPI_COMM_WORLD);
temperature.decompose({1000, 800}, 0, {}, {1, 1});

for (int t = 0; t < num_timesteps; t++) {
  // Read timestep
  temperature.read_parallel("simulation.nc", "temperature", t);

  // Exchange ghosts
  temperature.exchange_ghosts();

  // Compute local gradient magnitude
  auto& local = temperature.local_array();
  float local_max_gradient = 0.0f;

  for (size_t i = 1; i < local.dim(0) - 1; i++) {
    for (size_t j = 1; j < local.dim(1) - 1; j++) {
      float dx = (local.at(i+1, j) - local.at(i-1, j)) / 2.0f;
      float dy = (local.at(i, j+1) - local.at(i, j-1)) / 2.0f;
      float grad = std::sqrt(dx*dx + dy*dy);
      local_max_gradient = std::max(local_max_gradient, grad);
    }
  }

  // Global reduction
  float global_max_gradient;
  MPI_Reduce(&local_max_gradient, &global_max_gradient, 1,
             MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (temperature.rank() == 0) {
    std::cout << "Timestep " << t
              << " max gradient: " << global_max_gradient << std::endl;
  }
}
```

### Statistics Aggregation

```cpp
// Compute local statistics
float local_min = std::numeric_limits<float>::max();
float local_max = std::numeric_limits<float>::lowest();
double local_sum = 0.0;
size_t local_count = temperature.local_core().n();

for (size_t i = 0; i < temperature.local_core().size(0); i++) {
  for (size_t j = 0; j < temperature.local_core().size(1); j++) {
    float value = local.at(i, j);
    local_min = std::min(local_min, value);
    local_max = std::max(local_max, value);
    local_sum += value;
  }
}

// Global aggregation
float global_min, global_max;
double global_sum;
size_t global_count;

MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG,
           MPI_SUM, 0, MPI_COMM_WORLD);

if (temperature.rank() == 0) {
  double global_mean = global_sum / global_count;
  std::cout << "Global stats: min=" << global_min
            << ", max=" << global_max
            << ", mean=" << global_mean << std::endl;
}
```

## API Reference

### Constructor

```cpp
distributed_ndarray(MPI_Comm comm = MPI_COMM_WORLD)
```

Creates a distributed array with the given MPI communicator.

### Domain Decomposition

```cpp
void decompose(const std::vector<size_t>& global_dims,
               size_t nprocs = 0,
               const std::vector<size_t>& decomp = {},
               const std::vector<size_t>& ghost = {})
```

Decomposes the global domain across MPI ranks.

**Parameters:**
- `global_dims`: Global array dimensions (e.g., `{1000, 800}`)
- `nprocs`: Number of processes (0 = use all ranks in communicator)
- `decomp`: Decomposition pattern (empty = automatic, e.g., `{2, 2}` for 2×2 grid)
- `ghost`: Ghost layers per dimension (e.g., `{1, 1}` for 1-layer ghosts)

### Parallel I/O

```cpp
void read_parallel(const std::string& filename,
                   const std::string& varname = "",
                   int timestep = 0)
```

Reads data in parallel from file. Each rank reads its local portion.

**Parameters:**
- `filename`: Path to data file (.nc, .h5, .bin, .bp)
- `varname`: Variable/dataset name (not needed for binary files)
- `timestep`: Record dimension index (default 0)

### Ghost Exchange

```cpp
void exchange_ghosts()
```

Updates ghost cells by exchanging boundary data with neighboring ranks. Must be called after reading and before stencil operations.

### Index Conversion

```cpp
std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const
std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const
bool is_local(const std::vector<size_t>& global_idx) const
```

Convert between global and local index spaces.

### Accessors

```cpp
const ndarray<T, StoragePolicy>& local_array() const
ndarray<T, StoragePolicy>& local_array()
```

Access the local data array (includes core + ghosts).

```cpp
const lattice& global_lattice() const
const lattice& local_core() const
const lattice& local_extent() const
```

Access lattice information for global domain, local core, and local extent.

```cpp
int rank() const
int nprocs() const
MPI_Comm comm() const
```

Access MPI information.

## Examples

See the `examples/` directory for complete working examples:

- **distributed_io.cpp**: Basic parallel I/O workflow
- **distributed_stencil.cpp**: Stencil computation with ghost exchange
- **distributed_analysis.cpp**: Time-series analysis pipeline

Run examples with:
```bash
mpirun -np 4 ./bin/distributed_io
mpirun -np 4 ./bin/distributed_stencil
mpirun -np 4 ./bin/distributed_analysis
```

## Testing

Run the distributed memory test suite:

```bash
# Domain decomposition and index conversion tests
mpirun -np 4 ./bin/test_distributed_ndarray

# Ghost exchange tests
mpirun -np 4 ./bin/test_ghost_exchange
```

Test with different numbers of ranks:
```bash
mpirun -np 1 ./bin/test_distributed_ndarray  # Single rank
mpirun -np 2 ./bin/test_distributed_ndarray  # 1D or 2D decomposition
mpirun -np 4 ./bin/test_distributed_ndarray  # 2D decomposition (2×2)
mpirun -np 8 ./bin/test_distributed_ndarray  # 2D (4×2) or 3D (2×2×2)
```

## Limitations and Future Work

### Current Limitations

1. **Ghost exchange**: Simplified implementation for face-adjacent neighbors only
   - Corner and edge exchanges for multi-dimensional arrays require additional work
   - Currently optimized for 1D and 2D decompositions

2. **Parallel write**: Not yet implemented
   - Focus has been on read capabilities for vis/analysis
   - Will be added in future updates

3. **HDF5 parallel**: Stub implementation needs completion
   - Requires HDF5 compiled with parallel support
   - API similar to PNetCDF implementation

4. **Performance optimization**:
   - No overlapping of communication and computation
   - No GPU-aware MPI support
   - No asynchronous I/O

### Future Enhancements

1. **Parallel write operations**: Collective write for all formats
2. **Optimized ghost packing**: Reduce message count, pack multiple faces
3. **Overlap computation/communication**: Start ghost exchange, compute interior, finish ghosts
4. **Dynamic load balancing**: Repartition based on runtime characteristics
5. **GPU support**: GPU-aware MPI for direct device-to-device transfers
6. **Asynchronous I/O**: Use ADIOS2 async mode for background I/O

## Troubleshooting

### Common Issues

**Problem**: Segmentation fault during read_parallel()
- Check that decomposition matches file dimensions
- Verify file format matches extension
- Ensure MPI is properly initialized

**Problem**: Incorrect ghost values after exchange
- Call `exchange_ghosts()` after reading and before stencil operations
- Check that ghost width in `decompose()` matches stencil requirements
- Verify neighbor identification with small test case

**Problem**: Poor I/O performance
- Try 1D decomposition for better I/O patterns
- Check filesystem (Lustre striping, etc.)
- Use NetCDF or HDF5 instead of binary
- Verify collective I/O is being used

**Problem**: Memory issues with large datasets
- Reduce ghost layer width if possible
- Use fewer ranks (more memory per rank)
- Check that you're accessing local_core() not local_array() for statistics

### Debugging Tips

1. **Print decomposition**:
```cpp
std::cout << "Rank " << darray.rank() << " core: "
          << darray.local_core() << std::endl;
```

2. **Verify neighbors**:
```cpp
// Add debug output in exchange_ghosts() to see neighbor communication
```

3. **Check data integrity**:
```cpp
// Fill with rank-specific pattern, exchange, verify
auto& local = darray.local_array();
for (size_t i = 0; i < local.size(); i++) {
  local.data()[i] = static_cast<float>(darray.rank() * 1000 + i);
}
darray.exchange_ghosts();
// Now check ghost regions contain neighbor rank's values
```

4. **MPI error checking**:
```cpp
// All MPI calls should check return value in production code
int err = MPI_Init(&argc, &argv);
if (err != MPI_SUCCESS) {
  std::cerr << "MPI_Init failed" << std::endl;
  return 1;
}
```

## Additional Resources

- **MPI Tutorial**: https://mpitutorial.com/
- **NetCDF-4 Parallel I/O**: https://www.unidata.ucar.edu/software/netcdf/docs/parallel_io.html
- **HDF5 Parallel**: https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5
- **ADIOS2 Documentation**: https://adios2.readthedocs.io/

## Contact

For questions, issues, or contributions related to distributed memory functionality:
- Open an issue on the GitHub repository
- See main README.md for contact information
