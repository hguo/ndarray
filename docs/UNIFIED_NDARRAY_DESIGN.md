# Unified ndarray Design: Single Class for Serial and Parallel

> **Note**: This design document describes the implemented architecture. The unified `ndarray` class with optional MPI support via `decompose()` has been the implementation from the start. There is no separate `distributed_ndarray` class - references to it in this document are for conceptual comparison only.

## Core Concept

**One `ndarray` class that adapts to MPI configuration at runtime.**

- Serial mode (nprocs == 1 or no decomposition): Regular ndarray behavior
- Distributed mode (nprocs > 1 + decomposition): Domain-decomposed with MPI
- Replicated mode (nprocs > 1 + no decomposition): Full array on all ranks

## Unified ndarray Class

```cpp
template <typename T, typename StoragePolicy = native_storage>
class ndarray {
public:
  // === Existing interface (unchanged) ===

  void reshape(size_t n0, size_t n1, ...);
  void reshapef(size_t n0, size_t n1, ...);

  T& operator()(size_t i0, size_t i1, ...);  // Local indexing
  T& at(size_t i0, size_t i1, ...);          // Local indexing

  void read_netcdf(...);
  void write_netcdf(...);
  // ... all existing I/O methods ...

  // === NEW: Optional MPI/distribution support ===

  /**
   * @brief Configure array for distributed execution
   *
   * After calling decompose(), the array behaves as domain-decomposed:
   * - operator() and at() use local indices (within this rank's portion)
   * - read_*() and write_*() methods do parallel I/O automatically
   * - exchange_ghosts() updates ghost layers via MPI
   *
   * @param comm MPI communicator
   * @param global_dims Global array dimensions
   * @param nprocs Number of processes (0 = use comm size)
   * @param decomp Decomposition pattern (empty = auto)
   * @param ghost Ghost layers per dimension
   */
  void decompose(MPI_Comm comm,
                 const std::vector<size_t>& global_dims,
                 size_t nprocs = 0,
                 const std::vector<size_t>& decomp = {},
                 const std::vector<size_t>& ghost = {});

  /**
   * @brief Mark array as replicated across all ranks
   *
   * All ranks have full data, no domain decomposition.
   * I/O can use:
   * - Rank 0 reads/writes + MPI_Bcast
   * - Or all ranks read/write independently
   *
   * @param comm MPI communicator
   */
  void set_replicated(MPI_Comm comm);

  /**
   * @brief Exchange ghost layers with neighboring ranks
   *
   * No-op if array is not distributed.
   */
  void exchange_ghosts();

  /**
   * @brief Check if array is distributed
   */
  bool is_distributed() const { return dist_ && dist_->type == DistType::DISTRIBUTED; }

  /**
   * @brief Check if array is replicated
   */
  bool is_replicated() const { return dist_ && dist_->type == DistType::REPLICATED; }

  /**
   * @brief Check if array has MPI configuration (distributed or replicated)
   */
  bool has_mpi_config() const { return dist_ != nullptr; }

  // === Distribution-specific accessors ===

  /**
   * @brief Get global lattice (full domain)
   * Throws if not distributed.
   */
  const lattice& global_lattice() const;

  /**
   * @brief Get local core (owned data, no ghosts)
   * Throws if not distributed.
   */
  const lattice& local_core() const;

  /**
   * @brief Get local extent (owned + ghosts)
   * Throws if not distributed.
   */
  const lattice& local_extent() const;

  /**
   * @brief Convert global index to local index
   * Throws if not distributed or index not local.
   */
  std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const;

  /**
   * @brief Convert local index to global index
   * Throws if not distributed.
   */
  std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const;

  /**
   * @brief Check if global index is in local core
   */
  bool is_local(const std::vector<size_t>& global_idx) const;

  // === MPI accessors ===

  MPI_Comm comm() const;
  int rank() const;
  int nprocs() const;

private:
  // Existing members
  StoragePolicy storage_;
  std::vector<size_t> dims_;
  // ... all existing members ...

  // NEW: Optional distribution info
  enum class DistType { DISTRIBUTED, REPLICATED };

  struct distribution_info {
    DistType type;
    MPI_Comm comm;
    int rank;
    int nprocs;

    // For DISTRIBUTED arrays only:
    lattice global_lattice_;
    lattice local_core_;
    lattice local_extent_;
    std::unique_ptr<lattice_partitioner> partitioner_;

    // Neighbor info for ghost exchange
    std::vector<int> neighbor_ranks_;
    std::vector<lattice> send_regions_;
    std::vector<lattice> recv_regions_;
  };

  std::unique_ptr<distribution_info> dist_;  // nullptr = serial, non-null = parallel

  // Helper: Check if we should use parallel I/O
  bool should_use_parallel_io() const {
    return dist_ && dist_->type == DistType::DISTRIBUTED;
  }

  // Helper: Check if we should use replicated I/O (rank 0 + bcast)
  bool should_use_replicated_io() const {
    return dist_ && dist_->type == DistType::REPLICATED;
  }
};
```

## Key Design Principles

### 1. **Backward Compatible**
Existing code using `ndarray` works unchanged:
```cpp
// Existing serial code - no changes needed
ftk::ndarray<float> arr;
arr.reshapef(100, 100);
arr(50, 50) = 1.0f;
arr.to_netcdf("data.nc", "var");
```

### 2. **Opt-in Distribution**
Distribution is optional, configured explicitly:
```cpp
// NEW: Distributed code
ftk::ndarray<float> arr;
arr.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

// Now arr is distributed:
// - arr.dims() returns LOCAL dimensions (e.g., 250x400 on rank 0 of 4)
// - arr(i, j) uses LOCAL indices
// - arr.read_netcdf() does parallel I/O automatically

arr.read_netcdf("data.nc", "temperature");
arr.exchange_ghosts();

// Access local data
for (size_t i = 0; i < arr.dim(0); i++) {
  for (size_t j = 0; j < arr.dim(1); j++) {
    float val = arr(i, j);  // Local indexing
    // ...
  }
}
```

### 3. **Replicated Support**
```cpp
// Mesh data: all ranks need full array
ftk::ndarray<double> mesh;
mesh.set_replicated(MPI_COMM_WORLD);
mesh.read_netcdf("mesh.nc", "coordinates");  // Rank 0 reads + bcast

// All ranks have full mesh
for (size_t i = 0; i < mesh.dim(0); i++) {
  double x = mesh(i, 0);  // Same on all ranks
}
```

### 4. **I/O Auto-adapts**

```cpp
// Serial (no MPI config): Regular I/O
arr.read_netcdf("data.nc", "var");

// Distributed: Parallel I/O
arr.decompose(comm, dims);
arr.read_netcdf("data.nc", "var");  // Parallel read (PNetCDF/HDF5/ADIOS2)

// Replicated: Rank 0 read + broadcast
arr.set_replicated(comm);
arr.read_netcdf("data.nc", "var");  // Rank 0 reads, MPI_Bcast to others
```

## Unified Stream with YAML Configuration

### YAML Format (Default: Replicated)

```yaml
# Optional: Default decomposition for distributed variables
decomposition:
  global_dims: [1000, 800, 600]
  pattern: []  # auto
  ghost: [1, 1, 1]

# Variable configuration
variables:
  # Large field: distributed for efficiency
  temperature:
    type: distributed
    # Uses default decomposition from above

  # Vector field: custom decomposition
  velocity:
    type: distributed
    decomposition:
      dims: [1000, 800, 600, 3]
      pattern: [4, 2, 1, 0]  # Don't split vector components
      ghost: [1, 1, 1, 0]

  # Small data: replicated (default if not specified)
  mesh_coordinates:
    type: replicated  # or omit "type" for default

  # Default is replicated
  grid_spacing: {}  # No type specified = replicated

  timestep_info: {}  # replicated by default

# Data streams
streams:
  - name: fields
    format: netcdf
    filenames: data_{timestep:04d}.nc
    vars:
      - temperature
      - velocity

  - name: mesh
    format: netcdf
    filenames: mesh.nc
    static: true
    vars:
      - mesh_coordinates
      - grid_spacing

  - name: metadata
    format: yaml
    filenames: metadata_{timestep:04d}.yaml
    vars:
      - timestep_info
```

### Unified Stream Class

```cpp
template <typename T = float, typename StoragePolicy = native_storage>
class stream {
public:
  using array_type = ndarray<T, StoragePolicy>;
  using group_type = ndarray_group<T, StoragePolicy>;

  /**
   * @brief Constructor
   *
   * @param comm MPI communicator
   *   - If MPI_COMM_WORLD with nprocs > 1: Parallel mode
   *   - If MPI_COMM_SELF or nprocs == 1: Serial mode
   */
  stream(MPI_Comm comm = MPI_COMM_SELF);

  /**
   * @brief Parse YAML configuration
   *
   * Configures variables as distributed or replicated based on YAML.
   * If nprocs == 1, all variables behave as regular arrays.
   */
  void parse_yaml(const std::string& yaml_file);

  /**
   * @brief Read timestep
   *
   * Returns ndarray_group where each array is configured as:
   * - Distributed (if type: distributed in YAML and nprocs > 1)
   * - Replicated (if type: replicated in YAML and nprocs > 1)
   * - Regular (if nprocs == 1)
   */
  std::shared_ptr<group_type> read(int timestep);

  /**
   * @brief Read static variables
   */
  std::shared_ptr<group_type> read_static();

  // ... rest of interface same as before ...
};
```

## Usage Examples

### Example 1: Same Code for Serial and Parallel

```cpp
#include <ndarray/stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Same stream class, same YAML
  ftk::stream<> stream(MPI_COMM_WORLD);
  stream.parse_yaml("config.yaml");

  if (rank == 0) {
    std::cout << "Running with " << nprocs << " rank(s)" << std::endl;
  }

  for (int t = 0; t < stream.n_timesteps(); t++) {
    auto vars = stream.read(t);

    // Access arrays - same code for serial and parallel!
    auto& temp = (*vars)["temperature"];
    auto& mesh = (*vars)["mesh_coordinates"];

    // Check what mode we're in
    if (temp.is_distributed()) {
      // Parallel mode: exchange ghosts
      temp.exchange_ghosts();

      if (rank == 0) {
        std::cout << "Timestep " << t << ": distributed processing" << std::endl;
        std::cout << "  Local temp size: " << temp.dim(0) << " × " << temp.dim(1) << std::endl;
        std::cout << "  Global temp size: " << temp.global_lattice().size(0)
                  << " × " << temp.global_lattice().size(1) << std::endl;
      }
    } else {
      // Serial mode
      std::cout << "Timestep " << t << ": serial processing" << std::endl;
      std::cout << "  Temp size: " << temp.dim(0) << " × " << temp.dim(1) << std::endl;
    }

    // Process local data (same indexing!)
    for (size_t i = 1; i < temp.dim(0) - 1; i++) {
      for (size_t j = 1; j < temp.dim(1) - 1; j++) {
        float avg = (temp(i-1, j) + temp(i+1, j) +
                     temp(i, j-1) + temp(i, j+1)) / 4.0f;
        // ... compute ...
      }
    }

    // Mesh is replicated - all ranks have same data
    if (mesh.is_replicated()) {
      // Can access any global index on any rank
      double x0 = mesh(0, 0);  // Same value on all ranks
    }
  }

  MPI_Finalize();
  return 0;
}

// Run serial: ./program
// Run parallel: mpirun -n 4 ./program
// Same binary, same code!
```

### Example 2: Manual Configuration (No YAML)

```cpp
ftk::ndarray<float> temp;

int nprocs;
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

if (nprocs > 1) {
  // Configure for distributed execution
  temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});
} else {
  // Serial: just reshape
  temp.reshapef(1000, 800);
}

// Read data (auto-adapts to parallel or serial)
temp.read_netcdf("data.nc", "temperature");

// Process (same code!)
for (size_t i = 0; i < temp.dim(0); i++) {
  for (size_t j = 0; j < temp.dim(1); j++) {
    float val = temp(i, j);  // Local indexing
  }
}

// Exchange ghosts (no-op if serial)
temp.exchange_ghosts();
```

### Example 3: Mixed Distributed and Replicated

```cpp
ftk::stream<> stream(MPI_COMM_WORLD);
stream.parse_yaml("config.yaml");

auto vars = stream.read(0);

// Large distributed field
auto& temp = (*vars)["temperature"];
if (temp.is_distributed()) {
  temp.exchange_ghosts();

  // Use global coordinates from replicated mesh
  auto& mesh = (*vars)["mesh_coordinates"];

  for (size_t i = 0; i < temp.dim(0); i++) {
    // Convert local index to global
    auto global_idx = temp.local_to_global({i, 0});

    // Access global mesh data (available on all ranks)
    double x = mesh(global_idx[0], 0);
    double y = mesh(global_idx[0], 1);

    // Process using global coordinates
    // ...
  }
}
```

## Implementation Strategy

### Phase 1: Add Optional Distribution to ndarray
- Add `distribution_info` struct
- Add `decompose()` and `set_replicated()` methods
- No-op implementations for ghost exchange, etc.
- **Estimated: 2-3 days**

### Phase 2: Implement Distribution Logic
- Copy logic from current `distributed_ndarray.hh`
- Integrate lattice_partitioner
- Implement ghost exchange
- **Estimated: 2-3 days**

### Phase 3: Update I/O Methods
- Modify `read_netcdf()`, `read_hdf5()`, etc. to check distribution
- If distributed: Use parallel I/O paths
- If replicated: Rank 0 reads + MPI_Bcast
- If serial: Use existing paths
- **Estimated: 3-4 days**

### Phase 4: Update Stream Class
- Remove distinction between `stream` and `distributed_stream`
- Parse YAML variable configs
- Configure arrays as distributed/replicated/serial based on YAML + nprocs
- **Estimated: 2-3 days**

### Phase 5: Testing and Migration
- Update tests to use unified API
- Update examples
- Deprecate old `distributed_ndarray` class (keep as typedef for compatibility)
- **Estimated: 2-3 days**

**Total: ~2-3 weeks**

## Benefits

✅ **Simplicity**: One class, one API
✅ **Compatibility**: Existing serial code unchanged
✅ **Flexibility**: Same code runs serial or parallel
✅ **Clarity**: Distribution is explicit opt-in configuration
✅ **Safety**: Default is replicated (works for all cases)
✅ **Power**: Advanced users can control per-variable decomposition

## Current API (Implemented)

### Unified ndarray with optional MPI support
```cpp
// Create a regular array
ftk::ndarray<float> temp;

// Make it distributed by calling decompose()
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

// I/O methods auto-detect parallel mode
temp.read_netcdf("data.nc", "temperature");

// Exchange ghost cells with neighbors
temp.exchange_ghosts();
```

Key features:
- Single `ndarray` class for both serial and parallel
- Call `decompose()` to enable MPI domain decomposition
- I/O methods automatically use parallel I/O when decomposed
- No separate `distributed_ndarray` class needed

## Open Questions

1. **Replicated I/O strategy**: Rank 0 read + MPI_Bcast, or all ranks read independently?
   - **Recommendation**: Rank 0 + MPI_Bcast (less I/O contention, works for all formats)

2. **Storage overhead**: Replicated arrays use N × memory total. Document limits?
   - **Recommendation**: Warning if replicated array > 100 MB

3. **Default behavior**: When no configuration, what is default?
   - **Recommendation**: If nprocs > 1 and no config → error (user must be explicit)

4. **Backward compatibility**: Keep `distributed_ndarray` as typedef?
   - **Status**: Not needed - the unified `ndarray` was implemented from the start

## Summary

**Design Decision**: Single `ndarray` class with optional MPI support via `decompose()`.

**User Benefit**: Write once, run anywhere (serial or parallel).

**Default Behavior**: Replicated (safe, works for all cases).

**Advanced Usage**: Explicitly mark large arrays as `type: distributed` for efficiency.
