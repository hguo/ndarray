# Phase 4: Stream Integration with Per-Variable Distribution

## Overview

Integrated unified ndarray with stream class to support per-variable distribution configuration via YAML. Stream now automatically configures arrays as distributed, replicated, or serial based on YAML configuration and MPI context.

## Key Features

### 1. Per-Variable Distribution Types

```cpp
enum class VariableDistType {
  DISTRIBUTED,  // Domain-decomposed across ranks
  REPLICATED,   // Full data on all ranks (DEFAULT)
  AUTO          // Auto-detect based on size/usage
};
```

**Default**: `REPLICATED` (safe, works for all cases as user requested)

### 2. Variable-Specific Decomposition

```cpp
struct variable_decomposition {
  std::vector<size_t> dims;     // Variable-specific dimensions
  std::vector<size_t> pattern;  // Decomposition pattern (0 = don't split)
  std::vector<size_t> ghost;    // Ghost layers per dimension
};
```

### 3. Stream Configuration Methods

```cpp
// Set default decomposition for all distributed variables
void set_default_decomposition(const std::vector<size_t>& global_dims,
                                size_t nprocs = 0,
                                const std::vector<size_t>& decomp = {},
                                const std::vector<size_t>& ghost = {});

// Configure individual variable distribution type
void set_variable_distribution(const std::string& varname, VariableDistType type);

// Configure individual variable with custom decomposition
void set_variable_decomposition(const std::string& varname,
                                 const variable_decomposition& decomp);
```

## YAML Configuration Format

### Example 1: Basic Configuration with Default Replicated

```yaml
stream:
  # Default decomposition for distributed variables
  decomposition:
    global_dims: [1000, 800, 600]
    pattern: []  # auto
    ghost: [1, 1, 1]

  # Per-variable configuration
  variables:
    # Large field: explicitly mark as distributed
    temperature:
      type: distributed

    # Vector field with custom decomposition
    velocity:
      type: distributed
      decomposition:
        dims: [1000, 800, 600, 3]
        pattern: [4, 2, 1, 0]  # Don't decompose vector components
        ghost: [1, 1, 1, 0]

    # Small data: replicated (or omit for default)
    mesh_coordinates:
      type: replicated

    # No config = replicated by default
    grid_spacing: {}

  substreams:
    - name: fields
      format: netcdf
      filenames: data_{timestep:04d}.nc
      vars:
        - name: temperature
        - name: velocity

    - name: mesh
      format: netcdf
      filenames: mesh.nc
      static: true
      vars:
        - name: mesh_coordinates
        - name: grid_spacing
```

### Example 2: All Replicated (Safest Default)

```yaml
stream:
  # No decomposition section = all variables replicated
  # This is the safest default - works for serial and parallel

  substreams:
    - name: data
      format: netcdf
      filenames: data.nc
      vars:
        - name: temperature  # Replicated by default
        - name: pressure     # Replicated by default
```

### Example 3: Mixed Distributed/Replicated

```yaml
stream:
  decomposition:
    global_dims: [1000, 800, 600]
    ghost: [1, 1, 1]

  variables:
    # Distributed variables
    temperature: { type: distributed }
    pressure: { type: distributed }
    velocity:
      type: distributed
      decomposition:
        dims: [1000, 800, 600, 3]
        pattern: [4, 2, 1, 0]

    # Replicated variables (global data needed by all ranks)
    mesh_x: { type: replicated }
    mesh_y: { type: replicated }
    mesh_z: { type: replicated }
    timestep_info: { type: replicated }

  substreams:
    - name: fields
      format: netcdf
      filenames: simulation_{timestep:04d}.nc
      vars:
        - name: temperature
        - name: pressure
        - name: velocity

    - name: mesh
      format: netcdf
      filenames: mesh.nc
      static: true
      vars:
        - name: mesh_x
        - name: mesh_y
        - name: mesh_z
        - name: timestep_info
```

## Usage Examples

### Example 1: Programmatic Configuration

```cpp
#include <ndarray/ndarray_stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Create stream with MPI communicator
  ftk::stream<> s(MPI_COMM_WORLD);

  // Set default decomposition for distributed variables
  s.set_default_decomposition({1000, 800, 600}, 0, {}, {1, 1, 1});

  // Configure variables
  s.set_variable_distribution("temperature", ftk::VariableDistType::DISTRIBUTED);
  s.set_variable_distribution("pressure", ftk::VariableDistType::DISTRIBUTED);
  s.set_variable_distribution("mesh", ftk::VariableDistType::REPLICATED);

  // Custom decomposition for velocity (don't split vector components)
  ftk::variable_decomposition vel_decomp;
  vel_decomp.dims = {1000, 800, 600, 3};
  vel_decomp.pattern = {4, 2, 1, 0};
  vel_decomp.ghost = {1, 1, 1, 0};
  s.set_variable_decomposition("velocity", vel_decomp);

  // Parse YAML (will use configured decompositions)
  s.parse_yaml("config.yaml");

  // Read data - arrays automatically configured
  for (int t = 0; t < s.total_timesteps(); t++) {
    auto vars = s.read(t);

    // Temperature is distributed
    auto& temp = (*vars)["temperature"];
    if (temp.is_distributed()) {
      temp.exchange_ghosts();
      // Process local portion
    }

    // Mesh is replicated
    auto& mesh = (*vars)["mesh"];
    if (mesh.is_replicated()) {
      // All ranks have full mesh data
    }
  }

  MPI_Finalize();
  return 0;
}
```

### Example 2: YAML-Driven Configuration

```cpp
#include <ndarray/ndarray_stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create stream
  ftk::stream<> s(MPI_COMM_WORLD);

  // Parse YAML with variable configurations
  s.parse_yaml("simulation.yaml");

  if (rank == 0) {
    std::cout << "Running with " << nprocs << " ranks" << std::endl;
  }

  // Read and process
  for (int t = 0; t < s.total_timesteps(); t++) {
    auto vars = s.read(t);

    // Access arrays - distribution already configured
    auto& temp = (*vars)["temperature"];
    auto& mesh = (*vars)["mesh_coordinates"];

    // Check distribution
    if (temp.is_distributed()) {
      if (rank == 0) {
        std::cout << "Temperature: distributed" << std::endl;
        std::cout << "  Global: " << temp.global_lattice().size(0)
                  << "×" << temp.global_lattice().size(1) << std::endl;
        std::cout << "  Local: " << temp.dim(0) << "×" << temp.dim(1) << std::endl;
      }

      temp.exchange_ghosts();

      // Process local data
      for (size_t i = 0; i < temp.dim(0); i++) {
        for (size_t j = 0; j < temp.dim(1); j++) {
          float val = temp.f(i, j);
          // ...
        }
      }
    }

    if (mesh.is_replicated()) {
      if (rank == 0) {
        std::cout << "Mesh: replicated (full data on all ranks)" << std::endl;
      }

      // All ranks have full mesh
      for (size_t i = 0; i < mesh.dim(0); i++) {
        double x = mesh.f(i, 0);
        double y = mesh.f(i, 1);
        double z = mesh.f(i, 2);
        // ...
      }
    }
  }

  MPI_Finalize();
  return 0;
}
```

### Example 3: Same Code for Serial and Parallel

```cpp
// SAME CODE runs with: ./program  OR  mpirun -n 4 ./program

#include <ndarray/ndarray_stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create stream
  ftk::stream<> s(MPI_COMM_WORLD);
  s.parse_yaml("config.yaml");

  // Read data
  for (int t = 0; t < s.total_timesteps(); t++) {
    auto vars = s.read(t);

    // If nprocs == 1: arrays are serial
    // If nprocs > 1 + type:distributed: arrays are decomposed
    // If nprocs > 1 + type:replicated: arrays are replicated

    auto& temp = (*vars)["temperature"];

    // Optional: exchange ghosts (no-op if not distributed)
    if (temp.is_distributed()) {
      temp.exchange_ghosts();
    }

    // Process (works for all modes)
    for (size_t i = 0; i < temp.dim(0); i++) {
      for (size_t j = 0; j < temp.dim(1); j++) {
        float val = temp.f(i, j);
        // ...
      }
    }
  }

  MPI_Finalize();
  return 0;
}
```

## Implementation Details

### 1. Automatic Configuration in read()

When `read()` is called:
1. Substreams read data into ndarray_group
2. For each array in the group:
   - Check if variable has explicit distribution type
   - Default to `REPLICATED` if not specified
   - If `DISTRIBUTED`: Call `array.decompose()` with appropriate parameters
   - If `REPLICATED`: Call `array.set_replicated()`
   - If `AUTO`: Leave as serial (could add heuristics later)

### 2. YAML Parsing

`parse_yaml()` now parses:
- `decomposition`: Default decomposition for all distributed variables
- `variables`: Per-variable configuration mapping

### 3. Serial Mode (nprocs == 1)

When `nprocs == 1`, distribution configuration is skipped entirely:
- Arrays remain as regular serial ndarrays
- No decomposition overhead
- Backward compatible with existing code

### 4. Default Behavior

**User requested default: REPLICATED**

- Variables without explicit `type:` are replicated
- Safest option: works for all cases
- User must explicitly mark large arrays as `distributed` for efficiency

## Benefits

✅ **Safe Default**: Replicated works for serial and parallel
✅ **Explicit Control**: User chooses which variables to distribute
✅ **Flexible**: Per-variable customization
✅ **Automatic**: Stream configures arrays, no manual setup
✅ **Backward Compatible**: Existing YAML files work (all replicated)
✅ **Unified**: Same stream class for serial and parallel

## Migration from distributed_stream

### Old (distributed_stream - to be removed)

```cpp
ftk::distributed_stream<> stream(MPI_COMM_WORLD);
stream.set_decomposition({1000, 800}, 0, {}, {1, 1});
stream.parse_yaml("config.yaml");

auto vars = stream.read(0);
// Returns distributed_ndarray_group
```

### New (unified stream)

```yaml
# config.yaml
stream:
  decomposition:
    global_dims: [1000, 800]
    ghost: [1, 1]

  variables:
    temperature: { type: distributed }
    mesh: { type: replicated }

  substreams:
    - name: data
      format: netcdf
      filenames: data.nc
```

```cpp
ftk::stream<> stream(MPI_COMM_WORLD);
stream.parse_yaml("config.yaml");

auto vars = stream.read(0);
// Returns ndarray_group with configured arrays
```

## Performance Considerations

### Distributed Variables
- **Memory**: O(N/P) per rank (N = global size, P = ranks)
- **I/O**: Parallel I/O if using `_auto` methods
- **Communication**: Ghost exchange as needed
- **Best for**: Large arrays (> 1 GB)

### Replicated Variables
- **Memory**: O(N) per rank (full data)
- **I/O**: Rank 0 + broadcast
- **Communication**: One-time broadcast
- **Best for**: Small arrays (< 100 MB) needed by all ranks

## Testing

### Unit Tests Needed

1. **YAML Parsing**:
   - Test decomposition section parsing
   - Test variables section parsing
   - Test default replicated behavior

2. **Configuration**:
   - Test set_default_decomposition()
   - Test set_variable_distribution()
   - Test set_variable_decomposition()

3. **Array Configuration**:
   - Test distributed arrays are properly decomposed
   - Test replicated arrays have full data
   - Test serial mode (nprocs == 1)

4. **Mixed Variables**:
   - Test stream with both distributed and replicated
   - Verify correct configuration for each

### Integration Tests

- Run existing examples with new YAML configs
- Test with 1, 2, 4 ranks
- Verify backward compatibility (no variables section = all replicated)

## Status

✅ **Phase 4 Complete**:
- Per-variable distribution configuration
- YAML parsing for decomposition and variables
- Automatic array configuration in read()
- Default behavior: replicated (as requested)
- Programmatic API for configuration
- Backward compatible

**Next**: Phase 5 - Cleanup and remove distributed_ndarray classes

## Future Enhancements

1. **Auto-detection**: Implement heuristics for VariableDistType::AUTO
2. **Performance hints**: Add YAML hints for I/O optimization
3. **Dynamic reconfiguration**: Change distribution at runtime
4. **Validation**: Warn if replicated array is too large
5. **Statistics**: Report memory usage per rank
