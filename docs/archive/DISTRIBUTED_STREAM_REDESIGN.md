# Distributed Stream Redesign

## Current Limitations

1. **Single decomposition for all variables**: All variables must have the same dimensions and decomposition
2. **No replicated variables**: Can't handle global data (mesh, metadata) that all ranks need
3. **Rigid configuration**: Can't mix different decomposition patterns for different variables

## Proposed Design

### 1. Variable Types

**Distributed Variables:**
- Domain-decomposed across ranks
- Each rank owns a portion of the data
- Supports ghost layers for stencil operations
- Examples: temperature, pressure, velocity fields

**Replicated Variables:**
- Full data available on all ranks
- No decomposition, no ghost layers
- Efficient for small global data
- Examples: mesh coordinates, timestep info, global status

### 2. Per-Variable Configuration

```cpp
struct variable_config {
  enum class Type { DISTRIBUTED, REPLICATED, AUTO };

  Type type = Type::AUTO;  // AUTO: infer from dimensions/usage

  // For DISTRIBUTED variables:
  std::vector<size_t> dims;      // Variable-specific dimensions (empty = use global)
  std::vector<size_t> decomp;    // Decomposition pattern (empty = auto)
  std::vector<size_t> ghost;     // Ghost layers (empty = no ghosts)

  // Optional: dimension mapping
  // E.g., for 3D vector: dims=[1000,800,600,3], decomp=[4,2,1,0] (don't split last dim)
};
```

### 3. YAML Configuration Format

```yaml
# Default decomposition (applies to all distributed variables unless overridden)
decomposition:
  global_dims: [1000, 800, 600]
  pattern: [4, 2, 1]  # or [] for auto
  ghost: [1, 1, 1]

# Per-variable configuration
variables:
  # Use default decomposition
  temperature:
    type: distributed

  # Custom decomposition for different dimensions
  velocity:
    type: distributed
    decomposition:
      dims: [1000, 800, 600, 3]  # 3D vector field
      pattern: [4, 2, 1, 0]      # Don't decompose vector components
      ghost: [1, 1, 1, 0]

  # Different decomposition strategy
  pressure:
    type: distributed
    decomposition:
      dims: [1000, 800, 600]
      pattern: [2, 4, 1]  # Favor Y-direction decomposition
      ghost: [2, 2, 2]    # Larger ghosts for high-order schemes

  # Replicated (all ranks get full data)
  mesh_coordinates:
    type: replicated

  grid_spacing:
    type: replicated

  timestep_info:
    type: replicated

# Data streams (as before)
streams:
  - name: fields
    format: netcdf
    filenames: simulation_{timestep:04d}.nc
    vars:
      - temperature
      - velocity
      - pressure

  - name: mesh
    format: hdf5
    filenames: mesh.h5
    static: true
    vars:
      - mesh_coordinates
      - grid_spacing

  - name: metadata
    format: yaml
    filenames: timestep_{timestep:04d}.yaml
    vars:
      - timestep_info
```

### 4. New Return Type: mixed_ndarray_group

```cpp
template <typename T = float, typename StoragePolicy = native_storage>
class mixed_ndarray_group {
public:
  using distributed_array_type = distributed_ndarray<T, StoragePolicy>;
  using replicated_array_type = ndarray<T, StoragePolicy>;

  // Access distributed variables
  distributed_array_type& distributed(const std::string& name);
  const distributed_array_type& distributed(const std::string& name) const;

  // Access replicated variables
  replicated_array_type& replicated(const std::string& name);
  const replicated_array_type& replicated(const std::string& name) const;

  // Query variable types
  bool has_distributed(const std::string& name) const;
  bool has_replicated(const std::string& name) const;
  bool has_variable(const std::string& name) const;

  // Iterate over all variables
  std::vector<std::string> distributed_variables() const;
  std::vector<std::string> replicated_variables() const;
  std::vector<std::string> all_variables() const;

  // Add variables
  void add_distributed(const std::string& name, distributed_array_type&& array);
  void add_replicated(const std::string& name, replicated_array_type&& array);

  // Convenience: exchange ghosts on all distributed variables
  void exchange_all_ghosts();

private:
  MPI_Comm comm_;
  std::map<std::string, distributed_array_type> distributed_;
  std::map<std::string, replicated_array_type> replicated_;
};
```

### 5. Updated distributed_stream API

```cpp
template <typename T = float, typename StoragePolicy = native_storage>
class distributed_stream {
public:
  using mixed_group_type = mixed_ndarray_group<T, StoragePolicy>;

  // Constructor
  distributed_stream(MPI_Comm comm = MPI_COMM_WORLD);

  // Configuration
  void parse_yaml(const std::string& yaml_file);

  // Set default decomposition (applies to all distributed vars unless overridden)
  void set_default_decomposition(const std::vector<size_t>& global_dims,
                                 size_t nprocs = 0,
                                 const std::vector<size_t>& decomp = {},
                                 const std::vector<size_t>& ghost = {});

  // Configure individual variables
  void configure_variable(const std::string& name, const variable_config& config);

  // Mark variables as replicated (convenience)
  void mark_replicated(const std::vector<std::string>& var_names);
  void mark_replicated(const std::string& var_name);

  // Read returns mixed group
  std::shared_ptr<mixed_group_type> read(int timestep);
  std::shared_ptr<mixed_group_type> read_static();

  // Iterator interface
  template <typename Callback>
  void for_each_timestep(Callback callback);

  // Query configuration
  variable_config get_variable_config(const std::string& name) const;
  std::vector<std::string> get_distributed_variables() const;
  std::vector<std::string> get_replicated_variables() const;
};
```

### 6. Usage Examples

#### Example 1: Mixed Variables

```cpp
// Configure stream
ftk::distributed_stream<> stream(MPI_COMM_WORLD);
stream.parse_yaml("config.yaml");

// Read timestep
auto vars = stream.read(0);

// Access distributed variables (domain-decomposed)
auto& temp = vars.distributed("temperature");
auto& vel = vars.distributed("velocity");

// Access replicated variables (full data on all ranks)
auto& mesh = vars.replicated("mesh_coordinates");
auto& info = vars.replicated("timestep_info");

// Process distributed data with ghosts
temp.exchange_ghosts();
vel.exchange_ghosts();

// All ranks can access full mesh data
for (size_t i = 0; i < mesh.shape(0); i++) {
  // mesh is the same on all ranks
  double x = mesh.at(i, 0);
  double y = mesh.at(i, 1);
  double z = mesh.at(i, 2);
}

// Process local temperature using global mesh info
auto& local_temp = temp.local_array();
for (size_t i = 0; i < local_temp.shape(0); i++) {
  auto global_idx = temp.local_to_global({i, 0, 0});
  double x = mesh.at(global_idx[0], 0);  // Get global coordinate
  // ... process ...
}
```

#### Example 2: Programmatic Configuration

```cpp
ftk::distributed_stream<> stream(MPI_COMM_WORLD);

// Set default decomposition
stream.set_default_decomposition({1000, 800, 600}, 0, {}, {1, 1, 1});

// Configure temperature (uses default)
stream.configure_variable("temperature", {
  .type = variable_config::Type::DISTRIBUTED
});

// Configure velocity with different decomposition
stream.configure_variable("velocity", {
  .type = variable_config::Type::DISTRIBUTED,
  .dims = {1000, 800, 600, 3},
  .decomp = {4, 2, 1, 0},  // Don't split vector components
  .ghost = {1, 1, 1, 0}
});

// Mark mesh as replicated
stream.mark_replicated("mesh_coordinates");

// Read and process
auto vars = stream.read(0);
vars.exchange_all_ghosts();  // Exchange all distributed variables
```

#### Example 3: Automatic Type Detection

```cpp
// If variable type is AUTO (default), infer from:
// 1. Dimensions: If same as global_dims → distributed
// 2. Size: If small (< threshold) → replicated
// 3. Usage: First access determines type

ftk::distributed_stream<> stream(MPI_COMM_WORLD);
stream.set_default_decomposition({1000, 800, 600});

// Auto-detect: 1000×800×600 → distributed
auto vars = stream.read(0);
auto& temp = vars.distributed("temperature");  // Inferred as distributed

// Auto-detect: small metadata → replicated
auto& info = vars.replicated("timestep_info");  // Inferred as replicated
```

### 7. Implementation Strategy

#### Phase 1: Create mixed_ndarray_group
- New header: `include/ndarray/mixed_ndarray_group.hh`
- Dual storage (distributed + replicated maps)
- Access methods with error handling

#### Phase 2: Add variable_config
- Update `distributed_stream` to store per-variable configs
- Parse YAML variables section
- Default config fallback

#### Phase 3: Update read() method
- Check variable config before creating array
- Create distributed_ndarray or ndarray based on type
- Handle dimension/decomposition per variable

#### Phase 4: Parallel I/O for replicated
- Efficient read strategies:
  - **Rank 0 reads + broadcast**: Simple, good for small data
  - **All ranks read independently**: No communication, redundant I/O
  - **Collective read to rank 0 + broadcast**: Best for parallel filesystems

#### Phase 5: Testing and examples
- Update distributed_stream examples
- Add mixed variable examples
- Performance benchmarks

### 8. Backward Compatibility

**Option A: Breaking change**
- `read()` returns `mixed_ndarray_group` instead of `distributed_ndarray_group`
- Users must update: `vars["temp"]` → `vars.distributed("temp")`

**Option B: Deprecation path**
- Keep old `distributed_ndarray_group` return type
- Add new `read_mixed()` method
- Deprecate `read()` in documentation
- Remove in next major version

**Recommendation: Option A** - Clean break, better design, distributed_stream is new (no existing users)

### 9. Performance Considerations

**Replicated Variables:**
- **Memory**: Small overhead (one copy per rank)
- **I/O**: Rank 0 reads + MPI_Bcast is efficient for small data
- **Use when**: Data size < 10-100 MB and needed by all ranks

**Distributed Variables:**
- **Memory**: Distributed (1/N per rank)
- **I/O**: Parallel I/O for formats that support it
- **Use when**: Large data, only local portion needed

**Hybrid Approach:**
- Mesh (10 MB) → Replicated
- Fields (10 GB) → Distributed
- Total memory: 10 GB/N + 10 MB ≈ 10 GB/N

### 10. Open Questions

1. **Auto-detection threshold**: What size triggers replicated vs distributed for AUTO type?
   - Proposal: < 100 MB → replicated, >= 100 MB → distributed

2. **Replicated I/O strategy**: Which is default?
   - Proposal: Rank 0 reads + MPI_Bcast (simplest, works for all formats)
   - Advanced: Detect parallel format, collective read to rank 0

3. **Dimension validation**: Enforce that replicated vars are "small"?
   - Proposal: Warning only, no hard limit

4. **Static vs time-varying**: Should replicated variables be preferentially static?
   - Proposal: Allow both, but document that time-varying replicated has I/O overhead

5. **Ghost exchange convenience**: Should `exchange_all_ghosts()` be called automatically?
   - Proposal: No, explicit is better (user may want to exchange selectively)

### 11. Unified API for Serial and Parallel (Critical Design Decision!)

**Current Problem:**
- `stream` class for serial execution
- `distributed_stream` class for parallel execution
- Users must choose and write different code

**New Solution:**
Single `stream` class that adapts to MPI configuration automatically

```cpp
// Same code works for serial AND parallel!
ftk::stream<> stream(MPI_COMM_WORLD);
stream.parse_yaml("config.yaml");

auto vars = stream.read(0);

// Behavior adapts automatically:
// - If nprocs == 1: Variables are regular ndarray
// - If nprocs > 1: Variables are distributed/replicated based on config
```

**Implementation:**
- `stream` class detects MPI configuration at runtime
- YAML file includes optional decomposition section
- When nprocs == 1: Decomposition is ignored, everything is regular ndarray
- When nprocs > 1: Decomposition is applied

**Example YAML (works for both serial and parallel):**
```yaml
# Decomposition (only used if nprocs > 1)
decomposition:
  global_dims: [1000, 800, 600]
  pattern: []  # auto
  ghost: [1, 1, 1]

variables:
  temperature:
    type: distributed  # In serial: regular ndarray. In parallel: distributed_ndarray

  mesh:
    type: replicated  # In serial: regular ndarray. In parallel: replicated on all ranks

streams:
  - name: fields
    format: netcdf
    filenames: data_{timestep}.nc
```

**User code (identical for serial and parallel):**
```cpp
#include <ndarray/stream.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Same stream class!
  ftk::stream<> stream(MPI_COMM_WORLD);
  stream.parse_yaml("config.yaml");

  for (int t = 0; t < stream.n_timesteps(); t++) {
    auto vars = stream.read(t);

    // Access pattern is the same
    // Library handles whether it's distributed or not
    auto& temp = vars["temperature"];

    // Ghost exchange: no-op if serial, MPI exchange if parallel
    if (vars.is_distributed("temperature")) {
      vars.distributed("temperature").exchange_ghosts();
    }

    // Process data...
  }

  MPI_Finalize();
  return 0;
}

// Run with: ./program  (serial)
// Run with: mpirun -n 4 ./program  (parallel)
// Same binary, same code!
```

**Benefits:**
✅ **User-friendly**: Write once, run serial or parallel
✅ **Testing**: Test with serial run before scaling to parallel
✅ **Gradual scaling**: Start with 1 rank, scale up without code changes
✅ **Consistency**: Same YAML file for all configurations

**Architectural Decision: How to Unify?**

**Option 1: Type-erased return (std::variant or inheritance)**
```cpp
class stream {
  std::shared_ptr<base_group> read(int timestep);  // Returns polymorphic type
};

// Problem: Users need runtime checks or dynamic_cast
auto vars = stream.read(0);
if (auto* mixed = dynamic_cast<mixed_group*>(vars.get())) {
  mixed->distributed("temp").exchange_ghosts();
}
```
❌ Complex, not user-friendly

**Option 2: Template-based (compile-time)**
```cpp
template <bool Distributed>
class stream { ... };

using serial_stream = stream<false>;
using parallel_stream = stream<true>;
```
❌ Still requires different types, not unified

**Option 3: Unified group with runtime adaptation** ✅ **RECOMMENDED**
```cpp
class ndarray_group {
  // Works for both serial and parallel
  ndarray_base& operator[](const std::string& name);

  // Returns regular ndarray if serial, distributed_ndarray if parallel
  // User doesn't need to know the difference for basic operations
};

// Advanced users can query:
bool is_distributed(const std::string& name) const;
distributed_ndarray<T>& distributed(const std::string& name);
```

**Implementation Strategy:**
1. Extend `ndarray_group` to optionally hold `distributed_ndarray` instances
2. When MPI size == 1: Store regular `ndarray`
3. When MPI size > 1: Store `distributed_ndarray` (which internally holds a local `ndarray`)
4. `operator[]` returns a common base or wrapper that works for both

**Detailed Implementation:**
```cpp
// Common base for type erasure
class ndarray_base {
public:
  virtual ~ndarray_base() = default;
  virtual void* data() = 0;
  virtual const std::vector<size_t>& shape() const = 0;
  // ... common interface ...
};

// ndarray inherits from ndarray_base (or holds it via composition)
template <typename T, typename StoragePolicy>
class ndarray : public ndarray_base { ... };

// distributed_ndarray also provides ndarray_base interface
template <typename T, typename StoragePolicy>
class distributed_ndarray : public ndarray_base {
  ndarray<T, StoragePolicy>& local_array() { return local_data_; }
  // ... implements ndarray_base interface via local_data_ ...
};

// Unified group
template <typename T, typename StoragePolicy>
class ndarray_group {
  std::map<std::string, std::shared_ptr<ndarray_base>> arrays_;

  ndarray_base& operator[](const std::string& name) {
    return *arrays_[name];
  }

  // Type-safe access for distributed arrays
  distributed_ndarray<T, StoragePolicy>& distributed(const std::string& name) {
    auto* dist = dynamic_cast<distributed_ndarray<T, StoragePolicy>*>(arrays_[name].get());
    if (!dist) throw std::runtime_error("Not a distributed array");
    return *dist;
  }

  bool is_distributed(const std::string& name) const {
    return dynamic_cast<distributed_ndarray<T, StoragePolicy>*>(arrays_[name].get()) != nullptr;
  }
};
```

**Alternative: Wrapper approach (simpler)**
```cpp
// Wrapper that adapts to either ndarray or distributed_ndarray
template <typename T, typename StoragePolicy>
class array_handle {
  std::variant<
    ndarray<T, StoragePolicy>*,
    distributed_ndarray<T, StoragePolicy>*
  > ptr_;

public:
  T* data() { /* dispatch to underlying */ }
  const std::vector<size_t>& shape() const { /* dispatch */ }

  bool is_distributed() const { return std::holds_alternative<distributed_ndarray<T>*>(ptr_); }

  distributed_ndarray<T, StoragePolicy>& as_distributed() {
    return *std::get<distributed_ndarray<T, StoragePolicy>*>(ptr_);
  }
};
```

**Recommendation:**
Use the **wrapper approach** with `std::variant` - simpler than inheritance, type-safe, no virtual calls.

The `stream` class internally decides:
- `if (nprocs == 1)`: Create regular `ndarray`, wrap in `array_handle`
- `if (nprocs > 1)`: Create `distributed_ndarray` based on config, wrap in `array_handle`

## Summary

This redesign provides:
✅ **Flexibility**: Different variables can have different decompositions
✅ **Efficiency**: Replicated variables for small global data
✅ **Simplicity**: YAML-driven configuration with sensible defaults
✅ **Power**: Programmatic API for advanced use cases
✅ **Clarity**: Explicit variable types (distributed vs replicated)
✅ **Unified API**: Same code works for serial (1 rank) and parallel (N ranks)

The key insights:
1. **Not all variables should be decomposed the same way**, and some shouldn't be decomposed at all
2. **Users shouldn't need different APIs for serial vs parallel** - the library should adapt automatically
