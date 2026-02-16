# Distributed Array Indexing Clarification

## Summary

**For distributed arrays, all direct data access uses LOCAL indices.**

Global indices are only used for:
1. Checking if a point is owned by this rank: `is_local(global_idx)`
2. Converting between coordinate systems: `global_to_local()` / `local_to_global()`

## Index Coordinate Systems

### Local Indices (What You Use for Data Access)

```cpp
ftk::ndarray<float> temp;
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

// Access data with LOCAL indices
auto& local = temp.local_array();
for (size_t i = 0; i < temp.local_core().size(0); i++) {      // i is LOCAL
  for (size_t j = 0; j < temp.local_core().size(1); j++) {    // j is LOCAL
    float value = local.at(i, j);  // or local.f(i, j) or local(i, j)
    // ...
  }
}
```

**Key point**: `i` and `j` range from `0` to `local_core().size(d)`, NOT from `0` to `global_dims[d]`.

### Global Indices (What You Use for Coordinate Conversion)

```cpp
// Check if global point [500, 400] is on this rank
std::vector<size_t> global_point = {500, 400};

if (temp.is_local(global_point)) {
  // Convert to local index
  auto local_idx = temp.global_to_local(global_point);

  // NOW access with local index
  float value = temp.local_array().at(local_idx[0], local_idx[1]);

  std::cout << "Global [500, 400] maps to local ["
            << local_idx[0] << ", " << local_idx[1] << "]" << std::endl;
}
```

## Complete Example: Both Index Systems

```cpp
#include <ndarray/ndarray.hh>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Create distributed array
  ftk::ndarray<float> temp;
  temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

  // Read data (automatic parallel I/O)
  temp.read_netcdf_auto("input.nc", "temperature");

  // ========================================================================
  // Method 1: LOCAL indexing (typical use case)
  // ========================================================================

  std::cout << "Rank " << rank << " processing local data:" << std::endl;

  float local_sum = 0.0f;
  size_t local_count = 0;

  auto& local = temp.local_array();

  // Loop over LOCAL core region (no ghosts)
  for (size_t i = 0; i < temp.local_core().size(0); i++) {      // LOCAL i
    for (size_t j = 0; j < temp.local_core().size(1); j++) {    // LOCAL j
      float value = local.at(i, j);  // Access with LOCAL indices
      local_sum += value;
      local_count++;

      // Optional: compute corresponding global index
      size_t global_i = temp.local_core().start(0) + i;
      size_t global_j = temp.local_core().start(1) + j;

      // Now you know: local[i,j] corresponds to global[global_i, global_j]
    }
  }

  std::cout << "  Local mean: " << (local_sum / local_count) << std::endl;

  // ========================================================================
  // Method 2: GLOBAL indexing (when you have specific global coordinates)
  // ========================================================================

  // Example: Check specific global points
  std::vector<std::vector<size_t>> global_points = {
    {500, 400},  // Center of domain
    {0, 0},      // Corner
    {999, 799}   // Other corner
  };

  for (const auto& global_pt : global_points) {
    if (temp.is_local(global_pt)) {
      // This rank owns this global point
      auto local_pt = temp.global_to_local(global_pt);
      float value = local.at(local_pt[0], local_pt[1]);

      std::cout << "Rank " << rank << " has global ["
                << global_pt[0] << ", " << global_pt[1]
                << "] at local [" << local_pt[0] << ", " << local_pt[1]
                << "] with value " << value << std::endl;
    }
  }

  // ========================================================================
  // Method 3: MIXED (typical for algorithms with global coordinates)
  // ========================================================================

  // Example: Find all points where global_i + global_j > 1000

  for (size_t i = 0; i < temp.local_core().size(0); i++) {
    for (size_t j = 0; j < temp.local_core().size(1); j++) {
      // Convert local to global
      auto global_idx = temp.local_to_global({i, j});
      size_t global_i = global_idx[0];
      size_t global_j = global_idx[1];

      // Apply condition based on GLOBAL coordinates
      if (global_i + global_j > 1000) {
        // Access data with LOCAL indices
        float value = local.at(i, j);
        // ... process ...
      }
    }
  }

  // ========================================================================
  // Method 4: CONVENIENCE - Direct global access (NEW!)
  // ========================================================================

  // NEW convenience methods: at_global(), f_global(), c_global()
  // These do the conversion internally - simpler syntax!

  std::vector<std::vector<size_t>> points_of_interest = {
    {500, 400},  // Center
    {0, 0},      // Corner
    {999, 799}   // Other corner
  };

  for (const auto& pt : points_of_interest) {
    try {
      // Direct access with global indices (throws if not local)
      float value = temp.at_global(pt[0], pt[1]);

      std::cout << "Global [" << pt[0] << ", " << pt[1]
                << "] = " << value << std::endl;

      // Can also write with global indices
      temp.at_global(pt[0], pt[1]) = value * 2.0f;

    } catch (const std::out_of_range& e) {
      // This global point is not on this rank
    }
  }

  // Also available: f_global() and c_global() for explicit ordering
  if (temp.is_local({500, 400})) {
    float val_f = temp.f_global(500, 400);  // Fortran-order
    float val_c = temp.c_global(500, 400);  // C-order
  }

  MPI_Finalize();
  return 0;
}
```

## Coordinate System Diagram

```
Global domain: [0, 1000) × [0, 800)

Decomposition with 4 ranks (2×2):

Rank 0: global [0, 500) × [0, 400)     → local [0, 500) × [0, 400)
Rank 1: global [500, 1000) × [0, 400)  → local [0, 500) × [0, 400)
Rank 2: global [0, 500) × [400, 800)   → local [0, 500) × [0, 400)
Rank 3: global [500, 1000) × [400, 800) → local [0, 500) × [0, 400)

With 1-layer ghosts, local arrays are actually [502] × [402]
(core [500×400] + ghosts [1 on each side])
```

**Example on Rank 1**:
- Global index [600, 200] → Local index [100, 200]
  - Because: 600 - 500 (core start) = 100
  - And: 200 - 0 (core start) = 200

- Local index [100, 200] → Global index [600, 200]
  - Because: 100 + 500 (core start) = 600
  - And: 200 + 0 (core start) = 200

## Ghost Regions and Local Indexing

Ghost regions extend the local array beyond the core:

```cpp
ftk::ndarray<float> temp;
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

// Say you're on a rank with:
// - Core: starts at global [500, 400], size [500, 400]
// - Ghosts: 1 layer on each side

auto core = temp.local_core();
auto extent = temp.local_extent();

std::cout << "Core start: [" << core.start(0) << ", " << core.start(1) << "]" << std::endl;
// Output: Core start: [500, 400]

std::cout << "Core size: [" << core.size(0) << ", " << core.size(1) << "]" << std::endl;
// Output: Core size: [500, 400]

std::cout << "Extent start: [" << extent.start(0) << ", " << extent.start(1) << "]" << std::endl;
// Output: Extent start: [499, 399]  (1 before core start)

std::cout << "Extent size: [" << extent.size(0) << ", " << extent.size(1) << "]" << std::endl;
// Output: Extent size: [502, 402]  (core + 2 ghosts per dimension)

// Local array size matches EXTENT (includes ghosts)
std::cout << "Local array: [" << temp.local_array().dim(0)
          << ", " << temp.local_array().dim(1) << "]" << std::endl;
// Output: Local array: [502, 402]
```

**Accessing ghosts vs core**:

```cpp
auto& local = temp.local_array();

// Access core data (typically what you compute on)
for (size_t i = 0; i < core.size(0); i++) {
  for (size_t j = 0; j < core.size(1); j++) {
    float value = local.at(i, j);  // LOCAL indices into core
    // ...
  }
}

// Access including ghosts (e.g., for stencil operations after exchange_ghosts())
for (size_t i = 1; i < local.dim(0) - 1; i++) {  // Skip boundary ghosts
  for (size_t j = 1; j < local.dim(1) - 1; j++) {
    // 5-point stencil (needs ghosts)
    float laplacian = local.at(i-1, j) + local.at(i+1, j) +
                      local.at(i, j-1) + local.at(i, j+1) - 4.0f * local.at(i, j);
    // ...
  }
}
```

## API Summary

### Query Methods
```cpp
// Check distribution state
bool is_distributed() const;      // Is this array decomposed?
bool is_replicated() const;       // Is this array replicated across all ranks?

// Get lattice regions
const lattice& global_lattice() const;   // Full global domain
const lattice& local_core() const;       // This rank's owned region (no ghosts)
const lattice& local_extent() const;     // This rank's full region (with ghosts)

// MPI info
int rank() const;                  // MPI rank
int nprocs() const;               // Total number of ranks
MPI_Comm comm() const;            // MPI communicator
```

### Index Conversion
```cpp
// Convert global to local (throws if not in local core)
std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const;

// Convert local to global
std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const;

// Check if global index is in local core
bool is_local(const std::vector<size_t>& global_idx) const;
```

### Data Access with LOCAL Indices
```cpp
// Get local array (includes ghosts)
ndarray<T>& local_array();
const ndarray<T>& local_array() const;

// Access with local indices (on local_array())
T& at(size_t i0, size_t i1, ...);           // Fortran-order
T& f(size_t i0, size_t i1, ...);            // Fortran-order (explicit)
T& c(size_t i0, size_t i1, ...);            // C-order
T& operator()(size_t i0, size_t i1, ...);   // Default order
```

### Data Access with GLOBAL Indices (NEW - Convenience)
```cpp
// Direct access with global indices (throws if not in local core)
// These methods internally call global_to_local() then access local_array()

// Fortran-order (column-major)
T& at_global(size_t i0, size_t i1, ...);      // 1D, 2D, 3D, 4D
const T& at_global(size_t i0, size_t i1, ...) const;

// Explicit Fortran-order
T& f_global(size_t i0, size_t i1, ...);       // 1D, 2D, 3D, 4D
const T& f_global(size_t i0, size_t i1, ...) const;

// C-order (row-major)
T& c_global(size_t i0, size_t i1, ...);       // 1D, 2D, 3D, 4D
const T& c_global(size_t i0, size_t i1, ...) const;

// Example usage:
float value = temp.at_global(500, 400);       // Read with global indices
temp.at_global(500, 400) = 273.15f;           // Write with global indices

// Throws std::out_of_range if global index not in local core
// Throws std::runtime_error if array not distributed
```

## Common Patterns

### Pattern 1: Process All Local Data
```cpp
// No need for global indices, just iterate locally
for (size_t i = 0; i < temp.local_core().size(0); i++) {
  for (size_t j = 0; j < temp.local_core().size(1); j++) {
    float value = temp.local_array().at(i, j);
    // ... process ...
  }
}
```

### Pattern 2: Check Specific Global Points (Manual Conversion)
```cpp
std::vector<size_t> global_point = {500, 400};
if (temp.is_local(global_point)) {
  auto local_point = temp.global_to_local(global_point);
  float value = temp.local_array().at(local_point[0], local_point[1]);
  // ... process this specific point ...
}
```

### Pattern 2b: Direct Global Access (NEW - Convenience)
```cpp
// Simpler syntax with at_global() - does conversion internally
try {
  float value = temp.at_global(500, 400);    // Read
  temp.at_global(500, 400) = value * 2.0f;   // Write
  // ... process ...
} catch (const std::out_of_range& e) {
  // Global index not in local core (this rank doesn't own it)
}

// Or check first to avoid exception
if (temp.is_local({500, 400})) {
  float value = temp.at_global(500, 400);  // Safe - won't throw
}
```

### Pattern 3: Apply Global Coordinate-Based Logic
```cpp
for (size_t i = 0; i < temp.local_core().size(0); i++) {
  for (size_t j = 0; j < temp.local_core().size(1); j++) {
    // Get global coordinates
    auto global_idx = temp.local_to_global({i, j});

    // Apply logic based on global position
    if (global_idx[0] < 100) {  // Left boundary condition
      temp.local_array().at(i, j) = 0.0f;
    } else {
      // Regular processing
      float value = temp.local_array().at(i, j);
      // ...
    }
  }
}
```

### Pattern 4: Stencil with Ghost Exchange
```cpp
// Exchange ghost cells
temp.exchange_ghosts();

auto& local = temp.local_array();

// Apply stencil (can access ghosts)
for (size_t i = 1; i < local.dim(0) - 1; i++) {    // LOCAL extent indices
  for (size_t j = 1; j < local.dim(1) - 1; j++) {
    float laplacian = (local.at(i-1, j) + local.at(i+1, j) +
                       local.at(i, j-1) + local.at(i, j+1) -
                       4.0f * local.at(i, j));
    // ...
  }
}
```

## Key Takeaways

1. **Default: Local indexing for performance**: `temp.local_array().at(i, j)` where `i` and `j` are local

2. **NEW: Convenience methods for global access**: `temp.at_global(500, 400)` - does conversion internally

3. **Global indices for coordinate conversion**: Use `global_to_local()` manually when needed, or use `at_global()` for convenience

4. **Loop bounds use local sizes**: `for (i = 0; i < temp.local_core().size(0); i++)`

5. **Manual conversion when needed**: `global_i = temp.local_core().start(0) + local_i`

6. **Check ownership first**: `if (temp.is_local(global_point))` before accessing, or catch `std::out_of_range`

7. **Ghost regions are part of local array**: `local_array().dim(0)` includes ghosts, `local_core().size(0)` does not

## Why This Design?

**Performance**: Local indexing is zero-overhead. Direct array access without translation.

**Simplicity**: Most code just processes local data without caring about global coordinates.

**Flexibility**: When you need global coordinates (e.g., boundary conditions, coordinate-based logic), conversion methods are available.

**Compatibility**: Same indexing model as most distributed array libraries (PETSc, Trilinos, etc.).
