# Handling Multicomponent and Time-Varying Arrays in Distributed Mode

## Key Principle

**IMPORTANT**: Component and time dimensions are **NOT partitioned** across MPI ranks.

Only **spatial dimensions** are decomposed. Component and time dimensions are **replicated** on all ranks.

## Why This Design?

Arrays can have non-spatial dimensions that should NOT be decomposed:
- **Vector components**: velocity field [1000, 800, 600, 3] where 3 = (vx, vy, vz)
- **Tensor components**: stress tensor [1000, 800, 600, 6] where 6 = symmetric tensor components
- **Time slices**: time-varying data [1000, 800, 600, 100] where 100 = timesteps
- **Multiple variables**: combined storage [1000, 800, 600, 5] where 5 = (temp, pressure, density, ...)

**Key requirement**: Domain decomposition splits only spatial dimensions, leaving component and time dimensions intact on each rank.

**Benefits**:
- ✅ All vector/tensor components available locally (no communication for vector operations)
- ✅ All timesteps available locally (enables temporal analysis without communication)
- ✅ Cache-friendly: components at each spatial point are contiguous in memory
- ✅ Natural for physics: compute full vectors, divergence, curl, etc. at any local point

## Solution: Explicit Decomposition Pattern

### 1. Extended decompose() API

```cpp
void decompose(MPI_Comm comm,
               const std::vector<size_t>& global_dims,
               size_t nprocs = 0,
               const std::vector<size_t>& decomp = {},
               const std::vector<size_t>& ghost = {});
```

**Decomposition pattern interpretation:**
- `decomp[i] > 0`: **Split** dimension i into decomp[i] pieces (spatial dimensions)
- `decomp[i] == 0`: **DON'T split** dimension i - replicate on all ranks (components, time)
- `decomp[i] == 1`: **DON'T split** dimension i - explicitly specified (equivalent to 0)

**Default behavior**: Use `decomp[i] = 0` for component and time dimensions.

### 2. Examples

#### Example 1: 3D Velocity Field (Spatial + Vector)

```cpp
// Velocity: [1000, 800, 600, 3] where last dim = (vx, vy, vz)
ftk::ndarray<float> velocity;

// Decompose only spatial dimensions (first 3), keep vector components intact
velocity.decompose(MPI_COMM_WORLD,
                   {1000, 800, 600, 3},        // global dims
                   0,                           // nprocs = all
                   {4, 2, 1, 0},               // split x, y, z; don't split components
                   {1, 1, 1, 0});              // ghosts only in spatial dims

// Result with 8 ranks:
// - Each rank has local spatial portion: [250, 400, 600, 3]
// - All ranks have full 3 vector components
// - Ghost exchange only in spatial dimensions
```

#### Example 2: Time-Varying Temperature Field

```cpp
// Temperature over time: [1000, 800, 600, 100] where last dim = timesteps
ftk::ndarray<float> temp_time;

// Decompose spatial dimensions, keep all timesteps on each rank
temp_time.decompose(MPI_COMM_WORLD,
                    {1000, 800, 600, 100},      // global dims
                    0,                           // nprocs = all
                    {4, 2, 1, 0},               // split x, y, z; keep timesteps
                    {1, 1, 1, 0});              // no ghosts in time

// Result with 8 ranks:
// - Each rank has: [250, 400, 600, 100]
// - Spatial domain split, but all 100 timesteps available locally
// - Useful for temporal analysis without communication
```

#### Example 3: Stress Tensor Field

```cpp
// Stress tensor: [1000, 800, 600, 6] where 6 = (σxx, σyy, σzz, σxy, σxz, σyz)
ftk::ndarray<double> stress;

stress.decompose(MPI_COMM_WORLD,
                 {1000, 800, 600, 6},
                 0,
                 {4, 2, 1, 0},    // Don't split tensor components
                 {1, 1, 1, 0});
```

#### Example 4: 2D Field with Time (Different Order)

```cpp
// Time-varying 2D field: [100, 1000, 800] where first dim = timesteps
ftk::ndarray<float> temp_2d;

// Keep time dimension intact, split spatial dimensions
temp_2d.decompose(MPI_COMM_WORLD,
                  {100, 1000, 800},
                  0,
                  {0, 4, 2},      // Don't split time, split y and x
                  {0, 1, 1});     // No ghosts in time

// Result with 8 ranks:
// - Each rank has: [100, 250, 400]
// - All timesteps available locally
```

## Implementation Changes Needed

### 1. Update lattice_partitioner

Currently, lattice_partitioner may assume all dimensions are decomposed equally. Need to support:
- `decomp[i] == 0`: Dimension i is NOT decomposed (all ranks have full extent)
- Automatic decomposition should only consider dimensions with `decomp[i] > 0`

### 2. Update decompose() Logic

```cpp
void decompose(MPI_Comm comm,
               const std::vector<size_t>& global_dims,
               size_t nprocs,
               const std::vector<size_t>& decomp,
               const std::vector<size_t>& ghost)
{
  // Identify spatial dimensions (those to be decomposed)
  std::vector<int> spatial_dims;
  for (size_t i = 0; i < global_dims.size(); i++) {
    if (decomp.empty() || i >= decomp.size() || decomp[i] > 0) {
      spatial_dims.push_back(i);
    }
  }

  // If decomp is provided, validate it
  if (!decomp.empty() && decomp.size() != global_dims.size()) {
    throw std::invalid_argument("decomp size must match global_dims size");
  }

  // Create effective decomposition pattern
  std::vector<size_t> effective_decomp;
  if (decomp.empty()) {
    // Auto-decompose only spatial dimensions
    effective_decomp.resize(global_dims.size(), 0);
    // Use lattice_partitioner for auto-decomposition of spatial dims only
  } else {
    effective_decomp = decomp;
  }

  // For non-decomposed dimensions:
  // - local_core.start(i) = 0
  // - local_core.size(i) = global_dims[i]
  // - local_extent = local_core (no ghosts)
}
```

### 3. Update Ghost Exchange

Ghost exchange should only operate on decomposed spatial dimensions:

```cpp
void setup_ghost_exchange()
{
  for (size_t dim = 0; dim < global_dims.size(); dim++) {
    // Skip non-decomposed dimensions
    if (decomp_pattern_[dim] == 0) continue;

    // Check for neighbors in this dimension
    // ...
  }
}
```

### 4. Update Pack/Unpack

When packing boundary data, only iterate over spatial dimensions with ghosts:

```cpp
void pack_boundary_data(...)
{
  // For non-decomposed dimensions, include full extent
  // For decomposed dimensions, pack only boundary layer
}
```

## YAML Configuration

```yaml
decomposition:
  global_dims: [1000, 800, 600, 3]  # Spatial + vector
  pattern: [4, 2, 1, 0]              # Split spatial, not vector
  ghost: [1, 1, 1, 0]                # Ghosts in spatial only

variables:
  velocity:
    type: distributed
    decomposition:
      dims: [1000, 800, 600, 3]
      pattern: [4, 2, 1, 0]  # Explicit: don't decompose last dimension
      ghost: [1, 1, 1, 0]

  temperature_over_time:
    type: distributed
    decomposition:
      dims: [1000, 800, 600, 100]
      pattern: [4, 2, 1, 0]  # 100 timesteps available on each rank
      ghost: [1, 1, 1, 0]
```

## Special Cases

### Case 1: Time Series Processing

For truly time-varying data where each rank processes different timesteps:
- Store spatial slices separately
- Use separate ndarray per timestep
- Or use specialized time-distributed class (future enhancement)

### Case 2: Hybrid Decomposition

Some applications may want to decompose both space AND time:
- Rare use case
- Each rank gets spatial subdomain AND time slice
- Would need different YAML pattern

## Benefits

✅ **Natural**: Spatial decomposition is what users expect
✅ **Efficient**: No communication for vector/tensor components
✅ **Flexible**: Supports any dimension ordering
✅ **Clear**: Decomposition pattern is explicit in YAML/code
✅ **Compatible**: Works with existing lattice_partitioner (with minor updates)

## Testing Requirements

1. 3D vector field with ghost exchange
2. Time-varying 2D field
3. Tensor field (6 components)
4. Mixed: velocity [x, y, z, vx, vy, vz] decomposed as [4, 2, 1, 0, 0, 0]
5. Different dimension orderings (time first vs time last)

## Implementation Priority

**High Priority** (needed for correct behavior):
1. Support `decomp[i] == 0` in decompose()
2. Validate decomp size matches global_dims size
3. Update ghost exchange to skip non-decomposed dims
4. Update pack/unpack for non-decomposed dims

**Medium Priority** (usability):
1. YAML parsing for per-variable decomposition patterns
2. Auto-detection: guess which dims are spatial (first 2-3 dims)
3. Better error messages for invalid decomposition

**Low Priority** (future):
1. Time-distributed decomposition (spatial + temporal splitting)
2. Hybrid decomposition strategies
3. Adaptive decomposition
