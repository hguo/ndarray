# Time Dimension in ndarray

This document clarifies how the time dimension works in ndarray through the `tv` flag.

## Core Concept

The `tv` (time-varying) flag indicates whether the **last dimension** of an array represents time:

```cpp
bool tv = false;  // From ndarray_base.hh:298
```

When `tv = true`, the array represents time-series data where the last dimension indexes timesteps.

## Dimension Ordering

The complete dimension layout is:

```
[component_dims... , spatial_dims... , time_dim]
 <----- ncd -----> <-- nd()-ncd-1 --> <-- tv -->
```

**Examples:**

1. **Scalar field time series** (ncd=0, tv=true):
   ```cpp
   temperature.reshapef(100, 200, 50);  // Shape: [nx, ny, nt]
   temperature.set_has_time(true);
   // Dimensions: 100x200 spatial grid over 50 timesteps
   ```

2. **Vector field time series** (ncd=1, tv=true):
   ```cpp
   velocity.reshapef(3, 100, 200, 50);  // Shape: [components, nx, ny, nt]
   velocity.set_multicomponents(1);
   velocity.set_has_time(true);
   // Dimensions: 3-component velocity on 100x200 grid over 50 timesteps
   ```

3. **Static vector field** (ncd=1, tv=false):
   ```cpp
   velocity.reshapef(3, 100, 200);  // Shape: [components, nx, ny]
   velocity.set_multicomponents(1);
   // Dimensions: 3-component velocity on 100x200 grid (single snapshot)
   ```

## API Functions

### Setting/Getting Time Flag

```cpp
void set_has_time(bool b);   // Mark last dimension as time
bool has_time() const;       // Check if last dimension is time
```

### Time Slicing

Extract individual timesteps or all timesteps:

```cpp
// Extract single timestep (returns (n-1)-dimensional array)
ndarray<T> slice_time(size_t t) const;

// Extract all timesteps (returns vector of (n-1)-dimensional arrays)
std::vector<ndarray<T>> slice_time() const;
```

## Time Slicing Implementation

The `slice_time()` function removes the last dimension:

```cpp
template <typename T>
ndarray<T> ndarray<T>::slice_time(size_t t) const {
  ndarray<T> array;
  std::vector<size_t> mydims(dims);
  mydims.resize(nd()-1);  // Remove last dimension

  array.reshapef(mydims);
  memcpy(&array[0], &p[t * s[nd()-1]], s[nd()-1] * sizeof(T));

  return array;
}
```

**Key points:**
- Copies a contiguous block of memory for the specified timestep
- Resulting array has `nd() - 1` dimensions
- Uses stride `s[nd()-1]` to calculate memory offset

## NetCDF Integration

NetCDF files often use an **unlimited dimension** for time. When reading:

```cpp
void read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm) {
  // ... read variable dimensions ...
  st[0] = t;    // Start at timestep t
  sz[0] = 1;    // Read only one timestep

  read_netcdf(ncid, varid, st, sz, comm);
  set_has_time(true);  // Mark as time-series data
}
```

The first dimension in NetCDF is typically time (unlimited dimension), which becomes the **last dimension** in ndarray due to Fortran-order storage.

## Examples

### Example 1: Temperature Time Series

```cpp
ndarray<double> temp;
temp.reshapef(128, 128, 100);  // 128x128 grid, 100 timesteps
temp.set_has_time(true);

// Access specific point over time
for (size_t t = 0; t < 100; t++) {
  double T = temp.f(64, 64, t);
  std::cout << "t=" << t << ", T=" << T << std::endl;
}

// Extract single timestep
ndarray<double> temp_t50 = temp.slice_time(50);
// temp_t50.shapef() = [128, 128]
// temp_t50.has_time() = false

// Extract all timesteps
std::vector<ndarray<double>> all_timesteps = temp.slice_time();
// all_timesteps.size() = 100
// all_timesteps[i].shapef() = [128, 128]
```

### Example 2: Velocity Field Time Series

```cpp
ndarray<float> velocity;
velocity.reshapef(3, 64, 64, 64, 50);  // 3D velocity over 50 timesteps
velocity.set_multicomponents(1);
velocity.set_has_time(true);

std::cout << "nd() = " << velocity.nd() << std::endl;  // 5
std::cout << "ncd = " << velocity.multicomponents() << std::endl;  // 1
std::cout << "tv = " << velocity.has_time() << std::endl;  // true
std::cout << "Spatial dims: 64x64x64" << std::endl;
std::cout << "Timesteps: 50" << std::endl;

// Access velocity at specific point and time
float vx_t30 = velocity.f(0, 32, 32, 32, 30);  // x-component at center, t=30
float vy_t30 = velocity.f(1, 32, 32, 32, 30);  // y-component
float vz_t30 = velocity.f(2, 32, 32, 32, 30);  // z-component

// Extract velocity field at t=30
ndarray<float> vel_t30 = velocity.slice_time(30);
// vel_t30.shapef() = [3, 64, 64, 64]
// vel_t30.multicomponents() = 1
// vel_t30.has_time() = false  (no longer time-series)
```

### Example 3: NetCDF Time Series

```cpp
#if NDARRAY_HAVE_NETCDF
// Open NetCDF file with time dimension
int ncid;
nc_open("ocean_data.nc", NC_NOWRITE, &ncid);

// Read single timestep
ndarray<double> sst;
sst.read_netcdf_timestep(ncid, "sst", 100);  // Read timestep 100
// sst automatically has tv=true

// Read all timesteps
ndarray<double> sst_all;
sst_all.read_netcdf(ncid, "sst");
sst_all.set_has_time(true);

// Extract specific times
ndarray<double> sst_t50 = sst_all.slice_time(50);
ndarray<double> sst_t100 = sst_all.slice_time(100);
#endif
```

### Example 4: Processing Time Series

```cpp
ndarray<double> data;
data.reshapef(256, 256, 200);
data.set_has_time(true);

// Compute time-average
ndarray<double> mean;
mean.reshapef(256, 256);

for (size_t t = 0; t < 200; t++) {
  ndarray<double> snapshot = data.slice_time(t);
  for (size_t j = 0; j < 256; j++) {
    for (size_t i = 0; i < 256; i++) {
      mean.f(i, j) += snapshot.f(i, j) / 200.0;
    }
  }
}

// Or extract all and process
std::vector<ndarray<double>> snapshots = data.slice_time();
for (const auto& snap : snapshots) {
  // Process each snapshot
  double max_val = *std::max_element(snap.begin(), snap.end());
  std::cout << "Max: " << max_val << std::endl;
}
```

## Multilinear Interpolation

The `mlerp()` function understands both component and time dimensions:

```cpp
template <typename F>
bool mlerp(const F x[], T v[]) const {
  const size_t n = nd() - multicomponents();  // Excludes components AND time
  const size_t nc = multicomponents();

  // Interpolation happens in spatial dimensions only
  // Time is not interpolated - you must slice first
}
```

**Note:** If you need temporal interpolation, extract adjacent timesteps first:

```cpp
// Interpolate between t=10 and t=11
ndarray<T> t10 = data.slice_time(10);
ndarray<T> t11 = data.slice_time(11);

double alpha = 0.3;  // 30% towards t=11
T result = t10.at(x, y, z) * (1-alpha) + t11.at(x, y, z) * alpha;
```

## Streaming and Time

In streaming contexts, the time dimension is crucial:

```cpp
// From ndarray_group_stream.hh
struct stream {
  int total_timesteps() const;  // Total timesteps across all files
  std::shared_ptr<ndarray_group> read(int t);  // Read specific timestep
};

// NetCDF streams track unlimited time dimension
struct substream_netcdf {
  bool has_unlimited_time_dimension = false;
  std::vector<int> timesteps_per_file;
  int total_timesteps = 0;
};
```

## Memory Layout

Time being the **last dimension** affects memory layout:

For a scalar field `[nx, ny, nt]` with column-major (Fortran) ordering:
```
Memory: [t0: all spatial points, t1: all spatial points, ..., tN: all spatial points]
```

This is optimal for:
- **Spatial processing**: All points at one timestep are contiguous
- **Time-stepping simulations**: Each timestep is a contiguous block

Not optimal for:
- **Time-series analysis at a point**: Points at different times are far apart in memory

### Accessing Time Series at a Point

```cpp
// SLOW: Time values are strided by s[nd()-1]
std::vector<double> time_series_at_point;
for (size_t t = 0; t < nt; t++) {
  time_series_at_point.push_back(data.f(x, y, t));  // Large stride between accesses
}

// FASTER: Slice each timestep (better cache locality per timestep)
std::vector<double> time_series_at_point;
for (size_t t = 0; t < nt; t++) {
  ndarray<double> snapshot = data.slice_time(t);
  time_series_at_point.push_back(snapshot.f(x, y));
}
```

## Design Rationale

### Why Last Dimension for Time?

1. **Fortran Convention**: Scientific codes and NetCDF use time as first dimension (column-major), which becomes last in ndarray
2. **Spatial Locality**: Each timestep is contiguous in memory, efficient for spatial operations
3. **Incremental I/O**: Can append new timesteps without reorganizing existing data
4. **Streaming**: Timesteps can be processed sequentially from disk

### Relationship to NetCDF Unlimited Dimension

NetCDF's **unlimited dimension** (typically time) allows files to grow:

```
NetCDF (column-major):  [time, z, y, x]
                         ^^^^^^
                         unlimited dimension

ndarray (column-major):  [x, y, z, time]
                                   ^^^^^^
                                   last dimension (tv=true)
```

The unlimited dimension becomes the last dimension in ndarray, making it natural to set `tv=true`.

## Best Practices

1. **Always set tv for time-series data**:
   ```cpp
   data.reshapef(nx, ny, nz, nt);
   data.set_has_time(true);  // Don't forget!
   ```

2. **Use slice_time() for temporal operations**:
   ```cpp
   // Extract timesteps for processing
   auto t1 = data.slice_time(10);
   auto t2 = data.slice_time(11);
   auto diff = t2 - t1;  // Temporal difference
   ```

3. **Check tv flag before assuming time dimension**:
   ```cpp
   if (data.has_time()) {
     size_t nt = data.dimf(data.nd() - 1);
     std::cout << "Timesteps: " << nt << std::endl;
   }
   ```

4. **Preserve tv when copying**:
   ```cpp
   ndarray<T> copy = original;  // tv is copied automatically
   ```

5. **For point time-series, consider transposing**:
   ```cpp
   // If you need frequent time-series access at points,
   // consider storing [nt, nx, ny] instead of [nx, ny, nt]
   // But this breaks convention and may impact spatial operations
   ```

## Combining ncd and tv

You can have both multicomponent and time-varying arrays:

```cpp
ndarray<float> velocity_timeseries;
velocity_timeseries.reshapef(3, 128, 128, 128, 100);
velocity_timeseries.set_multicomponents(1);  // First dim is components
velocity_timeseries.set_has_time(true);       // Last dim is time

// Dimensions breakdown:
// [0] = 3         : components (vx, vy, vz)
// [1] = 128       : x spatial dimension
// [2] = 128       : y spatial dimension
// [3] = 128       : z spatial dimension
// [4] = 100       : time dimension

std::cout << "nd() = " << velocity_timeseries.nd() << std::endl;  // 5
std::cout << "ncd = " << velocity_timeseries.multicomponents() << std::endl;  // 1
std::cout << "tv = " << velocity_timeseries.has_time() << std::endl;  // true
std::cout << "Spatial dims: " << 128*128*128 << " points" << std::endl;
std::cout << "Components: 3, Timesteps: 100" << std::endl;

// Extract velocity field at t=50
ndarray<float> vel_t50 = velocity_timeseries.slice_time(50);
// vel_t50.shapef() = [3, 128, 128, 128]
// vel_t50.multicomponents() = 1  (preserved)
// vel_t50.has_time() = false  (removed)
```

## Common Pitfalls

### Pitfall 1: Forgetting to set tv

```cpp
ndarray<double> temp;
temp.reshapef(100, 100, 50);
// temp.has_time() = false  <-- WRONG! Should be true if last dim is time
```

**Fix:**
```cpp
temp.reshapef(100, 100, 50);
temp.set_has_time(true);  // Correct!
```

### Pitfall 2: Wrong index order with time

```cpp
// WRONG: Time should be last index
double T = data.f(t, x, y);

// CORRECT: Time is last
double T = data.f(x, y, t);
```

### Pitfall 3: Assuming time is always present

```cpp
// BAD: Assumes time dimension exists
size_t nt = data.dimf(data.nd() - 1);

// GOOD: Check first
if (data.has_time()) {
  size_t nt = data.dimf(data.nd() - 1);
}
```

### Pitfall 4: slice_time doesn't preserve tv

```cpp
ndarray<T> snapshot = timeseries.slice_time(10);
// snapshot.has_time() = false  (time dimension removed)

// Don't expect:
ndarray<T> t0 = snapshot.slice_time(0);  // ERROR: snapshot has no time dim
```

## API Summary

```cpp
// Setting/getting
void set_has_time(bool b);
bool has_time() const;

// Time slicing
ndarray<T> slice_time(size_t t) const;              // Extract timestep t
std::vector<ndarray<T>> slice_time() const;         // Extract all timesteps

// NetCDF I/O with time
void read_netcdf_timestep(int ncid, int varid, int t, MPI_Comm comm);
```

## See Also

- [MULTICOMPONENT_ARRAYS.md](MULTICOMPONENT_ARRAYS.md) - Component dimensions (ncd)
- [ARRAY_INDEXING.md](ARRAY_INDEXING.md) - F/C ordering semantics
- [VTK_TESTS.md](VTK_TESTS.md) - VTK integration (typically time=0 snapshot)
