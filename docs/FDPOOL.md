# NetCDF File Descriptor Pool (fdpool)

## Overview

The `fdpool_nc` class implements a **singleton file descriptor pool** for NetCDF files. Its primary purpose is to **prevent opening the same NetCDF file multiple times simultaneously**, which can lead to corruption, resource exhaustion, and undefined behavior.

## Problem Statement

### Why is a File Descriptor Pool Needed?

When working with NetCDF files in complex applications (especially with streaming workflows), the same file may need to be accessed multiple times:

1. **Multiple variables from same file**
   ```cpp
   // Without fdpool: Opens file 3 times!
   ndarray<float> temp = read_netcdf("data.nc", "temperature");
   ndarray<float> press = read_netcdf("data.nc", "pressure");
   ndarray<float> humid = read_netcdf("data.nc", "humidity");
   ```

2. **Multiple timesteps from same file**
   ```cpp
   // Without fdpool: Opens file 100 times!
   for (int t = 0; t < 100; t++) {
     ndarray<float> data = read_netcdf_timestep("data.nc", "field", t);
   }
   ```

3. **Multiple substreams referencing same file**
   ```yaml
   stream:
     substreams:
       - variables: [temperature, pressure]
         filenames: [data.nc]  # Both vars from same file
       - variables: [velocity_x, velocity_y]
         filenames: [data.nc]  # Same file again!
   ```

### Consequences of Multiple Opens

Opening the same NetCDF file multiple times can cause:

- **File Corruption**: Concurrent writes or metadata modifications
- **Race Conditions**: Undefined behavior with parallel NetCDF
- **Resource Exhaustion**: Running out of file descriptors (ulimit)
- **Performance Degradation**: Repeated open/close overhead
- **Memory Leaks**: If close() not called for every open()

### NetCDF Library Limitations

The NetCDF-C library has specific behaviors:
- Opening a file returns a unique `ncid` (file identifier)
- Multiple `nc_open()` calls on same file return **different** `ncid` values
- With classic format: usually safe but inefficient
- With NetCDF-4/HDF5: can cause **data corruption** if writing
- With parallel NetCDF: **undefined behavior** with multiple opens

## Solution: File Descriptor Pool

The `fdpool_nc` singleton maintains a cache of open file descriptors:

```
┌─────────────────────────────────────────┐
│         fdpool_nc (Singleton)           │
├─────────────────────────────────────────┤
│  pool: map<filename, ncid>              │
│                                         │
│  "/path/data1.nc"  →  ncid: 1001       │
│  "/path/data2.nc"  →  ncid: 1002       │
│  "/path/data3.nc"  →  ncid: 1003       │
└─────────────────────────────────────────┘
         ↑                      ↑
    open(filename)         close_all()
    returns cached         closes all files
    ncid if exists
```

## Implementation

### Class Structure

```cpp
struct fdpool_nc {
  // Singleton pattern
  static fdpool_nc& get_instance();

  // Open file (or return cached ncid)
  int open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);

  // Close all cached files
  void close_all();

private:
  fdpool_nc() {}  // Private constructor
  std::map<std::string, int> pool;  // filename -> ncid cache
};
```

### Key Features

1. **Singleton Pattern**
   - One global pool per process
   - Accessible via `get_instance()`
   - Cannot be copied or assigned

2. **Lazy Opening**
   - File opened on first `open()` call
   - Cached `ncid` returned on subsequent calls
   - No need to track open/close manually

3. **Parallel NetCDF Support**
   - Tries `nc_open_par()` if available
   - Falls back to `nc_open()` if parallel fails
   - Uses provided `MPI_Comm` for collective access

4. **Automatic Cleanup**
   - `close_all()` closes all files in pool
   - Called by `ndarray_finalize()`
   - Idempotent (safe to call multiple times)

## Usage

### Basic Usage

```cpp
#include <ndarray/fdpool.hh>

// Get singleton instance
auto& pool = fdpool_nc::get_instance();

// Open file (or get cached ncid)
int ncid = pool.open("data.nc");

// Use ncid with NetCDF API
int varid;
nc_inq_varid(ncid, "temperature", &varid);
nc_get_var_float(ncid, varid, data);

// Don't call nc_close(ncid) - pool manages lifecycle

// At program exit:
pool.close_all();  // Or call ndarray_finalize()
```

### Integration with ndarray_group_stream

The stream functionality uses fdpool internally:

```cpp
// ndarray_group_stream.hh implementation:
std::shared_ptr<ndarray_group> read_netcdf(
  const std::string& f,
  const std::vector<variable_t>& variables,
  MPI_Comm comm)
{
  // Get the file descriptor pool
  auto &pool = fdpool_nc::get_instance();

  // Open file (or get cached ncid) - AVOIDS DOUBLE-OPENING
  int ncid = pool.open(f, comm);

  // Read multiple variables using same ncid
  for (const auto &var : variables) {
    int varid;
    nc_inq_varid(ncid, var.nc_name.c_str(), &varid);
    // ... read variable ...
  }

  // Don't close ncid - pool manages it
  return group;
}
```

### Cleanup at Program Exit

**Always call `ndarray_finalize()` before program exit:**

```cpp
int main(int argc, char** argv) {
  #if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
  #endif

  // Application code using streams/NetCDF
  ftk::stream s;
  s.parse_yaml("config.yaml");
  auto data = s.read(0);
  // ... process data ...

  // IMPORTANT: Cleanup before exit
  ftk::ndarray_finalize();  // Closes all files in fdpool

  #if NDARRAY_HAVE_MPI
  MPI_Finalize();
  #endif

  return 0;
}
```

### What Happens if You Forget?

**Without `ndarray_finalize()`:**
- File descriptors remain open
- May leak file descriptors (exhaust `ulimit -n`)
- May prevent file deletion/moving
- NetCDF files may be left in inconsistent state
- Valgrind/ASan may report leaks

**With `ndarray_finalize()`:**
- All files properly closed
- Resources released
- Clean shutdown

## Thread Safety

**WARNING**: `fdpool_nc` is **NOT thread-safe**.

### Single-Threaded: ✅ Safe
```cpp
int main() {
  auto& pool = fdpool_nc::get_instance();
  int ncid = pool.open("data.nc");
  // ... use ncid ...
  pool.close_all();
}
```

### Multi-Threaded: ⚠️ Unsafe
```cpp
// BAD: Race condition!
#pragma omp parallel for
for (int i = 0; i < 100; i++) {
  auto& pool = fdpool_nc::get_instance();  // Multiple threads
  int ncid = pool.open("data.nc");         // Race on pool.find()
  // ... undefined behavior ...
}
```

### Multi-Threaded Solution: 🔒 Add Mutex
```cpp
// GOOD: Protect with mutex
#include <mutex>

std::mutex pool_mutex;

#pragma omp parallel for
for (int i = 0; i < 100; i++) {
  int ncid;
  {
    std::lock_guard<std::mutex> lock(pool_mutex);
    auto& pool = fdpool_nc::get_instance();
    ncid = pool.open("data.nc");
  }
  // Use ncid outside critical section
  // ... read data ...
}
```

**Better**: Open files in serial section, use ncid in parallel:
```cpp
// Open once before parallel region
auto& pool = fdpool_nc::get_instance();
int ncid = pool.open("data.nc");

// Use ncid in parallel (reading is thread-safe in NetCDF-4)
#pragma omp parallel for
for (int i = 0; i < nvars; i++) {
  // Each thread reads different variable
  nc_get_var_float(ncid, varids[i], &data[i][0]);
}
```

## MPI Behavior

### Per-Process Pool

Each MPI rank has its **own** fdpool instance:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Rank 0    │  │   Rank 1    │  │   Rank 2    │
│             │  │             │  │             │
│  fdpool_nc  │  │  fdpool_nc  │  │  fdpool_nc  │
│  pool: {...}│  │  pool: {...}│  │  pool: {...}│
└─────────────┘  └─────────────┘  └─────────────┘
```

### Parallel NetCDF

With `nc_open_par()`, all ranks open the same file collectively:

```cpp
// All ranks call this together
auto& pool = fdpool_nc::get_instance();
int ncid = pool.open("data.nc", MPI_COMM_WORLD);

// If NC_HAS_PARALLEL defined:
//   Uses nc_open_par() - collective open
//   All ranks get same "logical" file but different ncid values
//
// Otherwise:
//   Each rank calls nc_open() independently
```

### Cleanup in MPI

```cpp
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // ... MPI work ...

  // Each rank closes its own pool
  ftk::ndarray_finalize();  // Call on ALL ranks

  MPI_Finalize();
  return 0;
}
```

## Performance Impact

### Benchmark: Multiple Variable Reads

**Scenario**: Read 100 variables from same NetCDF file

**Without fdpool**:
```
Open file:     100 times × 50ms  = 5000ms
Close file:    100 times × 30ms  = 3000ms
Total:         8000ms
```

**With fdpool**:
```
Open file:     1 time × 50ms     = 50ms
Close file:    1 time × 30ms     = 30ms
Total:         80ms
```

**Speedup**: 100×

### Memory Overhead

**Per cached file**:
- `std::string` filename: ~24-32 bytes + filename length
- `int` ncid: 4 bytes
- `std::map` overhead: ~32 bytes per entry

**Total**: ~60 bytes + filename length per file

For 1000 files: ~60 KB (negligible)

## Design Decisions

### Why Singleton?

**Pros**:
- Global cache accessible anywhere
- No need to pass pool around
- Ensures only one pool per process

**Cons**:
- Global state (harder to test)
- Not thread-safe
- Can't have multiple independent pools

**Alternative** (not used): Dependency injection
```cpp
// Could pass pool explicitly (more testable but verbose)
auto group = stream.read(0, my_custom_pool);
```

### Why Not RAII?

**Current**: Manual `close_all()` call

**Alternative**: RAII with destructor
```cpp
struct fdpool_nc {
  ~fdpool_nc() { close_all(); }  // Automatic cleanup
};
```

**Problem**: Destructor order with static variables is undefined. If fdpool destructor runs before other code finishes, crashes can occur.

**Solution**: Explicit `ndarray_finalize()` call gives user control.

### Why Map Instead of Unordered_Map?

**Current**: `std::map<std::string, int>`

**Why**:
- Filename lookup not performance-critical (only on first open)
- Deterministic iteration order (helpful for debugging)
- Slightly smaller memory overhead

**Could use**: `std::unordered_map` for O(1) lookup (vs O(log n))
- Benefit negligible for typical file counts (<100)

## Limitations

### 1. Read-Only Files

Currently only opens with `NC_NOWRITE` flag:
```cpp
nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
```

**Limitation**: Can't use fdpool for writing files.

**Workaround**: Write operations don't need pooling (usually one write per file).

### 2. No Per-Thread Pools

Single global pool per process.

**Limitation**: Thread safety requires external synchronization.

**Future**: Could use `thread_local` for per-thread pools.

### 3. No Eviction Policy

Pool grows indefinitely until `close_all()`.

**Limitation**: Memory grows with number of unique files accessed.

**Future**: Could add LRU eviction when pool size exceeds threshold.

### 4. Filename Comparison

Uses string comparison for cache lookup:
```cpp
pool.find(filename)  // String equality
```

**Limitation**:
- `"./data.nc"` vs `"data.nc"` treated as different files
- Symbolic links not resolved
- Relative vs absolute paths matter

**Future**: Could canonicalize paths before caching.

## Best Practices

### ✅ DO

1. **Always call `ndarray_finalize()` before exit**
   ```cpp
   int main() {
     // ... work ...
     ftk::ndarray_finalize();
     return 0;
   }
   ```

2. **Let fdpool manage lifecycle**
   ```cpp
   auto& pool = fdpool_nc::get_instance();
   int ncid = pool.open("data.nc");
   // Use ncid...
   // DON'T call nc_close(ncid) manually
   ```

3. **Use consistent file paths**
   ```cpp
   // GOOD: Same path string
   pool.open("data.nc");
   pool.open("data.nc");  // Reuses cached ncid

   // BAD: Different path strings for same file
   pool.open("data.nc");
   pool.open("./data.nc");  // Opens file twice!
   ```

### ❌ DON'T

1. **Don't close ncid manually**
   ```cpp
   int ncid = pool.open("data.nc");
   nc_close(ncid);  // DON'T! Pool still thinks it's open
   ```

2. **Don't mix fdpool and direct nc_open**
   ```cpp
   int ncid1 = pool.open("data.nc");      // In pool
   int ncid2;
   nc_open("data.nc", NC_NOWRITE, &ncid2);  // Opens again!
   ```

3. **Don't forget to finalize**
   ```cpp
   int main() {
     // ... use streams ...
     return 0;  // BAD: Files still open!
   }
   ```

## Debugging

### Enable Debug Output

Uncomment debug lines in fdpool.hh:
```cpp
// In open():
fprintf(stderr, "[fdpool] opened netcdf file %s, ncid=%d.\n", f.c_str(), ncid);

// In close_all():
fprintf(stderr, "[fdpool] closing netcdf file %s, ncid=%d.\n", kv.first.c_str(), kv.second);
```

### Check Open Files

**Linux**:
```bash
# See what files your program has open
lsof -p $(pgrep your_program)

# Count open file descriptors
ls -l /proc/$(pgrep your_program)/fd | wc -l
```

**macOS**:
```bash
lsof -p $(pgrep your_program)
```

### Valgrind

Check for leaks:
```bash
valgrind --leak-check=full ./your_program
```

With fdpool properly finalized, you should see:
```
All heap blocks were freed -- no leaks are possible
```

## Related Files

- **`include/ndarray/fdpool.hh`** - Implementation
- **`include/ndarray/util.hh`** - `ndarray_finalize()` function
- **`include/ndarray/ndarray_group_stream.hh`** - Usage in streams
- **`tests/test_ndarray_stream.cpp`** - Example usage

## See Also

- [NetCDF-C API Documentation](https://docs.unidata.ucar.edu/netcdf-c/current/)
- [Parallel NetCDF Guide](https://docs.unidata.ucar.edu/netcdf-c/current/parallel_io.html)
- [YAML Stream Documentation](./STREAM.md) - Stream usage patterns
