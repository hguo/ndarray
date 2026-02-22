# Phase 3: Distribution-Aware I/O with Auto-Detection

## Implemented Methods

### NetCDF I/O (with parallel support)
```cpp
void read_netcdf_auto(const std::string& filename, const std::string& varname);
void write_netcdf_auto(const std::string& filename, const std::string& varname);
```

**Behavior:**
- **Distributed**: Parallel read/write using NC_PARALLEL (each rank reads/writes local core)
- **Replicated**: Rank 0 reads/writes + MPI_Bcast to all ranks
- **Serial**: Regular serial NetCDF I/O

**Implementation Details:**
- Distributed: Uses hyperslab with start/count from local_core
- Replicated: Rank 0 does I/O, broadcasts full array
- Auto-detects parallel NetCDF capability at runtime

### HDF5 I/O (with parallel support)
```cpp
void read_hdf5_auto(const std::string& filename, const std::string& varname);
void write_hdf5_auto(const std::string& filename, const std::string& varname);
```

**Behavior:**
- **Distributed**: Parallel HDF5 with H5Pset_fapl_mpio + hyperslab selection
- **Replicated**: Rank 0 reads/writes + MPI_Bcast
- **Serial**: Regular serial HDF5 I/O

**Implementation Details:**
- Distributed: Collective I/O with H5FD_MPIO_COLLECTIVE
- Uses hyperslab selection for local core region
- Rank 0 creates file/dataset, all ranks write

### Binary I/O (with MPI-IO)
```cpp
void read_binary_auto(const std::string& filename);
void write_binary_auto(const std::string& filename);
```

**Behavior:**
- **Distributed**: MPI-IO with MPI_File_read/write_at_all
- **Replicated**: Rank 0 reads/writes + MPI_Bcast
- **Serial**: Regular binary file I/O

**Implementation Details:**
- Distributed: Column-major (Fortran) order for 2D arrays
- Calculates proper offset based on local_core position
- Collective operations for performance

## Usage Examples

### Example 1: Distributed NetCDF Read

```cpp
ftk::ndarray<float> temp;

// Configure for distributed execution
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});

// Read distributed - each rank gets local portion
temp.read_netcdf_auto("data.nc", "temperature");

// Exchange ghosts for stencil operations
temp.exchange_ghosts();
```

### Example 2: Replicated Mesh Read

```cpp
ftk::ndarray<double> mesh;

// Configure as replicated (all ranks get full data)
mesh.set_replicated(MPI_COMM_WORLD);

// Read replicated - rank 0 reads, broadcasts to all
mesh.read_netcdf_auto("mesh.nc", "coordinates");

// All ranks have full mesh data
```

### Example 3: Serial Mode (Backward Compatible)

```cpp
ftk::ndarray<float> data;

// No MPI configuration = serial mode
data.read_netcdf_auto("data.nc", "temperature");

// Behaves exactly like regular read_netcdf()
```

### Example 4: Mixed Distributed and Replicated

```cpp
// Large field: distributed
ftk::ndarray<float> velocity;
velocity.decompose(MPI_COMM_WORLD, {1000, 800, 600, 3}, 0, {4, 2, 1, 0}, {1, 1, 1, 0});
velocity.read_netcdf_auto("fields.nc", "velocity");
velocity.exchange_ghosts();

// Small global data: replicated
ftk::ndarray<double> mesh;
mesh.set_replicated(MPI_COMM_WORLD);
mesh.read_netcdf_auto("mesh.nc", "coordinates");

// Process using both: each rank has local velocity portion + full mesh
for (size_t i = 0; i < velocity.dim(0); i++) {
  auto global_idx = velocity.local_to_global({i, 0, 0, 0});
  double x = mesh(global_idx[0], 0);  // Access global mesh coordinate
  // ... process ...
}
```

## Implementation Notes

### 1. Auto-Detection Logic

Methods check distribution state and dispatch:

```cpp
if (should_use_parallel_io()) {
  // Distributed: parallel I/O with local core region
} else if (should_use_replicated_io()) {
  // Replicated: rank 0 I/O + broadcast
} else {
  // Serial: regular I/O
}
```

### 2. Parallel NetCDF

- Uses `nc_open_par()` for parallel file access
- Hyperslab defined by local_core start/count
- Each rank reads/writes its portion independently
- No communication during I/O

### 3. Parallel HDF5

- Uses `H5Pset_fapl_mpio()` for MPI-IO file access
- `H5Pset_dxpl_mpio()` with `H5FD_MPIO_COLLECTIVE` for collective I/O
- Hyperslab selection for local core region
- Rank 0 creates file/dataset structure

### 4. Binary MPI-IO

- Column-major (Fortran) order matching ndarray internal format
- `MPI_File_read_at_all` / `MPI_File_write_at_all` for collective I/O
- Offset calculation handles multi-dimensional arrays
- Special handling for 2D: column-by-column I/O

### 5. Replicated I/O Strategy

**Why Rank 0 + Broadcast?**
- Simple and works for all file formats
- No redundant I/O from multiple ranks
- Efficient for small-to-medium data (< 100 MB)

**Alternatives Considered:**
- All ranks read independently: Redundant I/O, filesystem contention
- Scatter/gather: More complex, not beneficial for replicated case

## Performance Considerations

### Distributed I/O
- **Scalability**: Linear with number of ranks (each reads 1/N data)
- **Best for**: Large arrays (> 1 GB), many ranks (> 4)
- **Limitation**: Filesystem parallel I/O capability

### Replicated I/O
- **Scalability**: Serial read + O(log N) broadcast
- **Best for**: Small arrays (< 100 MB), needed by all ranks
- **Limitation**: All ranks store full data (N × memory)

### Binary I/O
- **Fastest**: MPI-IO is typically fastest for parallel filesystems
- **Portable**: Works with any MPI implementation
- **Limitation**: No self-describing format (must know dimensions)

## Testing

### Unit Tests Needed

1. **Distributed NetCDF**:
   - Create global array, write distributed, read back
   - Verify each rank gets correct portion
   - Test with 1, 2, 4 ranks

2. **Replicated NetCDF**:
   - Write on rank 0, read replicated
   - Verify all ranks have identical data

3. **Binary MPI-IO**:
   - Test 1D, 2D, 3D arrays
   - Verify column-major ordering
   - Test with different decompositions

4. **HDF5 Parallel**:
   - Test collective read/write
   - Verify hyperslab selection

5. **Mixed Mode**:
   - Distributed + replicated in same application
   - Verify no interference

### Integration Tests

Run existing MPI tests with new `_auto` methods:
- test_distributed_ndarray: Use read_binary_auto
- test_pnetcdf: Compare with read_netcdf_auto
- test_ghost_exchange: Use read_binary_auto + exchange_ghosts

## Migration Path

### Old API (distributed_ndarray)
```cpp
ftk::distributed_ndarray<float> temp(MPI_COMM_WORLD);
temp.decompose({1000, 800}, 0, {}, {1, 1});
temp.read_parallel("data.nc", "temperature", 0);
```

### New API (unified ndarray)
```cpp
ftk::ndarray<float> temp;
temp.decompose(MPI_COMM_WORLD, {1000, 800}, 0, {}, {1, 1});
temp.read_netcdf_auto("data.nc", "temperature");
```

**Note**: Eventually, we can rename `read_netcdf_auto` → `read_netcdf` after removing old base class methods or making them dispatch to auto-detection.

## Future Enhancements

1. **Automatic method selection**: Make `read_netcdf()` automatically use parallel I/O
2. **ADIOS2 support**: Add `read_adios2_auto()` with BP format
3. **PNetCDF integration**: Add `read_pnetcdf_auto()` wrapper
4. **Async I/O**: Non-blocking reads for overlapping computation
5. **I/O hints**: MPI_Info hints for filesystem optimization
6. **Compression**: Transparent compression/decompression
7. **Caching**: Cache replicated data to avoid repeated broadcasts

## Benefits

✅ **Automatic**: No manual parallel I/O code required
✅ **Flexible**: Same code for distributed, replicated, serial
✅ **Efficient**: Uses optimal I/O strategy for each case
✅ **Transparent**: Works with existing file formats
✅ **Safe**: Backward compatible with serial code

## Status

✅ **Phase 3 Complete**:
- read/write_netcdf_auto implemented
- read/write_hdf5_auto implemented
- read/write_binary_auto implemented
- All methods compile successfully
- Auto-detection logic working

**Next**: Phase 4 - Stream integration with YAML configuration
