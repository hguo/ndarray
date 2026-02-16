# Unified ndarray Implementation Progress

## Completed (Phase 1 - Basic Infrastructure)

### 1. Added MPI/Distribution Support to ndarray.hh

**Includes:**
- Added `#include <ndarray/lattice_partitioner.hh>`

**New Public Methods:**
```cpp
// Configuration
void decompose(MPI_Comm comm, const std::vector<size_t>& global_dims, ...);
void set_replicated(MPI_Comm comm);

// Query methods
bool is_distributed() const;
bool is_replicated() const;
bool has_mpi_config() const;

// Distribution accessors (throw if not distributed)
const lattice& global_lattice() const;
const lattice& local_core() const;
const lattice& local_extent() const;

// Index conversion
std::vector<size_t> global_to_local(const std::vector<size_t>& global_idx) const;
std::vector<size_t> local_to_global(const std::vector<size_t>& local_idx) const;
bool is_local(const std::vector<size_t>& global_idx) const;

// MPI accessors
MPI_Comm comm() const;
int rank() const;
int nprocs() const;

// Ghost exchange (placeholder)
void exchange_ghosts();
```

**New Private Members:**
```cpp
enum class DistType { DISTRIBUTED, REPLICATED };

struct distribution_info {
  DistType type;
  MPI_Comm comm;
  int rank;
  int nprocs;

  // For DISTRIBUTED arrays:
  lattice global_lattice_;
  lattice local_core_;
  lattice local_extent_;
  std::unique_ptr<lattice_partitioner> partitioner_;

  // Ghost exchange topology
  std::vector<int> neighbor_ranks_;
  std::vector<lattice> send_regions_;
  std::vector<lattice> recv_regions_;
};

std::unique_ptr<distribution_info> dist_;  // nullptr = serial
```

**Implementations Added:**
- ✅ `decompose()` - Creates distribution info, partitioner, reshapes to local extent
- ✅ `set_replicated()` - Marks array as replicated across all ranks
- ✅ `is_distributed()`, `is_replicated()`, `has_mpi_config()` - Query methods
- ✅ `global_lattice()`, `local_core()`, `local_extent()` - Accessors
- ✅ `comm()`, `rank()`, `nprocs()` - MPI accessors
- ✅ `global_to_local()`, `local_to_global()`, `is_local()` - Index conversion
- ⚠️  `setup_ghost_exchange()` - Basic neighbor identification (simplified)
- ⚠️  `exchange_ghosts()` - Placeholder (TODO: copy from distributed_ndarray.hh)

### 2. Compilation Status

✅ **Library compiles successfully** with new additions

## Next Steps (Phase 2 - Complete Ghost Exchange)

### 1. Add Neighbor Struct to distribution_info

Need to add:
```cpp
struct Neighbor {
  int rank;              // Neighbor's MPI rank
  int direction;         // Which face: 0=left, 1=right (dim 0); 2=down, 3=up (dim 1)
  size_t send_count;     // Number of elements to send
  size_t recv_count;     // Number of elements to receive
};

std::vector<Neighbor> neighbors_;
bool neighbors_identified_ = false;
```

### 2. Implement Full Ghost Exchange

Copy and adapt from `distributed_ndarray.hh`:
- `exchange_ghosts()` - MPI communication with pack/unpack
- `pack_boundary_data()` - Extract boundary data into buffer
- `unpack_ghost_data()` - Write received data into ghost regions

**Changes needed:**
- Replace `at()` with `f()` (at() is deprecated)
- Adapt to work with unified ndarray storage
- Handle multi-dimensional cases properly

### 3. Update I/O Methods (Phase 3)

Modify these methods in ndarray.hh to check distribution mode:
- `read_netcdf()`, `write_netcdf()`
- `read_hdf5()`, `write_hdf5()`
- `read_pnetcdf()`, `write_pnetcdf()`
- `read_adios2()`, `write_adios2()`
- Binary I/O methods

Logic:
```cpp
if (should_use_parallel_io()) {
  // Use parallel I/O paths (copy from distributed_ndarray)
} else if (should_use_replicated_io()) {
  // Rank 0 reads/writes + MPI_Bcast
} else {
  // Serial I/O (existing code)
}
```

### 4. Update Stream Class (Phase 4)

- Parse YAML variable configurations (distributed/replicated types)
- Configure ndarrays with decompose() or set_replicated() based on YAML
- Default behavior: replicated (safe for all cases)
- Handle nprocs == 1 case (no distribution)

### 5. Migration and Cleanup (Phase 5)

- Remove `distributed_ndarray.hh` (no longer needed)
- Remove `distributed_ndarray_group.hh` (use regular ndarray_group)
- Remove `distributed_ndarray_stream.hh` (merge into stream)
- Update tests to use unified API
- Update examples to use unified API

## Testing Plan

### Unit Tests
1. **Serial mode**: Ensure existing tests still pass
2. **Distributed mode**: Test decompose() with different patterns
3. **Replicated mode**: Test set_replicated()
4. **Index conversion**: Test global_to_local/local_to_global
5. **Ghost exchange**: Test with known values
6. **I/O**: Test parallel/replicated/serial I/O

### Integration Tests
1. Run existing MPI tests with unified API
2. Test stream with YAML configs (distributed/replicated)
3. Test mixed distributed+replicated variables
4. Test with 1, 2, 4, 8 ranks

## Design Decisions Made

✅ **Default behavior**: Replicated (user requested)
✅ **No backward compatibility**: distributed_ndarray will be removed (user confirmed no existing users)
✅ **Unified API**: Same code works for serial and parallel
✅ **Optional distribution**: Arrays are serial by default, distribution is opt-in
✅ **Runtime configuration**: Distribution is configured at runtime, not compile-time

## Benefits

1. **Simplicity**: One class instead of two
2. **Flexibility**: Same binary runs serial or parallel
3. **Safety**: Default (replicated) works for all cases
4. **Clarity**: Explicit opt-in for distribution
5. **Maintainability**: Less code duplication

## Timeline Estimate

- Phase 1 (Basic infrastructure): ✅ **DONE**
- Phase 2 (Ghost exchange): ~1-2 days
- Phase 3 (I/O methods): ~2-3 days
- Phase 4 (Stream integration): ~1-2 days
- Phase 5 (Migration/cleanup): ~1 day

**Total**: ~1 week for complete implementation

## Current Status

**Phase 1 complete** - Basic infrastructure compiles and is ready for testing.
Ready to proceed with Phase 2 (ghost exchange implementation).
