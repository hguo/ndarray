# Dimension Ordering Unification - Summary

## Overview

Unified dimension ordering across all I/O backends (NetCDF, HDF5, ADIOS2, Binary) by:
1. Adding clean helper functions (`c_to_f_order`, `f_to_c_order`)
2. Replacing all ad-hoc `std::reverse()` calls with these helpers
3. Establishing clear principle: **keep ndarray Fortran-order internally, convert only at API boundaries**

## Key Changes

### 1. Added Helper Functions (`include/ndarray/ndarray_base.hh`)

```cpp
// Convert C-order (external formats) → Fortran-order (ndarray internal)
static std::vector<size_t> c_to_f_order(const std::vector<size_t>& c_shape);
static std::vector<size_t> c_to_f_order(const size_t* c_shape, size_t ndims);

// Convert Fortran-order → C-order (alias of shapec() for clarity)
static std::vector<size_t> f_to_c_order(const std::vector<size_t>& f_shape);

// Existing member function (no changes needed)
const std::vector<size_t> shapec() const;  // Returns C-order dimensions
```

### 2. NetCDF (`include/ndarray/ndarray_base.hh`)

**Before** (ad-hoc reversals):
```cpp
std::vector<size_t> mysizes(sizes, sizes+ndims);
std::reverse(mysizes.begin(), mysizes.end());
reshapef(mysizes);

std::vector<size_t> nc_starts(starts, starts+ndims);
std::vector<size_t> nc_sizes(sizes, sizes+ndims);
std::reverse(nc_starts.begin(), nc_starts.end());
std::reverse(nc_sizes.begin(), nc_sizes.end());
```

**After** (clean helpers):
```cpp
auto ndarray_sizes = c_to_f_order(sizes, ndims);
reshapef(ndarray_sizes);

auto nc_starts = c_to_f_order(starts, ndims);
auto nc_sizes = c_to_f_order(sizes, ndims);
```

### 3. ADIOS2 (`include/ndarray/ndarray.hh`)

**Before** (double reshape - inefficient!):
```cpp
std::vector<size_t> shape(var.Shape());
reshapef(shape);  // First reshape
var.SetSelection({zeros, shape});
// ... read ...
std::reverse(shape.begin(), shape.end());  // Then reverse
reshapef(shape);  // Second reshape!
```

**After** (single reshape):
```cpp
std::vector<size_t> adios_shape(var.Shape());
auto ndarray_shape = c_to_f_order(adios_shape);
reshapef(ndarray_shape);  // Only one reshape

std::vector<size_t> zeros(adios_shape.size(), 0);
var.SetSelection({zeros, adios_shape});  // Use original C-order
```

**Performance improvement**: Eliminated redundant reshape operation

### 4. ADIOS1 Legacy (`include/ndarray/ndarray.hh`)

**Before**:
```cpp
std::reverse(mydims.begin(), mydims.end());
reshapef(mydims);
```

**After**:
```cpp
auto ndarray_dims = c_to_f_order(mydims);
reshapef(ndarray_dims);
```

### 5. HDF5 Read (`include/ndarray/ndarray.hh`)

**Before**:
```cpp
std::vector<size_t> dims(h5ndims);
for (auto i = 0; i < h5ndims; i++)
  dims[i] = h5dims[i];
std::reverse(dims.begin(), dims.end());
reshapef(dims);
```

**After**:
```cpp
std::vector<size_t> h5_dims(h5ndims);
for (auto i = 0; i < h5ndims; i++)
  h5_dims[i] = h5dims[i];

auto ndarray_dims = c_to_f_order(h5_dims);
reshapef(ndarray_dims);
```

### 6. HDF5 Write (`include/ndarray/ndarray.hh`)

**Before** (manual reversal loop):
```cpp
const size_t nd = this->nd();
std::vector<hsize_t> dims(nd);
for (size_t d = 0; d < nd; d++) {
  dims[nd - 1 - d] = this->dimf(d);  // Manual reversal
}
```

**After** (use shapec()):
```cpp
const size_t nd = this->nd();
auto h5_dims_vec = shapec();  // Get C-order dimensions
std::vector<hsize_t> h5_dims(h5_dims_vec.begin(), h5_dims_vec.end());
```

### 7. Binary Streams (`include/ndarray/ndarray_stream_binary.hh`)

**Before**:
```cpp
std::vector<size_t> gdims;
for (int d : var.dimensions) gdims.push_back(static_cast<size_t>(d));
std::reverse(gdims.begin(), gdims.end());
```

**After**:
```cpp
std::vector<size_t> yaml_dims_c_order;
for (int d : var.dimensions) yaml_dims_c_order.push_back(static_cast<size_t>(d));

auto ndarray_dims = ndarray_base::c_to_f_order(yaml_dims_c_order);
```

## Benefits

### 1. **Clarity**
- Intent is explicit: `c_to_f_order()` vs manual `std::reverse()`
- No more cryptic comments like "we use a different dimension ordering than..."
- Self-documenting code

### 2. **Maintainability**
- Single source of truth for dimension conversion
- Easy to modify conversion logic if needed
- New I/O backends know exactly what to do

### 3. **Correctness**
- Eliminates manual reversal errors
- Consistent pattern across all backends
- Fixed ADIOS2 double-reshape bug

### 4. **Performance**
- ADIOS2: Eliminated redundant reshape (2x → 1x)
- HDF5 write: Cleaner code with same performance
- No overhead from helper functions (compiler inlines)

## Testing

All existing tests pass:
- ✅ NetCDF I/O (serial & parallel)
- ✅ HDF5 I/O (serial)
- ✅ ADIOS2 I/O
- ✅ Binary stream I/O
- ✅ Distributed NetCDF (Test 6 now enabled)

## Documentation

Created comprehensive guides:
- `docs/DIMENSION_ORDERING.md` - Full explanation with examples
- Clear comments at all API boundaries
- Docstrings for helper functions

## Migration Guide (for contributors)

**When adding new I/O backend**:

1. **Reading** (external → ndarray):
```cpp
std::vector<size_t> external_dims = get_dims_from_api();
auto ndarray_dims = c_to_f_order(external_dims);
reshapef(ndarray_dims);

// Convert back for API selection/hyperslab:
auto api_starts = c_to_f_order(ndarray_starts);
auto api_counts = c_to_f_order(ndarray_counts);
api_read(api_starts.data(), api_counts.data(), data);
```

2. **Writing** (ndarray → external):
```cpp
auto external_dims = shapec();  // Get C-order from ndarray
api_create_dataset(external_dims.data());

// If needed for hyperslab:
auto api_starts = c_to_f_order(ndarray_starts);
auto api_counts = c_to_f_order(ndarray_counts);
```

**Rule of thumb**:
- ndarray works in Fortran-order internally
- External APIs work in C-order
- Convert at boundary using helpers, never manual `std::reverse()`

## Files Modified

```
include/ndarray/ndarray_base.hh          (helper functions + NetCDF)
include/ndarray/ndarray.hh               (ADIOS2, HDF5)
include/ndarray/ndarray_stream_binary.hh (Binary streams)
docs/DIMENSION_ORDERING.md               (new documentation)
```

## Commits

This work should be committed as:
```
git add include/ndarray/ndarray_base.hh \
        include/ndarray/ndarray.hh \
        include/ndarray/ndarray_stream_binary.hh \
        docs/DIMENSION_ORDERING.md \
        DIMENSION_ORDERING_FIXES_SUMMARY.md

git commit -m "Unify dimension ordering across all I/O backends

- Add c_to_f_order() and f_to_c_order() helper functions
- Replace all ad-hoc std::reverse() calls with clean helpers
- Fix ADIOS2 double-reshape bug (performance improvement)
- Simplify HDF5 write using shapec()
- Enhance binary stream dimension conversion clarity
- Add comprehensive DIMENSION_ORDERING.md documentation
- Establish clear principle: keep ndarray Fortran-order, convert at API boundaries

All I/O tests pass. No breaking changes.
"
```

---

**Principle**: Keep ndarray's internal dimension order (Fortran) throughout. Convert only at API boundaries using explicit helper functions. Never use ad-hoc `std::reverse()` calls.
