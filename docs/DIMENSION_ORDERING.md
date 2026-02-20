# Dimension Ordering in ndarray and I/O Formats

## Summary

**Key Fact**: NetCDF, HDF5, and ADIOS2 **all use C-order** (row-major, last dimension varies fastest). ndarray uses **Fortran-order** (column-major, first dimension varies fastest) internally, so all three require the same dimension reversal at API boundaries.

## Memory Layout Comparison

### ndarray (Fortran-order)
```cpp
arr.reshapef({3, 4, 5});  // Shape: [3, 4, 5]
// Memory layout: first dimension (3) varies FASTEST
// Linear index order: [0,0,0], [1,0,0], [2,0,0], [0,1,0], [1,1,0], ..., [2,3,4]
```

### NetCDF/HDF5/ADIOS2 (C-order)
```cpp
// Variable with dimensions [3, 4, 5]
// Memory layout: last dimension (5) varies FASTEST
// Linear index order: [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], ..., [2,3,4]
```

## Why Reversal is Needed

When ndarray reads from or writes to these formats, dimensions must be reversed:

```cpp
// Reading from NetCDF/HDF5/ADIOS2:
std::vector<size_t> external_dims = {100, 200, 50};  // C-order from file
auto ndarray_dims = c_to_f_order(external_dims);     // → {50, 200, 100} Fortran-order
arr.reshapef(ndarray_dims);

// Writing to NetCDF/HDF5/ADIOS2:
auto external_dims = arr.shapec();  // Get C-order from ndarray's Fortran-order
// Pass external_dims to API
```

## Format-Specific Details

### NetCDF

**API Convention**:
```c
nc_def_dim(ncid, "x", 100, &dimids[0]);
nc_def_dim(ncid, "y", 200, &dimids[1]);
nc_def_dim(ncid, "z", 50, &dimids[2]);
nc_def_var(ncid, "data", NC_FLOAT, 3, dimids, &varid);
// Creates variable [100, 200, 50] where 50 varies fastest (C-order)
```

**Memory Layout**: Last dimension (z=50) varies fastest in memory

**ndarray Conversion**:
```cpp
// To match NetCDF [100, 200, 50] C-order memory:
arr.reshapef({50, 200, 100});  // First dim (50) varies fastest (Fortran-order)
// Now arr's memory layout matches NetCDF's file layout
```

### HDF5

**API Convention**:
```c
hsize_t dims[3] = {100, 200, 50};
hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
// Creates dataspace [100, 200, 50] where 50 varies fastest (C-order)
```

**Memory Layout**: Identical to NetCDF - last dimension varies fastest

**ndarray Conversion**: Same as NetCDF

### ADIOS2

**API Convention**:
```cpp
adios2::Variable<float> var = io.DefineVariable<float>(
    "data",
    {100, 200, 50},  // shape in C-order
    {0, 0, 0},       // start
    {100, 200, 50}   // count
);
// Creates variable [100, 200, 50] where 50 varies fastest (C-order)
```

**Memory Layout**: Identical to NetCDF/HDF5 - last dimension varies fastest

**ndarray Conversion**: Same as NetCDF/HDF5

## Distributed I/O Considerations

When doing parallel I/O with MPI domain decomposition:

```cpp
// Global array [1000, 800] in NetCDF (C-order)
// ndarray stores as [800, 1000] (Fortran-order)

// Rank 0 reads portion: NetCDF indices [0:499, 0:799]
size_t netcdf_starts[2] = {0, 0};
size_t netcdf_counts[2] = {500, 800};

// Convert to ndarray Fortran-order:
auto ndarray_starts = c_to_f_order(netcdf_starts, 2);  // → {0, 0}
auto ndarray_counts = c_to_f_order(netcdf_counts, 2);  // → {800, 500}

// But when calling NetCDF API, reverse back:
auto nc_starts = c_to_f_order(ndarray_starts, 2);  // → {0, 0}
auto nc_counts = c_to_f_order(ndarray_counts, 2);  // → {500, 800}
nc_get_vara_float(ncid, varid, nc_starts.data(), nc_counts.data(), data);
```

## Helper Functions

ndarray provides helper functions to avoid manual reversals:

```cpp
// Convert C-order to Fortran-order (for reading)
auto f_dims = ndarray_base::c_to_f_order(c_dims);

// Convert Fortran-order to C-order (for writing)
auto c_dims = ndarray_base::f_to_c_order(f_dims);
// Or use member function:
auto c_dims = arr.shapec();  // Returns C-order dimensions
```

## Why Not Use C-Order Internally?

**Historical reasons**:
1. Fortran is traditional in scientific computing
2. Interop with Fortran libraries
3. NetCDF was designed for Fortran originally (despite C-order memory!)

**Modern perspective**:
- NumPy uses C-order by default
- Most C/C++ code expects row-major
- But ndarray maintains Fortran-order for backward compatibility

## Best Practices

1. **Never manually reverse dimensions** - use `c_to_f_order()` and `f_to_c_order()` or `shapec()`
2. **Add comments at API boundaries** explaining conversion
3. **Be consistent** - all I/O backends follow the same pattern
4. **Trust the helpers** - they handle the conversion correctly

## References

- [NetCDF C API Guide](https://www.unidata.ucar.edu/software/netcdf/docs/index.html)
- [HDF5 Memory Layout](https://portal.hdfgroup.org/display/HDF5/Memory+Layout)
- [ADIOS2 Documentation](https://adios2.readthedocs.io/)
- [NumPy ndarray.strides](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
