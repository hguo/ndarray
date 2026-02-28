# ndarray Documentation Index

**Version**: 0.0.6
**Last Updated**: 2026-02-28

---

## Getting Started

- **[Getting Started Guide](GETTING_STARTED.md)** ⭐ **START HERE** - Complete tutorial for new users
- [Installation Guide](#installation) - Build from source, dependencies
- [Quick Examples](#quick-examples) - Copy-paste code snippets

---

## Core Concepts

### Array Basics
- **[Array Indexing](ARRAY_INDEXING.md)** - Fortran vs C order, multi-dimensional indexing
- **[Dimension Ordering](DIMENSION_ORDERING.md)** - Understanding Fortran/C conventions
- **[Multi-component Arrays](MULTICOMPONENT_ARRAYS.md)** - Vector and tensor fields

### Storage
- **[Storage Backends](STORAGE_BACKENDS.md)** - Native, xtensor, Eigen storage policies
- **[Backend Design](BACKEND_DESIGN.md)** - Policy-based storage architecture

---

## I/O Formats

### File Formats
- **[HDF5](PARALLEL_HDF5.md)** - HDF5 I/O with parallel support
- **[NetCDF](IO_BACKEND_AGNOSTIC.md)** - NetCDF and PNetCDF I/O
- **[ADIOS2](ADIOS2_TESTS.md)** - High-performance parallel I/O
- **[VTK](VTK_TESTS.md)** - Visualization data formats
- **[PNG](PNG_SUPPORT.md)** - Image I/O

### Time Series
- **[Streams](OPTIONAL_YAML.md)** - YAML-based stream configuration
- **[Time Dimension](TIME_DIMENSION.md)** - Time-varying data handling

---

## Parallel Computing

### MPI Parallelism
- **[Distributed Arrays](DISTRIBUTED_NDARRAY.md)** - Domain decomposition, ghost cells
- **[Distributed Indexing](DISTRIBUTED_INDEXING_CLARIFICATION.md)** - Global/local index conversion
- **[Multi-component Distributed Arrays](MULTICOMPONENT_ARRAYS_DISTRIBUTED.md)**

### Parallel I/O
- **[Parallel HDF5](PARALLEL_HDF5.md)** - MPI-parallel HDF5 I/O
- **[PNetCDF](PNETCDF_TESTS.md)** - Parallel NetCDF I/O

---

## GPU Acceleration

- **[GPU Support](GPU_SUPPORT.md)** - CUDA, HIP, SYCL for data movement
- **[GPU-Aware MPI Plan](GPU_AWARE_MPI_PLAN.md)** - Direct GPU-to-GPU transfers
- **[GPU-Aware MPI Summary](GPU_AWARE_MPI_SUMMARY.md)**

---

## Advanced Topics

### Architecture
- **[Unified Design](UNIFIED_NDARRAY_DESIGN.md)** - Overall architecture
- **[Error Handling](ERROR_HANDLING.md)** - Exception-based error handling
- **[Exception Handling](EXCEPTION_HANDLING.md)** - Best practices

### Performance
- **[Zero-Copy Optimization](ZERO_COPY_OPTIMIZATION.md)** - Reference accessors
- **[Fortran/C Conventions](FORTRAN_C_CONVENTIONS.md)** - Performance implications

---

## Build & CI

- **[CMake Simplification](CMAKE_SIMPLIFICATION.md)** - Build system overview
- **[CI Fixes](CI_FIXES.md)** - Continuous integration setup

---

## Development & Internal

See [progress/](progress/) directory for:
- Implementation progress tracking
- Design decisions and rationale
- Critical analysis and improvements

Key documents:
- [Progress Summary](progress/IMPROVEMENTS_SUMMARY_2026-02-20.md)
- [Critical Analysis](progress/CRITICAL_ANALYSIS.md)
- [Error Handling Phases](progress/ERROR_HANDLING_PHASES_1-4_COMPLETE.md)
- [GPU RAII Improvements](progress/GPU_RAII_IMPROVEMENTS.md)

---

## Troubleshooting

### Common Issues

**Compilation Errors**:
- Ensure C++17: `-std=c++17`
- Check include path: `-I /path/to/ndarray/include`

**Linking Errors**:
- Link libraries in order: `-lndarray -lhdf5 -lnetcdf`
- For MPI: use `mpic++` instead of `g++`

**Runtime Errors**:
- Check dimension ordering (Fortran vs C)
- Verify MPI collective operations (all ranks participate)
- Enable exceptions for better error messages

**Performance Issues**:
- Use direct pointer access (`.data()`) for tight loops
- Consider storage backends (xtensor for SIMD)
- Profile I/O vs computation time

---

## Reference

### API Documentation
See header files in `include/ndarray/`:
- `ndarray.hh` - Core array class
- `ndarray_base.hh` - Base class and I/O methods
- `ndarray_group_stream.hh` - Stream processing
- `error.hh` - Exception types

### Test Examples
See `tests/` directory for working examples:
- `test_ndarray.cpp` - Basic operations
- `test_distributed_ndarray.cpp` - MPI parallelism
- `test_hdf5_auto.cpp` - Parallel HDF5
- `test_adios2_stream.cpp` - ADIOS2 time series
- `test_storage_backends.cpp` - Storage policies

---

## External Resources

- **GitHub**: https://github.com/hguo/ndarray
- **Issues**: https://github.com/hguo/ndarray/issues
- **License**: MIT

---

## Quick Reference Card

### Creating Arrays
```cpp
ftk::ndarray<float> arr;
arr.reshapef(100, 200);       // 2D Fortran order
arr.reshapef(100, 200, 50);   // 3D
```

### Accessing Elements
```cpp
arr.f(i, j);      // Fortran order (i varies fastest)
arr.c(i, j);      // C order (j varies fastest)
arr[index];       // Linear index
arr.data()[index]; // Direct pointer access
```

### File I/O
```cpp
arr.to_h5("file.h5", "dataset");
arr.read_h5("file.h5", "dataset");

arr.to_nc("file.nc", "variable");
arr.read_nc("file.nc", "variable");

arr.to_bp("file.bp", "variable");
auto loaded = ftk::ndarray<float>::from_bp("file.bp", "variable", 0);
```

### Distributed (MPI)
```cpp
arr.decompose(MPI_COMM_WORLD, {1000, 2000});
arr.start_exchange_ghosts();
arr.finish_exchange_ghosts();
arr.write_hdf5_auto("parallel.h5", "data");
```

---

**Need help?** Start with [Getting Started](GETTING_STARTED.md) or check specific topics above.
