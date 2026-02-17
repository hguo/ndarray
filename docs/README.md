# Documentation Index

This directory contains documentation for the ndarray library. Documentation is organized into user guides, developer notes, and planning/historical documents.

---

## User Guides

**Start here if you're using the library:**

### Core Concepts
- **[MAINTENANCE-MODE.md](MAINTENANCE-MODE.md)** - Library status, features, and roadmap
- **[STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)** - Storage policy system (native, xtensor, Eigen)
- **[EXCEPTION_HANDLING.md](EXCEPTION_HANDLING.md)** - Error handling and exceptions

### Array Operations
- **[ARRAY_ACCESS.md](ARRAY_ACCESS.md)** - Accessing array elements (row-major vs column-major)
- **[ARRAY_INDEXING.md](ARRAY_INDEXING.md)** - Indexing conventions and patterns
- **[FORTRAN_C_CONVENTIONS.md](FORTRAN_C_CONVENTIONS.md)** - Row-major (C) vs column-major (Fortran)
- **[ZERO_COPY_OPTIMIZATION.md](ZERO_COPY_OPTIMIZATION.md)** - Using `get_ref()` for zero-copy access
- **[VECTOR_CONVERSION.md](VECTOR_CONVERSION.md)** - Converting between ndarrays and std::vector

### Distributed Computing
- **[DISTRIBUTED_NDARRAY.md](DISTRIBUTED_NDARRAY.md)** - MPI domain decomposition and ghost cells ⭐
- **[DISTRIBUTED_INDEXING_CLARIFICATION.md](DISTRIBUTED_INDEXING_CLARIFICATION.md)** - Global/local index conversion
- **[MULTICOMPONENT_ARRAYS_DISTRIBUTED.md](MULTICOMPONENT_ARRAYS_DISTRIBUTED.md)** - Vector/tensor fields in distributed memory

### GPU Support
- **[GPU_SUPPORT.md](GPU_SUPPORT.md)** - CUDA/HIP GPU acceleration ⭐
- **[GPU_AWARE_MPI_SUMMARY.md](GPU_AWARE_MPI_SUMMARY.md)** - GPU-direct MPI (experimental)

### I/O Formats
- **[HDF5_TIMESTEPS_PER_FILE.md](HDF5_TIMESTEPS_PER_FILE.md)** - Time-series data in HDF5
- **[PNG_SUPPORT.md](PNG_SUPPORT.md)** - Reading/writing PNG images
- **[OPTIONAL_YAML.md](OPTIONAL_YAML.md)** - YAML stream configuration

### Advanced Topics
- **[MULTICOMPONENT_ARRAYS.md](MULTICOMPONENT_ARRAYS.md)** - Vector/tensor fields
- **[TIME_DIMENSION.md](TIME_DIMENSION.md)** - Time-varying data conventions
- **[VARIABLE_NAMING_BEST_PRACTICES.md](VARIABLE_NAMING_BEST_PRACTICES.md)** - Format-specific variable names

---

## Developer Notes

**Internal documentation for contributors:**

- **[BACKEND_DESIGN.md](BACKEND_DESIGN.md)** - Storage backend architecture
- **[BACKENDS.md](BACKENDS.md)** - I/O backend system design
- **[IO_BACKEND_AGNOSTIC.md](IO_BACKEND_AGNOSTIC.md)** - Format-agnostic I/O design
- **[ERROR_HANDLING.md](ERROR_HANDLING.md)** - Error handling implementation
- **[CI_FIXES.md](CI_FIXES.md)** - CI/CD troubleshooting notes
- **[CMAKE_SIMPLIFICATION.md](CMAKE_SIMPLIFICATION.md)** - Build system documentation
- **[VARIABLE_NAMING_PROBLEMS.md](VARIABLE_NAMING_PROBLEMS.md)** - Variable name resolution issues

---

## Testing Documentation

- **[ADIOS2_TESTS.md](ADIOS2_TESTS.md)** - ADIOS2 I/O testing notes
- **[PNETCDF_TESTS.md](PNETCDF_TESTS.md)** - Parallel NetCDF testing notes
- **[VTK_TESTS.md](VTK_TESTS.md)** - VTK I/O testing notes

---

## Planning & Historical Documents

**Archived documentation from development phases:**

- **[VALIDATION_PLAN.md](VALIDATION_PLAN.md)** - Testing roadmap (2-3 month plan)
- **[WEEK1_TASKS.md](WEEK1_TASKS.md)** - Week 1 implementation tasks (completed)
- **[GPU_AWARE_MPI_PLAN.md](GPU_AWARE_MPI_PLAN.md)** - GPU-aware MPI implementation plan
- **[UNIFIED_NDARRAY_DESIGN.md](UNIFIED_NDARRAY_DESIGN.md)** - Unified distributed design document
- **[UNIFIED_NDARRAY_PROGRESS.md](UNIFIED_NDARRAY_PROGRESS.md)** - Implementation progress tracking
- **[DISTRIBUTED_STREAM_REDESIGN.md](DISTRIBUTED_STREAM_REDESIGN.md)** - Stream redesign document
- **[PHASE3_IO_AUTO_DETECTION.md](PHASE3_IO_AUTO_DETECTION.md)** - Phase 3 implementation plan
- **[PHASE4_STREAM_INTEGRATION.md](PHASE4_STREAM_INTEGRATION.md)** - Phase 4 implementation plan

---

## Obsolete/Legacy Documents

**These documents reference outdated features or are superseded by newer docs:**

- **[FDPOOL.md](FDPOOL.md)** - Old distributed array design (superseded by DISTRIBUTED_NDARRAY.md)
- **[STUDENT_ISSUES.md](STUDENT_ISSUES.md)** - Student project notes

---

## Quick Start

**New to the library?** Read these in order:

1. [MAINTENANCE-MODE.md](MAINTENANCE-MODE.md) - Understand library status and features
2. [STORAGE_BACKENDS.md](STORAGE_BACKENDS.md) - Choose your storage backend
3. [ARRAY_ACCESS.md](ARRAY_ACCESS.md) - Learn array indexing
4. [DISTRIBUTED_NDARRAY.md](DISTRIBUTED_NDARRAY.md) - Use MPI (if needed)
5. [GPU_SUPPORT.md](GPU_SUPPORT.md) - Use GPUs (if needed)

**Looking for examples?** See `../examples/` directory.

---

## Contributing

When adding new documentation:

1. **Place it in the appropriate category above**
2. **Update this README.md with a link**
3. **Use descriptive filenames** (e.g., `FEATURE_NAME.md`)
4. **Include examples and code snippets**
5. **Keep planning docs in "Planning & Historical" section**

---

**Last updated**: 2026-02-14
