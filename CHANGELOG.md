# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.6] - 2026-02-28

### Added
- **Comprehensive test suite** for arithmetic, statistics, data operations, and gradient functions

### Fixed
- **reshapef() backend dimension mismatch** - Storage backends (xtensor, Eigen) now receive C-order dims consistent with ndarray's internal state, fixing shape inconsistencies in `get_xarray()` and `get_matrix()` interop
- **Eigen 2D resize** - Reversed resize args to maintain correct (rows, cols) layout when receiving C-order dims
- **CUDA transpose** - Fixed width/height swap and compile test with nvcc
- **reshapec() overloads** - Fixed reshapec() hiding base class vector overloads
- **11 code quality issues** across codebase
- **CUDA error checking** improvements and documentation link fixes

---

## [0.0.3] - 2026-02-21

### Added
- **Getting Started Guide** (`docs/GETTING_STARTED.md`) - Comprehensive 15-minute tutorial
- **Documentation Index** (`docs/INDEX.md`) - Organized topic-based navigation
- **ADIOS2 YAML stream tests** - Test suite for ADIOS2 time-series via YAML configuration
- **ADIOS2 parallel tests** - MPI-parallel ADIOS2 I/O tests with domain decomposition
- **ADIOS2 CI workflow** - GitHub Actions job with ADIOS2 caching
- **PerformGets()** call after ADIOS2 Get() operations (fixes data reading)

### Fixed
- **ADIOS2 data reading** - All data was reading as zeros, now fixed
- **ADIOS2 multi-step reading** - Properly reads all timesteps with correct 3D shape
- **ADIOS2 MPI deadlock** - Use `MPI_COMM_SELF` for serial tests, `MPI_COMM_WORLD` for parallel
- **ADIOS2 stream reader** - Pass `step=0` to read first timestep from BP files
- **ADIOS2 stream test MPI** - Only rank 0 creates test files, preventing file corruption in multi-rank runs
- **Mode::ReadRandomAccess** - All ADIOS2 file opens now use correct mode for BeginStep/EndStep files
- **Documentation API references** - Corrected all references to non-existent `distributed_ndarray` and `distributed_stream` classes

### Changed
- **Version badge** updated to 0.0.3-alpha
- **Documentation organization** - Moved internal/progress docs to subdirectories
- **README** - Added Quick Start section with example code
- **ADIOS2 workflow** - Enabled MPI support (`NDARRAY_USE_MPI=ON`)

### Documentation
- Reorganized docs directory with clear user/developer separation
- Created archive for historical/internal documentation
- Updated all documentation links and cross-references
- Added quick reference card to INDEX.md

---

## [Unreleased]

**Major focus**: Stabilization, distributed computing, and GPU support. This release completes GPU-aware MPI features, removes misleading performance claims, expands CI coverage from 5 to 14 configurations, and fixes numerous compilation issues across platforms and storage backends. The library is now reliably buildable on Linux (GCC, Clang), macOS, with C++17/C++20, and properly handles all optional dependencies.

### Added
- **GPU-aware MPI**: Complete 1D, 2D, and 3D support for distributed GPU arrays
  - CUDA kernels for pack/unpack boundary data in all dimensions
  - GPU-direct MPI with zero host staging
  - Automatic fallback to CPU staging when GPU-aware MPI unavailable
  - Runtime detection of GPU-aware MPI support
- **Documentation index** (`docs/README.md`) organizing 39 documentation files into categories
- **Comprehensive CI coverage**: 14 build configurations testing all major features
  - Basic builds (GCC, Clang, Debug, Release, Linux, macOS)
  - Storage backends (Eigen + xtensor)
  - All I/O formats (NetCDF, HDF5, YAML, PNG, VTK)
  - MPI and PNetCDF support
  - Sanitizers (AddressSanitizer)
  - C++ standards (C++17, C++20)
- **Distributed memory support** for MPI-based domain decomposition
  - `distributed_ndarray` class for parallel I/O with automatic/manual decomposition
  - Ghost layer exchange functionality for stencil computations (1D, 2D, 3D)
  - Two-pass ghost exchange algorithm for proper corner cell handling
  - Global/local index conversion for distributed algorithms
- **Storage backend abstraction** with multiple policies (native, xtensor, Eigen)
  - Zero migration cost - existing code unchanged
  - Compile-time policy selection
  - Preprocessor guards for optional backends
- Format-specific variable name support (h5_name, nc_name)
- Dry-run mode for validating configurations without loading data
- Maintenance mode documentation

### Changed
- **Removed all performance claims** from documentation - library focus is I/O functionality, not speed
  - Removed "5-10x faster" claims from Eigen storage
  - Removed "2-4x faster" claims from xtensor storage
  - Removed performance comparison tables
  - Changed "high-performance" to capability descriptions
- **Replaced exit() calls with exceptions** in error handling macros - library is now safe to call
- **Switched from OpenMPI to MPICH** in CI for better stability
- **Storage backend includes** now properly guarded with `NDARRAY_HAVE_*` preprocessor checks
- Converted substream classes to templates with StoragePolicy parameter
- Enhanced CMake configuration to respect `OFF` settings for optional dependencies
- Improved error messages with specific format information
- Enabled C language in CMake for MPI support

### Fixed
- **Template specialization issues** across storage policies
  - Changed `vtk_data_type()`, `h5_mem_type_id()`, `mpi_dtype()`, `nc_dtype()` from full specializations to general templates with `if constexpr`
  - Fixes compilation with Eigen and xtensor storage backends
- **CMake cache issues** with Eigen and xtensor
  - Explicitly set `NDARRAY_HAVE_EIGEN/XTENSOR` to FALSE when `USE_*=OFF`
  - Prevents cached configuration from overriding explicit disable
- **xtensor compatibility** with version 0.24.x
  - Fixed include path from `<xtensor/containers/xarray.hpp>` to `<xtensor/xarray.hpp>`
  - Fixed default constructor to initialize empty state
  - Changed `reshape()` to use `resize()` for proper behavior
- **Ghost exchange corner cells** in distributed arrays
  - Implemented two-pass exchange algorithm
  - Fixed bug where high side was checking `ghost_low` instead of `ghost_high`
- **GPU memory transfer** - `nelem()` now checks `dims.empty()` instead of `storage_.empty()`
- **CI/CD stability**
  - PNetCDF build caching reduces CI time from 5-10 min to 30 sec
  - VTK build includes Qt5 dependencies and MPI support
  - Clean build directory for VTK to prevent CMake cache interference
- HDF5 library linking for all targets that include ndarray headers
- PNetCDF read function syntax and proper #endif placement
- Variable name resolution across different file formats
- Template dependent name lookup issues in stream classes
- Missing `<iomanip>` header includes

### Removed
- All unvalidated performance claims from documentation
  - Library focus is I/O abstraction, not speed optimization
  - Performance benchmarks were never conducted
  - Removed "10x faster", "5-10x faster", "2-4x faster" marketing language

### Deprecated
- None

## [0.1.0] - Initial Release

### Added
- Core ndarray implementation with multi-dimensional array support
- NetCDF, HDF5, VTK, ADIOS2 I/O backends
- YAML-based stream configuration
- MPI support for parallel I/O (PNetCDF)
- GPU memory management (CUDA, SYCL)
- Zero-copy optimizations
- Convolution operations
- F/C ordering support
