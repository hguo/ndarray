# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Distributed memory support for MPI-based domain decomposition
- `distributed_ndarray` class for parallel I/O with automatic/manual decomposition
- `distributed_ndarray_stream` for YAML-based distributed workflows
- Ghost layer exchange functionality for stencil computations
- Dry-run mode for validating configurations without loading data
- Format-specific variable name support (h5_name, nc_name)
- Storage backend abstraction with multiple policies (native, xtensor, Eigen)
- Maintenance mode documentation

### Changed
- Replaced exit() calls with exceptions in error handling macros
- Converted substream classes to templates with StoragePolicy parameter
- Improved error messages with specific format information
- Enhanced CMake configuration for optional dependencies

### Fixed
- HDF5 library linking for all targets that include ndarray headers
- PNetCDF read function syntax and proper #endif placement
- Variable name resolution across different file formats
- Template dependent name lookup issues in stream classes

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
