# Maintenance Mode Notice

**Last Updated**: 2026-02-14

## Current Status

ndarray is currently in **maintenance mode**. This means:

- ✅ **Bug fixes**: Critical bugs will be fixed
- ✅ **Existing features**: Core functionality is maintained
- ⚠️ **New features**: Limited - only essential additions
- ⚠️ **API changes**: Minimal - backward compatibility prioritized
- ❌ **Major refactoring**: Not planned

## What This Means for Users

### ✅ You Should Use ndarray If:

1. **You're already using it** in production code
   - Your existing code will continue to work
   - Critical bugs will be fixed
   - You need the YAML stream abstraction features

2. **You need specific features** that work well
   - NetCDF/HDF5/VTK I/O
   - YAML-based stream configuration
   - Variable name pattern matching
   - Time-series data abstraction

3. **You're working with FTK** (Feature Tracking Kit)
   - ndarray is the native array type for FTK
   - Tight integration with FTK workflows

### ⚠️ Consider Alternatives If:

1. **Starting a new project** from scratch
   - [Eigen](https://eigen.tuxfamily.org/) - Mature, optimized linear algebra
   - [xtensor](https://github.com/xtensor-stack/xtensor) - NumPy-like API, expression templates
   - [Boost.MultiArray](https://www.boost.org/doc/libs/1_84_0/libs/multi_array/) - Stable, well-tested
   - [mdspan](https://github.com/kokkos/mdspan) - C++23 standard (future-proof)

2. **Need cutting-edge performance**
   - Eigen and xtensor have better SIMD optimization
   - Expression templates for lazy evaluation
   - More mature optimization

3. **Want extensive documentation**
   - Limited API documentation currently
   - Mature alternatives have comprehensive docs

4. **Need commercial support**
   - ndarray is a research project with minimal support
   - Commercial alternatives available

## Implementation Status

### ✅ Fully Implemented and Tested

#### Core Array Operations
- Basic array creation, reshaping, indexing
- Multi-dimensional array support (up to 5D tested)
- Fortran and C ordering
- Data type conversions
- Move semantics and zero-copy access

#### I/O Formats (Basic)
- **NetCDF**: Read/write single variables
- **HDF5**: Read/write datasets
- **VTK**: Read/write image data and unstructured grids
- **Binary**: Raw binary file I/O
- **PNG**: Image I/O for visualization

#### Parallel I/O
- **MPI support**: Basic parallel I/O patterns
- **Parallel-NetCDF**: Collective I/O operations (recently completed)

#### Stream Abstraction
- **YAML configuration**: Parse stream definitions
- **Multiple formats**: NetCDF, HDF5, ADIOS2, VTK, binary, synthetic
- **Time-series handling**: Read sequential timesteps
- **Variable aliasing**: Handle inconsistent variable names
- **Format-specific names**: h5_name, nc_name patterns (recently added)

### ⚠️ Partially Implemented

#### GPU Support
- **CUDA**: Basic to_device/to_host (experimental)
- **HIP**: AMD GPU support (experimental)
- **SYCL**: Cross-platform acceleration (experimental)
- ⚠️ **Limitation**: No GPU kernels, just memory transfer

#### Storage Backend System
- **Native storage**: Default std::vector-based (fully tested)
- **xtensor storage**: SIMD-accelerated backend (implemented, needs testing)
- **Eigen storage**: Linear algebra backend (implemented, needs testing)
- **Policy-based design**: Choose storage at compile-time
- **Zero migration cost**: Existing code works unchanged (backward compatible)

#### ADIOS2 Support
- **Basic I/O**: Read/write BP files
- ⚠️ **Limitation**: Advanced features (streaming, staging) untested

### ❌ Not Implemented (Despite API Declarations)

#### Write Functions
- `write_pnetcdf()` - **Declared but not implemented**
- Some symmetric write functions may be missing

#### Advanced Features
- SIMD vectorization - not implemented
- Expression templates - not implemented
- Lazy evaluation - not implemented
- Thread safety - not guaranteed

## Known Limitations

### 1. API Inconsistencies

**Variable naming**:
```cpp
arr.dim(0)     // deprecated
arr.dimf(0)    // new (Fortran-order dimensions)
arr.ncd        // cryptic abbreviation
arr.tv         // cryptic abbreviation
```

**I/O function naming**:
```cpp
void read_netcdf(...)    // verb_format
void read_pnetcdf_all()  // verb_format_modifier
static ndarray from_bp() // from_format (different pattern)
```

### 2. 57 Deprecated Functions

Many deprecated functions still exist for backward compatibility:
- `operator()` overloads (14 variants)
- Old `reshape()` functions
- Old `dim()` and `at()` functions

**Impact**: API can be confusing for new users.

### 3. Build Complexity

15 optional dependencies = 32,768 possible configurations:
- Not all combinations are tested
- Some configurations may not compile
- CMake configuration can be complex

**Mitigation**: Use provided build examples for common configurations.

### 4. No Python Bindings

Despite `NDARRAY_HAVE_PYBIND11` flag:
- Bindings are not implemented
- No PyPI package available

Use NumPy or Xarray for Python projects.

### 5. Header-Only Build Issues

- Large compilation times when including all features
- Template instantiation bloat
- Each translation unit recompiles everything

### 6. Error Messages

Some error messages may not be helpful:
- Limited context in some cases
- Stack traces not always available
- Cryptic variable names in errors

## Migration Guidance

### From ndarray to Eigen

```cpp
// ndarray
ftk::ndarray<double> arr;
arr.reshapef(10, 20);
arr.f(i, j) = value;

// Eigen equivalent
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(10, 20);
mat(i, j) = value;  // Column-major by default (like Fortran)
```

### From ndarray to xtensor

```cpp
// ndarray
ftk::ndarray<double> arr;
arr.reshapef(10, 20);
arr.f(i, j) = value;

// xtensor equivalent
xt::xarray<double> arr = xt::zeros<double>({10, 20});
arr(i, j) = value;  // Fortran-order: xt::layout_type::column_major
```

### Keeping YAML Stream Features

If you need the YAML stream abstraction:
```cpp
// ndarray provides unique value here
ftk::stream s;
s.parse_yaml("config.yaml");
for (int t = 0; t < s.total_timesteps(); t++) {
  auto data = s.read(t);  // Abstracts format differences
  // ... process data ...
}
```

**New Approach**: Use xtensor/Eigen storage directly (Feb 2026):
```cpp
// Use ndarray with xtensor storage for both I/O and computation
ftk::stream<ftk::xtensor_storage> s;
s.parse_yaml("config.yaml");

auto data = s.read(t);  // Returns ndarray_group<xtensor_storage>
const auto& temp = data->get_ref<double>("temperature");
// temp is ndarray<double, xtensor_storage> - fast SIMD operations

// Or use Eigen storage for linear algebra
ftk::stream<ftk::eigen_storage> s2;
// ... optimized matrix operations ...
```

This eliminates the need for manual conversion while maintaining the YAML stream abstraction.

## Support and Contact

### Getting Help

1. **GitHub Issues**: [hguo/ndarray/issues](https://github.com/hguo/ndarray/issues)
   - Report bugs
   - Request critical bug fixes
   - Ask questions

2. **Documentation**: Limited but improving
   - See `docs/` directory
   - README.md for overview
   - Individual feature docs

3. **Examples**: Learn by example
   - See `tests/` directory
   - Working code for all features

### What to Expect

- **Bug reports**: Will be addressed for critical issues
- **Feature requests**: Likely won't be implemented (maintenance mode)
- **Pull requests**: Welcome for bug fixes, carefully reviewed for features
- **Response time**: Best effort (research project, not commercial)

## Roadmap

### Short-term (3-6 months)

- ✅ Fix critical safety issues (exit() calls) - **COMPLETED**
- ✅ Enable disabled tests (Test 13) - **COMPLETED**
- ✅ Complete PNetCDF implementation - **COMPLETED**
- 🔲 Document known limitations - **In Progress**
- 🔲 Add basic API documentation (Doxygen)

### Medium-term (6-12 months)

- 🔲 Improve error messages
- 🔲 Add more examples
- 🔲 Performance benchmarks for I/O operations
- 🔲 CI/CD for major configurations

### Long-term (1-2 years)

- ✅ **Templated storage backend** - **COMPLETED** (Option B implemented)
  - ndarray is now a thin wrapper with pluggable backends
  - Users can choose native, xtensor, or Eigen storage
  - Zero migration cost (native storage is default)
- 🔲 Extract YAML stream abstraction as standalone library
- 🔲 Gradual migration path for existing users

### Not Planned

- ❌ Python bindings
- ❌ SIMD optimization
- ❌ Expression templates
- ❌ Major API redesign
- ❌ New I/O format support (beyond critical needs)

## Recent Improvements (2026)

- ✅ **Templated storage backend system** (Feb 2026)
  - Policy-based design for choosing storage implementation
  - Native storage (std::vector) - default, 100% backward compatible
  - xtensor storage - SIMD vectorization, NumPy-like semantics
  - Eigen storage - optimized linear algebra, BLAS/LAPACK integration
  - Zero migration cost - existing code unchanged
  - See [Storage Backends Guide](docs/STORAGE_BACKENDS.md)

- ✅ **Exception-based error handling** (Feb 2026)
  - Replaced exit() calls with exceptions
  - Library is now safe for production use

- ✅ **HDF5 timesteps per file** (Feb 2026)
  - Multiple timesteps per HDF5 file
  - Format-specific variable names (h5_name, nc_name)

- ✅ **PNetCDF collective I/O** (Feb 2026)
  - Implemented read_pnetcdf_all()
  - Full parallel NetCDF support

- ✅ **Documentation improvements** (Feb 2026)
  - Error handling guide
  - HDF5 features documented
  - Variable naming patterns clarified
  - Storage backends comprehensive documentation

## Why Maintenance Mode?

### Honest Assessment

1. **Limited Resources**: Research project with one maintainer
2. **Competing Solutions**: Mature alternatives (Eigen, xtensor) exist
3. **Technical Debt**: 57 deprecated functions, inconsistent API
4. **Focus Shift**: Resources better spent on FTK research

### Strategic Decision

Rather than:
- Add features that mature libraries already have better
- Maintain increasingly complex codebase
- Promise features we can't deliver

We choose to:
- **Be honest** about current status
- **Support existing users** with bug fixes
- **Focus resources** on FTK research
- **Recommend alternatives** for new projects

## Conclusion

ndarray **works** for its intended use case (FTK and time-series scientific data I/O), but:

- Not a general-purpose array library
- Not competing with Eigen/xtensor
- Focused on niche features (YAML streams, variable naming)

**If it solves your specific problem**, use it with confidence - we'll keep it working.

**If you're starting fresh**, consider mature alternatives that better match modern C++ practices and have broader community support.

---

**Questions?** Open an issue on GitHub: https://github.com/hguo/ndarray/issues

**Alternative Recommendations**:
- **Eigen**: https://eigen.tuxfamily.org/
- **xtensor**: https://github.com/xtensor-stack/xtensor
- **Boost.MultiArray**: https://www.boost.org/doc/libs/release/libs/multi_array/
