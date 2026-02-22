# ndarray Roadmap

This document outlines planned features and improvements for future releases.

---

## Future Features

### C++23 mdspan Storage Backend

**Status**: Proposed
**Target**: v0.1.0 or later
**Priority**: Medium

#### Overview

Add `mdspan_storage` as a fourth storage backend option, leveraging the C++23 standard library's `std::mdspan` for multi-dimensional array views.

#### Motivation

- **Standards-based**: Uses C++23 standard library, no external dependencies when compiler supports it
- **Future-proof**: Part of the C++ standard, guaranteed long-term support
- **Interoperability**: Standard way to pass multi-dimensional views between libraries
- **Performance**: mdspan enables compiler optimizations with layout policies and accessor policies
- **Zero-cost abstraction**: Fits existing policy-based design pattern seamlessly

#### Design Approach

```cpp
// Storage backend interface
template<typename T>
struct mdspan_storage {
  using value_type = T;
  using extents_type = std::dextents<size_t, std::dynamic_extent>;

  std::vector<T> data_;           // Owns the data
  std::vector<size_t> shape_;     // Dimension sizes

  // Required storage interface
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  size_t size() const { return data_.size(); }
  void resize(size_t n) { data_.resize(n); }
  void fill(const T& value) { std::fill(data_.begin(), data_.end(), value); }

  // mdspan-specific: create view with shape
  auto view() {
    return std::mdspan<T, extents_type>(data_.data(), to_extents(shape_));
  }
};

// Usage
ftk::ndarray<float, ftk::mdspan_storage> arr;
arr.reshapec(100, 200, 50);
auto view = arr.get_mdspan_view();  // std::mdspan view
```

#### Implementation Tasks

1. **CMake configuration**:
   ```cmake
   ndarray_option(USE_MDSPAN "Enable C++23 mdspan storage backend" AUTO)
   ```
   - Detect C++23 support and `<mdspan>` availability
   - Fallback to standalone mdspan library for C++17/20
   - Set `NDARRAY_HAVE_MDSPAN` flag

2. **Storage backend**:
   - Create `include/ndarray/storage_mdspan.hh`
   - Implement storage interface: `data()`, `size()`, `resize()`, `fill()`
   - Handle dimension tracking for extents conversion
   - Support both C-order and Fortran-order layouts

3. **Layout policies**:
   - `std::layout_right` for C-order (row-major)
   - `std::layout_left` for Fortran-order (column-major)
   - `std::layout_stride` for general strides

4. **Public API extensions** (optional):
   - `arr.to_mdspan()` - create mdspan view
   - `arr.from_mdspan(view)` - construct from mdspan
   - Interop with other mdspan-based libraries

5. **Testing**:
   - Storage backend tests (following `test_storage_backends.cpp` pattern)
   - I/O tests with mdspan storage
   - Layout conversion tests
   - CI job with C++23 compiler

6. **Documentation**:
   - Update `docs/STORAGE_BACKENDS.md`
   - Add mdspan examples to README
   - Document compiler requirements

#### Considerations

**Compiler Support**:
- GCC 14+ (C++23)
- Clang 18+ (C++23)
- MSVC 19.37+ (VS 2022 17.7+)
- Fallback: Use [kokkos/mdspan](https://github.com/kokkos/mdspan) for C++17/20

**Ownership Model**:
- mdspan is non-owning (view only)
- Pair with `std::vector` for data ownership
- Clear semantics: `mdspan_storage` owns data, exposes views

**Layout Flexibility**:
- Support different layout policies (row-major, column-major, strided)
- Allow users to choose layout via template parameter
- Default to Fortran-order (layout_left) for consistency with ndarray internals

**API Design**:
- Keep storage backend interface unchanged
- mdspan views as optional feature (not required for basic usage)
- Maintain backward compatibility

#### Benefits

1. **Modern C++**: Positions ndarray as forward-looking library
2. **Zero external dependencies**: When using C++23 compiler
3. **Standard interoperability**: Works with any mdspan-compatible library
4. **Performance**: Compiler can optimize based on layout information
5. **Fits existing design**: No changes to storage policy architecture

#### Timeline

- **Research phase**: Evaluate compiler support, test standalone mdspan library
- **Prototype**: Implement basic mdspan_storage backend
- **Testing**: Comprehensive tests across compilers
- **Documentation**: Usage guide and API reference
- **Release**: Include in v0.1.0 or later

#### Related Work

- C++23 standard: [P0009R18](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r18.html)
- Standalone library: [kokkos/mdspan](https://github.com/kokkos/mdspan)
- Similar libraries: xarray, xtensor (no mdspan yet)

---

## Other Potential Features

### API Stabilization → v1.0.0
- Remove deprecated methods or bump to v2.0
- Commit to stable API contract
- Production validation with real users

### Build System Simplification
- Reduce CMake complexity (currently 512 lines)
- Modularize optional dependencies
- Improve configuration messages

### Performance Optimization
- Benchmark suite expansion
- SIMD optimizations for operations
- Lazy evaluation for expressions

### Extended GPU Support
- Additional compute kernels (stencils, reductions)
- Multi-GPU support
- GPU-aware MPI improvements

---

**Last Updated**: 2026-02-21
**Current Version**: v0.0.3-alpha
