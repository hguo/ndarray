# Making YAML an Optional Dependency

## Problem

Original ndarray required yaml-cpp as a mandatory dependency, even for users who don't need YAML stream functionality.

From `CMakeLists.txt` (old):
```cmake
find_package (yaml-cpp REQUIRED)  # Always required!
```

This forces all users to install yaml-cpp, increasing build complexity for minimal builds.

## Solution

YAML support is now optional and controlled via CMake option:

```bash
# With YAML (default behavior, auto-detect)
cmake .. -DNDARRAY_USE_YAML=AUTO

# Explicitly enable YAML
cmake .. -DNDARRAY_USE_YAML=TRUE

# Disable YAML (minimal build)
cmake .. -DNDARRAY_USE_YAML=FALSE
```

## Implementation Details

### 1. CMake Changes

**File:** `CMakeLists.txt`

Added option:
```cmake
ndarray_option (YAML "Use YAML (for stream functionality)" AUTO)
```

Changed from `REQUIRED` to conditional:
```cmake
if (NDARRAY_USE_YAML STREQUAL AUTO)
  find_package (yaml-cpp QUIET)
elseif (NDARRAY_USE_YAML)
  find_package (yaml-cpp REQUIRED)
endif ()
if (yaml-cpp_FOUND)
  set (NDARRAY_HAVE_YAML ON)
endif ()
```

### 2. Config Header Changes

**File:** `include/ndarray/config.hh.in`

Added macro:
```cpp
#cmakedefine NDARRAY_HAVE_YAML 1
```

### 3. Code Changes

**File:** `include/ndarray/ndarray_group_stream.hh`

Wrapped entire file:
```cpp
#if NDARRAY_HAVE_YAML

// ... all stream functionality ...

#endif // NDARRAY_HAVE_YAML
```

**Rationale:** The entire `ndarray_group_stream.hh` header is dedicated to YAML-based stream functionality. Without YAML, users simply don't include this header.

### 4. Test Changes

**File:** `tests/CMakeLists.txt`

Made stream tests conditional:
```cmake
if (NDARRAY_HAVE_YAML)
  add_executable(test_ndarray_stream ...)
  target_link_libraries(test_ndarray_stream yaml-cpp::yaml-cpp)
  add_test(NAME ndarray_stream ...)
endif()
```

## Usage

### Scenario 1: Full Build (with YAML streams)

```bash
# Auto-detect yaml-cpp
cmake ..

# Or explicitly enable
cmake .. -DNDARRAY_USE_YAML=TRUE

# Your code can use YAML streams
#include <ndarray/ndarray_group_stream.hh>
```

### Scenario 2: Minimal Build (no YAML)

```bash
cmake .. -DNDARRAY_USE_YAML=FALSE

# Your code uses only core ndarray
#include <ndarray/ndarray.hh>  # Works
// #include <ndarray/ndarray_group_stream.hh>  # Don't include this
```

### Scenario 3: Conditional Code

```cpp
#include <ndarray/ndarray.hh>

#if NDARRAY_HAVE_YAML
#include <ndarray/ndarray_group_stream.hh>
#endif

int main() {
  // Core functionality always available
  ftk::ndarray<double> arr;
  arr.reshapef(100);

#if NDARRAY_HAVE_YAML
  // YAML streams only if available
  ftk::ndarray_group_stream stream("config.yaml");
  auto g = stream.read(0);
#endif

  return 0;
}
```

## What Still Requires YAML

Only YAML-based stream functionality requires yaml-cpp:
- `#include <ndarray/ndarray_group_stream.hh>`
- `ftk::ndarray_group_stream` class
- `ftk::stream` class
- YAML configuration parsing

## What Works Without YAML

Everything else works without yaml-cpp:
- Core ndarray arrays (`ftk::ndarray<T>`)
- File I/O (NetCDF, HDF5, binary, VTK)
- Array operations (reshape, slice, transpose, etc.)
- Zero-copy optimization (`get_ref()`)
- Variable name matching utilities
- Examples (except those using streams)
- Most tests (except `test_ndarray_stream`)

## Benefits

1. **Simpler minimal builds**: No yaml-cpp dependency needed for basic usage
2. **Faster build times**: One less dependency to configure/compile
3. **Easier deployment**: Reduced system requirements
4. **Backward compatible**: Full builds still work exactly as before

## Migration Guide

### For Users Who DON'T Need YAML Streams

No changes needed if you're not using `ndarray_group_stream`:

```cpp
// This code works with or without YAML
#include <ndarray/ndarray.hh>

ftk::ndarray<double> arr;
arr.reshapef(1000);
arr.to_file("output.nc");  // NetCDF I/O works without YAML
```

### For Users Who DO Need YAML Streams

Also no changes needed - YAML is auto-detected by default:

```cpp
#include <ndarray/ndarray_group_stream.hh>  // Requires YAML

ftk::ndarray_group_stream stream("config.yaml");
auto g = stream.read(0);
```

Just ensure yaml-cpp is installed, or explicitly enable it:
```bash
cmake .. -DNDARRAY_USE_YAML=TRUE
```

### For Package Maintainers

Update build scripts to make yaml-cpp optional:

```spec
# RPM spec example
%if 0%{?with_yaml}
BuildRequires: yaml-cpp-devel
%endif

%cmake \
  -DNDARRAY_USE_YAML=%{?with_yaml:TRUE}%{!?with_yaml:FALSE}
```

## Summary

| Feature | YAML Required | Alternative |
|---------|---------------|-------------|
| Core arrays | No | Built-in |
| NetCDF/HDF5 I/O | No | Direct file I/O |
| YAML streams | Yes | Use direct file I/O |
| Examples | Mostly no | Most examples don't need YAML |
| Tests | Mostly no | Only stream test needs YAML |

**Default behavior:** AUTO (detects yaml-cpp if available)

**Minimal build:** `-DNDARRAY_USE_YAML=FALSE` (removes yaml-cpp dependency)
