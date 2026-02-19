# Exception Handling Guide

## Overview

As of version 2026.02, ndarray uses **C++ exceptions** for error handling instead of calling `exit()` directly. This makes the library safer to use as it allows callers to handle errors gracefully without terminating the entire program.

## Exception Hierarchy

```
std::exception
    └── ftk::exception (base class for all ndarray exceptions)
        ├── ftk::file_error
        ├── ftk::feature_not_available
        ├── ftk::invalid_operation
        ├── ftk::not_implemented
        ├── ftk::device_error
        ├── ftk::vtk_error
        ├── ftk::netcdf_error
        ├── ftk::adios2_error
        └── ftk::stream_error
```

## Exception Classes

### `ftk::exception`

**Base class** for all ndarray exceptions.

**Methods**:
- `const char* what() const noexcept` - Full error message
- `int error_code() const noexcept` - Error code from error.hh enum
- `const std::string& message() const noexcept` - User-provided context

**Example**:
```cpp
try {
  arr.read_netcdf("missing.nc", "temperature");
} catch (const ftk::exception& e) {
  std::cerr << "Error: " << e.what() << std::endl;
  std::cerr << "Code: " << e.error_code() << std::endl;
  std::cerr << "Message: " << e.message() << std::endl;
}
```

### `ftk::file_error`

Thrown for **file I/O errors**:
- File not found
- Cannot open file
- Cannot read/write file
- Unrecognized file extension
- File format errors

**Error codes**:
- `ERR_FILE_NOT_FOUND`
- `ERR_FILE_CANNOT_OPEN`
- `ERR_FILE_CANNOT_WRITE`
- `ERR_FILE_CANNOT_READ_EXPECTED_BYTES`
- `ERR_FILE_UNRECOGNIZED_EXTENSION`
- `ERR_FILE_FORMAT`
- `ERR_FILE_FORMAT_AMIRA`

**Example**:
```cpp
try {
  arr.read_h5("nonexistent.h5", "data");
} catch (const ftk::file_error& e) {
  std::cerr << "File error: " << e.what() << std::endl;
  // Retry with different file or exit gracefully
}
```

### `ftk::feature_not_available`

Thrown when using a **feature not compiled in**.

**Error codes**: All `ERR_NOT_BUILT_WITH_*` codes:
- `ERR_NOT_BUILT_WITH_NETCDF`
- `ERR_NOT_BUILT_WITH_HDF5`
- `ERR_NOT_BUILT_WITH_ADIOS2`
- `ERR_NOT_BUILT_WITH_VTK`
- `ERR_NOT_BUILT_WITH_MPI`
- (and others)

**Example**:
```cpp
try {
  arr.read_netcdf("data.nc", "temperature");
} catch (const ftk::feature_not_available& e) {
  std::cerr << "Feature not available: " << e.what() << std::endl;
  std::cerr << "Please recompile with: -DNDARRAY_USE_NETCDF=ON" << std::endl;

  // Fall back to different format
  arr.read_h5("data.h5", "temperature");
}
```

### `ftk::invalid_operation`

Thrown for **invalid array operations**:
- Reshaping empty array
- Unsupported dimensionality
- Invalid component counts
- Device reshape errors

**Error codes**:
- `ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS`
- `ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY`
- `ERR_NDARRAY_RESHAPE_EMPTY`
- `ERR_NDARRAY_RESHAPE_DEVICE`
- `ERR_NDARRAY_UNKNOWN_DEVICE`

**Example**:
```cpp
try {
  ftk::ndarray<float> arr;
  arr.reshapef({});  // Empty dimensions!
} catch (const ftk::invalid_operation& e) {
  std::cerr << "Invalid operation: " << e.what() << std::endl;
}
```

### `ftk::not_implemented`

Thrown for **declared but not-yet-implemented functions**.

**Error code**: `ERR_NOT_IMPLEMENTED`

**Example**:
```cpp
try {
  arr.some_experimental_feature();
} catch (const ftk::not_implemented& e) {
  std::cerr << "Not implemented: " << e.what() << std::endl;
  // Use alternative approach
}
```

### `ftk::device_error`

Thrown for **GPU/accelerator errors**:
- CUDA errors
- SYCL errors
- Unsupported accelerators
- Device memory errors

**Error codes**:
- `ERR_NOT_BUILT_WITH_CUDA`
- `ERR_NOT_BUILT_WITH_SYCL`
- `ERR_ACCELERATOR_UNSUPPORTED`
- `ERR_THREAD_BACKEND_UNSUPPORTED`

**Example**:
```cpp
try {
  arr.to_device(NDARRAY_DEVICE_CUDA);
} catch (const ftk::device_error& e) {
  std::cerr << "Device error: " << e.what() << std::endl;
  // Fall back to CPU
}
```

### `ftk::vtk_error`

Thrown for **VTK-related errors**.

**Error codes**:
- `ERR_VTK_VARIABLE_NOT_FOUND`
- `ERR_VTK_UNSUPPORTED_OUTPUT_FORMAT`

**Example**:
```cpp
try {
  arr.read_vtk_image_data_file("data.vti", "missing_var");
} catch (const ftk::vtk_error& e) {
  std::cerr << "VTK error: " << e.what() << std::endl;
}
```

### `ftk::netcdf_error`

Thrown for **NetCDF-related errors**.

**Error codes**:
- `ERR_NETCDF_MISSING_VARIABLE`
- `ERR_NETCDF_FILE_NOT_OPEN`

**Example**:
```cpp
try {
  arr.read_netcdf("data.nc", "missing_variable");
} catch (const ftk::netcdf_error& e) {
  std::cerr << "NetCDF error: " << e.what() << std::endl;
}
```

### `ftk::adios2_error`

Thrown for **ADIOS2-related errors**.

**Error codes**:
- `ERR_ADIOS2`
- `ERR_ADIOS2_VARIABLE_NOT_FOUND`

**Example**:
```cpp
try {
  auto arr = ftk::ndarray<float>::from_bp("data.bp", "missing_var", 0);
} catch (const ftk::adios2_error& e) {
  std::cerr << "ADIOS2 error: " << e.what() << std::endl;
}
```

### `ftk::stream_error`

Thrown for **stream-related errors**.

**Error codes**:
- `ERR_STREAM_FORMAT`

**Example**:
```cpp
try {
  ftk::stream s;
  s.parse_yaml("invalid.yaml");
} catch (const ftk::stream_error& e) {
  std::cerr << "Stream error: " << e.what() << std::endl;
}
```

---

## Usage Patterns

### Pattern 1: Catch All Errors

```cpp
try {
  ftk::ndarray<float> arr;
  arr.read_netcdf("data.nc", "temperature");
  // ... process data ...
} catch (const ftk::exception& e) {
  // Catches ALL ndarray exceptions
  std::cerr << "ndarray error: " << e.what() << std::endl;
  return EXIT_FAILURE;
}
```

### Pattern 2: Catch Specific Errors

```cpp
try {
  ftk::ndarray<float> arr;
  arr.read_netcdf("data.nc", "temperature");

} catch (const ftk::file_error& e) {
  std::cerr << "File error: " << e.what() << std::endl;
  // Try alternative file
  arr.read_netcdf("backup.nc", "temperature");

} catch (const ftk::feature_not_available& e) {
  std::cerr << "NetCDF not available" << std::endl;
  // Fall back to HDF5
  arr.read_h5("data.h5", "temperature");

} catch (const ftk::exception& e) {
  // Other ndarray errors
  std::cerr << "Other error: " << e.what() << std::endl;
}
```

### Pattern 3: Catch Multiple Levels

```cpp
try {
  ftk::ndarray<float> temp = load_temperature("file.nc");
  ftk::ndarray<float> press = load_pressure("file.nc");

} catch (const ftk::netcdf_error& e) {
  // NetCDF-specific handling
  std::cerr << "NetCDF error: " << e.what() << std::endl;

} catch (const ftk::file_error& e) {
  // File I/O handling
  std::cerr << "File error: " << e.what() << std::endl;

} catch (const ftk::exception& e) {
  // Generic ndarray error
  std::cerr << "ndarray error: " << e.what() << std::endl;

} catch (const std::exception& e) {
  // Standard library exceptions
  std::cerr << "Standard exception: " << e.what() << std::endl;
}
```

### Pattern 4: Check Error Codes

```cpp
try {
  arr.read_netcdf("data.nc", "var");

} catch (const ftk::exception& e) {
  if (e.error_code() == ftk::ERR_FILE_NOT_FOUND) {
    std::cerr << "File not found, using defaults" << std::endl;
    // Use default data
  } else if (e.error_code() == ftk::ERR_NETCDF_MISSING_VARIABLE) {
    std::cerr << "Variable not found, trying alternative" << std::endl;
    // Try different variable name
  } else {
    std::cerr << "Unexpected error: " << e.what() << std::endl;
    throw;  // Re-throw
  }
}
```

### Pattern 5: RAII with Cleanup

```cpp
struct DataLoader {
  ftk::ndarray<float> data;
  bool loaded = false;

  DataLoader(const std::string& file) {
    try {
      data.read_netcdf(file, "temperature");
      loaded = true;
    } catch (const ftk::exception& e) {
      std::cerr << "Failed to load: " << e.what() << std::endl;
      // Object still constructed, but loaded = false
    }
  }

  bool is_loaded() const { return loaded; }
};

// Usage
DataLoader loader("data.nc");
if (loader.is_loaded()) {
  // Process loader.data
} else {
  // Handle failure
}
```

---

## Migration from Old Code

### Before (Old Style)

```cpp
// OLD: Program exits on error
ftk::ndarray<float> arr;
arr.read_netcdf("data.nc", "temperature");
// If file doesn't exist: program exits with exit(1)
```

### After (New Style)

```cpp
// NEW: Caller can handle errors
try {
  ftk::ndarray<float> arr;
  arr.read_netcdf("data.nc", "temperature");
} catch (const ftk::exception& e) {
  std::cerr << "Error: " << e.what() << std::endl;
  // Handle error gracefully
  return EXIT_FAILURE;
}
```

### Backward Compatibility

The `fatal()` function now **throws exceptions** instead of calling `exit()`. However, if you don't catch these exceptions, the program will terminate (default C++ behavior), which is similar to the old behavior but cleaner.

**Old behavior** (before 2026.02):
```cpp
fatal(ERR_FILE_NOT_FOUND);  // Called exit(1)
```

**New behavior** (2026.02+):
```cpp
fatal(ERR_FILE_NOT_FOUND);  // Throws ftk::file_error
```

**Migration**: No code changes required if you weren't catching errors before. But now you CAN catch them:

```cpp
// Before: No way to prevent exit
arr.read_netcdf("file.nc", "var");  // Exits on error

// After: Can catch and handle
try {
  arr.read_netcdf("file.nc", "var");
} catch (const ftk::exception& e) {
  // Recover instead of exiting
}
```

---

## Best Practices

### ✅ DO

1. **Catch specific exceptions when possible**
   ```cpp
   catch (const ftk::file_error& e)  // Specific
   catch (const ftk::exception& e)   // Generic fallback
   ```

2. **Use exception hierarchy**
   ```cpp
   try {
     // ...
   } catch (const ftk::netcdf_error& e) {
     // NetCDF-specific handling
   } catch (const ftk::file_error& e) {
     // File-level handling
   } catch (const ftk::exception& e) {
     // Generic handling
   }
   ```

3. **Add context when rethrowing**
   ```cpp
   try {
     arr.read_netcdf(file, var);
   } catch (const ftk::exception& e) {
     throw ftk::file_error(e.error_code(),
       "Failed to load " + file + ": " + e.message());
   }
   ```

4. **Use RAII for cleanup**
   ```cpp
   class DataManager {
     ftk::ndarray<float> data;
   public:
     DataManager(const std::string& file) {
       data.read_netcdf(file, "var");
       // If exception thrown, destructor still called
     }
   };
   ```

### ❌ DON'T

1. **Don't catch by value** (loses derived type info)
   ```cpp
   // BAD
   catch (ftk::exception e)  // Slices exception!

   // GOOD
   catch (const ftk::exception& e)
   ```

2. **Don't swallow exceptions silently**
   ```cpp
   // BAD
   try {
     arr.read_netcdf(file, var);
   } catch (...) {
     // Silent failure - very bad!
   }

   // GOOD
   try {
     arr.read_netcdf(file, var);
   } catch (const ftk::exception& e) {
     std::cerr << "Error: " << e.what() << std::endl;
     // Handle or rethrow
   }
   ```

3. **Don't mix C++ exceptions with MPI**
   ```cpp
   // BAD: Exception on rank 0, deadlock on others
   try {
     if (rank == 0) {
       throw ftk::exception("error");
     }
     MPI_Barrier(comm);  // Other ranks wait forever
   } catch (...) {}

   // GOOD: Communicate errors across ranks
   int local_error = 0;
   try {
     // ... operation ...
   } catch (const ftk::exception& e) {
     std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
     local_error = 1;
   }
   int global_error;
   MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_SUM, comm);
   if (global_error > 0) {
     // All ranks handle error together
   }
   ```

---

## Exception Safety Guarantees

### Strong Exception Safety

Most ndarray operations provide **strong exception safety**: if an exception is thrown, the array remains in its original state.

**Example**:
```cpp
ftk::ndarray<float> arr;
arr.reshapef(10, 20);  // State 1

try {
  arr.read_netcdf("file.nc", "var");  // Throws
} catch (...) {
  // arr still has shape (10, 20) - original state preserved
}
```

### Basic Exception Safety

Some operations provide **basic exception safety**: if an exception is thrown, the array is in a valid but unspecified state.

**Examples**:
- Device transfers
- Reshape operations
- Some I/O operations

### No-throw Guarantee

Query operations are typically **noexcept**:

```cpp
size_t nd = arr.nd();        // noexcept
size_t size = arr.size();    // noexcept
T* ptr = arr.data();         // noexcept
```

---

## Testing Exception Handling

### Example Test

```cpp
#include <ndarray/ndarray.hh>
#include <cassert>

void test_file_not_found() {
  ftk::ndarray<float> arr;

  bool caught = false;
  try {
    arr.read_netcdf("nonexistent.nc", "var");
  } catch (const ftk::file_error& e) {
    caught = true;
    assert(e.error_code() == ftk::ERR_FILE_NOT_FOUND);
  }

  assert(caught && "Should have thrown file_error");
}

int main() {
  test_file_not_found();
  std::cout << "All tests passed" << std::endl;
  return 0;
}
```

---

## Performance Impact

**Minimal**: Exception handling has near-zero cost when no exceptions are thrown (zero-cost exception model in modern C++). When exceptions ARE thrown, the cost is acceptable since errors are by definition exceptional cases.

---

## See Also

- [error.hh](../include/ndarray/error.hh) - Exception class definitions
- [CRITICAL_ANALYSIS.md](../CRITICAL_ANALYSIS.md) - Error handling design critique
- [C++ Exception Handling Best Practices](https://isocpp.org/wiki/faq/exceptions)
