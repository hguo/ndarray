# Error Handling in ndarray

## Overview

ndarray uses C++ exceptions for error handling. **Library code never calls `exit()` or `abort()`**, ensuring that applications can properly handle errors and clean up resources.

## Exception Hierarchy

All ndarray exceptions derive from `ftk::exception`, which inherits from `std::exception`:

```cpp
std::exception
  └── ftk::exception
      ├── ftk::file_error            // File I/O errors
      ├── ftk::feature_not_available // Missing compile-time feature
      ├── ftk::invalid_operation     // Invalid array operations
      ├── ftk::not_implemented       // Feature not yet implemented
      ├── ftk::device_error          // GPU/accelerator errors
      ├── ftk::vtk_error             // VTK-specific errors
      ├── ftk::netcdf_error          // NetCDF/PNetCDF errors
      ├── ftk::adios2_error          // ADIOS2 errors
      └── ftk::stream_error          // Stream configuration errors
```

## Basic Usage

### Catching All ndarray Errors

```cpp
try {
  ftk::ndarray<double> arr;
  arr.read_netcdf("data.nc", "temperature");
} catch (const ftk::exception& e) {
  std::cerr << "Error: " << e.what() << std::endl;
  std::cerr << "Error code: " << e.error_code() << std::endl;
  // Handle error appropriately
}
```

### Catching Specific Error Types

```cpp
try {
  arr.read_netcdf("data.nc", "temperature");
} catch (const ftk::netcdf_error& e) {
  std::cerr << "NetCDF error: " << e.what() << std::endl;
  // Handle NetCDF-specific error
} catch (const ftk::file_error& e) {
  std::cerr << "File error: " << e.what() << std::endl;
  // Handle file I/O error
} catch (const ftk::exception& e) {
  std::cerr << "Other ndarray error: " << e.what() << std::endl;
  // Handle other errors
}
```

## Error Codes

Error codes are defined in `<ndarray/error.hh>`:

```cpp
// File errors (1000-1999)
ERR_FILE_NOT_FOUND
ERR_FILE_CANNOT_OPEN
ERR_FILE_CANNOT_WRITE
ERR_FILE_FORMAT

// Feature not available errors (2000-2999)
ERR_NOT_BUILT_WITH_NETCDF
ERR_NOT_BUILT_WITH_HDF5
ERR_NOT_BUILT_WITH_PNETCDF

// Array operation errors (3000-3999)
ERR_NDARRAY_RESHAPE_EMPTY
ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY

// NetCDF errors (6500-6999)
ERR_NETCDF_MISSING_VARIABLE
ERR_NETCDF_IO
ERR_PNETCDF_IO

// And many more...
```

### Querying Error Codes

```cpp
try {
  arr.read_netcdf("data.nc", "missing_var");
} catch (const ftk::netcdf_error& e) {
  if (e.error_code() == ftk::ERR_NETCDF_MISSING_VARIABLE) {
    std::cerr << "Variable not found in file" << std::endl;
    // Try alternative variable name
  } else if (e.error_code() == ftk::ERR_NETCDF_IO) {
    std::cerr << "NetCDF I/O operation failed" << std::endl;
    // Check file accessibility
  }
}
```

## NetCDF/PNetCDF Error Handling

### NC_SAFE_CALL and PNC_SAFE_CALL Macros

These macros wrap NetCDF/PNetCDF API calls and throw exceptions on error:

```cpp
// Before: Would call exit() on error (DANGEROUS!)
NC_SAFE_CALL( nc_open("file.nc", NC_NOWRITE, &ncid) );

// Now: Throws ftk::netcdf_error exception
try {
  NC_SAFE_CALL( nc_open("file.nc", NC_NOWRITE, &ncid) );
} catch (const ftk::netcdf_error& e) {
  std::cerr << "Failed to open NetCDF file: " << e.what() << std::endl;
  // Application continues running - can try alternative file
}
```

### Error Message Format

Exception messages include:
- **Error description** from NetCDF/PNetCDF library
- **File location** where error occurred
- **Line number** for debugging

Example:
```
[NDARRAY ERROR] NetCDF I/O error: NetCDF: I/O failure at ndarray.hh:1234
```

## Best Practices

### 1. Always Use try-catch with I/O Operations

```cpp
// Good: Proper error handling
try {
  arr.read_netcdf("data.nc", "temperature");
  process_data(arr);
} catch (const ftk::exception& e) {
  std::cerr << "Error: " << e.what() << std::endl;
  return false;
}

// Bad: No error handling - exception propagates to caller
arr.read_netcdf("data.nc", "temperature");  // May throw!
```

### 2. Clean Up Resources in catch Blocks

```cpp
int ncid = -1;
try {
  NC_SAFE_CALL( nc_open("file.nc", NC_NOWRITE, &ncid) );
  // ... operations ...
} catch (const ftk::netcdf_error& e) {
  if (ncid >= 0) {
    nc_close(ncid);  // Clean up
  }
  throw;  // Re-throw after cleanup
}
```

### 3. Use RAII for Automatic Cleanup

```cpp
class NetCDFFile {
  int ncid_;
public:
  NetCDFFile(const std::string& filename) {
    NC_SAFE_CALL( nc_open(filename.c_str(), NC_NOWRITE, &ncid_) );
  }
  ~NetCDFFile() {
    if (ncid_ >= 0) nc_close(ncid_);
  }
  int id() const { return ncid_; }
};

// Usage: Automatic cleanup even on exception
try {
  NetCDFFile file("data.nc");
  // ... operations ...
  // ncid automatically closed when file goes out of scope
} catch (const ftk::netcdf_error& e) {
  std::cerr << "Error: " << e.what() << std::endl;
}
```

### 4. Provide Context in Error Messages

```cpp
try {
  arr.read_netcdf(filename, varname);
} catch (const ftk::exception& e) {
  std::cerr << "Failed to read variable '" << varname
            << "' from file '" << filename << "': "
            << e.what() << std::endl;
}
```

### 5. Don't Catch Exceptions You Can't Handle

```cpp
// Bad: Catching and ignoring
try {
  arr.read_netcdf("data.nc", "temp");
} catch (...) {
  // Swallowing error - caller doesn't know it failed!
}

// Good: Let it propagate or handle properly
try {
  arr.read_netcdf("data.nc", "temp");
} catch (const ftk::netcdf_error& e) {
  std::cerr << "Error: " << e.what() << std::endl;
  throw;  // Re-throw for caller to handle
}
```

## MPI Considerations

When using MPI, ensure all ranks handle errors consistently:

```cpp
int local_error = 0;
try {
  arr.read_pnetcdf("data.nc", "temperature");
} catch (const ftk::exception& e) {
  std::cerr << "Rank " << rank << " error: " << e.what() << std::endl;
  local_error = 1;
}

// Check if any rank had an error
int global_error;
MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

if (global_error > 0) {
  // At least one rank failed
  MPI_Finalize();
  return EXIT_FAILURE;
}
```

## Migration from Old Code

### Before (Dangerous!)

```cpp
// Old code called exit() on error - killed entire application
NC_SAFE_CALL( nc_open("file.nc", NC_NOWRITE, &ncid) );
// If file doesn't exist: ENTIRE PROGRAM TERMINATES
```

### After (Safe!)

```cpp
// New code throws exception - application can recover
try {
  NC_SAFE_CALL( nc_open("file.nc", NC_NOWRITE, &ncid) );
} catch (const ftk::netcdf_error& e) {
  // Application continues - can try alternative file
  std::cerr << "Warning: " << e.what() << std::endl;
  std::cerr << "Trying alternative file..." << std::endl;
  NC_SAFE_CALL( nc_open("alternative.nc", NC_NOWRITE, &ncid) );
}
```

## Implementation Details

### Exception Construction

```cpp
// With error code
throw ftk::netcdf_error(ftk::ERR_NETCDF_IO, "Additional context");

// With message only
throw ftk::netcdf_error("Custom error message");

// Using fatal() helper
ftk::fatal(ftk::ERR_FILE_NOT_FOUND, "data.nc");
```

### Exception Methods

```cpp
catch (const ftk::exception& e) {
  e.what();         // Full error message with [NDARRAY ERROR] prefix
  e.error_code();   // Numeric error code (or 0 if none)
  e.message();      // User-provided context message
}
```

## Testing

Use `test_exception_handling` to verify exception behavior:

```bash
./bin/test_exception_handling
```

This test confirms:
- Exceptions are thrown (not `exit()`)
- Correct exception types
- Error codes are properly set
- Error messages include context

## See Also

- `include/ndarray/error.hh` - Exception definitions
- `include/ndarray/config.hh` - NC_SAFE_CALL and PNC_SAFE_CALL macros
- `tests/test_exception_handling.cpp` - Exception handling tests
