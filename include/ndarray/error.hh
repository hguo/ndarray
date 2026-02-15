#ifndef __NDARRAY_ERROR_HH
#define __NDARRAY_ERROR_HH

#include <ndarray/config.hh>
#include <string>
#include <iostream>
#include <exception>
#include <execinfo.h>

namespace ftk {

namespace nd {

enum {
  ERR_NOT_IMPLEMENTED = 1,
  ERR_FILE_NOT_FOUND = 1000,
  ERR_FILE_CANNOT_OPEN,
  ERR_FILE_CANNOT_WRITE,
  ERR_FILE_CANNOT_READ_EXPECTED_BYTES,
  ERR_FILE_UNRECOGNIZED_EXTENSION,
  ERR_FILE_FORMAT,
  ERR_FILE_FORMAT_AMIRA,
  ERR_NOT_BUILT_WITH_ADIOS2 = 2000,
  ERR_NOT_BUILT_WITH_ADIOS1,
  ERR_NOT_BUILT_WITH_BOOST,
  ERR_NOT_BUILT_WITH_CGAL,
  ERR_NOT_BUILT_WITH_CUDA,
  ERR_NOT_BUILT_WITH_GMP,
  ERR_NOT_BUILT_WITH_HDF5,
  ERR_NOT_BUILT_WITH_HIPSYCL,
  ERR_NOT_BUILT_WITH_SYCL,
  ERR_NOT_BUILT_WITH_KOKKOS,
  ERR_NOT_BUILT_WITH_LEVELDB,
  ERR_NOT_BUILT_WITH_METIS,
  ERR_NOT_BUILT_WITH_MPI,
  ERR_NOT_BUILT_WITH_MPSOLVE,
  ERR_NOT_BUILT_WITH_NETCDF,
  ERR_NOT_BUILT_WITH_OPENMP,
  ERR_NOT_BUILT_WITH_PARAVIEW,
  ERR_NOT_BUILT_WITH_PNETCDF,
  ERR_NOT_BUILT_WITH_PNG,
  ERR_NOT_BUILT_WITH_PYBIND11,
  ERR_NOT_BUILT_WITH_ROCKSDB,
  ERR_NOT_BUILT_WITH_QT5,
  ERR_NOT_BUILT_WITH_QT,
  ERR_NOT_BUILT_WITH_TBB,
  ERR_NOT_BUILT_WITH_VTK,
  ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS = 3000, // only support one dim for components
  ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY,
  ERR_NDARRAY_RESHAPE_EMPTY,
  ERR_NDARRAY_RESHAPE_DEVICE,
  ERR_NDARRAY_UNKNOWN_DEVICE,
  ERR_ACCELERATOR_UNSUPPORTED = 4000,
  ERR_THREAD_BACKEND_UNSUPPORTED = 5000,
  ERR_VTK_VARIABLE_NOT_FOUND = 6000,
  ERR_VTK_UNSUPPORTED_OUTPUT_FORMAT,
  ERR_NETCDF_MISSING_VARIABLE = 6500,
  ERR_NETCDF_FILE_NOT_OPEN,
  ERR_NETCDF_IO,              // NetCDF I/O operation failed
  ERR_PNETCDF_IO,             // Parallel-NetCDF I/O operation failed
  ERR_ADIOS2 = 7000,
  ERR_ADIOS2_VARIABLE_NOT_FOUND,
  ERR_MESH_UNSUPPORTED_FORMAT = 8000,
  ERR_MESH_NONSIMPLICIAL,
  ERR_MESH_EMPTY,
  ERR_STREAM_FORMAT = 9000,
  ERR_UNKNOWN_OPTIONS = 10000
};

inline std::string err2str(int e)
{
  switch (e) {
  case ERR_NOT_IMPLEMENTED: return "not implemented yet";
  case ERR_FILE_NOT_FOUND: return "file not found";
  case ERR_FILE_CANNOT_OPEN: return "cannot open file";
  case ERR_FILE_CANNOT_WRITE: return "cannot write file";
  case ERR_FILE_CANNOT_READ_EXPECTED_BYTES: return "cannot read expected number of bytes";
  case ERR_FILE_FORMAT: return "file format error";
  case ERR_FILE_FORMAT_AMIRA: return "file format error with AmiraMesh data";
  case ERR_FILE_UNRECOGNIZED_EXTENSION: return "unrecognized file extension";
  case ERR_NOT_BUILT_WITH_ADIOS2: return "ndarray not compiled with ADIOS2";
  case ERR_NOT_BUILT_WITH_ADIOS1: return "ndarray not compiled with ADIOS1";
  case ERR_NOT_BUILT_WITH_BOOST: return "ndarray not compiled with Boost";
  case ERR_NOT_BUILT_WITH_CGAL: return "ndarray not compiled with CGAL";
  case ERR_NOT_BUILT_WITH_CUDA: return "ndarray not compiled with CUDA";
  case ERR_NOT_BUILT_WITH_GMP: return "ndarray not compiled with GMP";
  case ERR_NOT_BUILT_WITH_HDF5: return "ndarray not compiled with HDF5";
  case ERR_NOT_BUILT_WITH_HIPSYCL: return "ndarray not compiled with hipSYCL";
  case ERR_NOT_BUILT_WITH_SYCL: return "ndarray not compiled with SYCL";
  case ERR_NOT_BUILT_WITH_KOKKOS: return "ndarray not compiled with Kokkos";
  case ERR_NOT_BUILT_WITH_LEVELDB: return "ndarray not compiled with LevelDB";
  case ERR_NOT_BUILT_WITH_METIS: return "ndarray not compiled with Metis";
  case ERR_NOT_BUILT_WITH_MPI: return "ndarray not compiled with MPI";
  case ERR_NOT_BUILT_WITH_MPSOLVE: return "ndarray not compiled with MPSolve";
  case ERR_NOT_BUILT_WITH_NETCDF: return "ndarray not compiled with NetCDF";
  case ERR_NOT_BUILT_WITH_OPENMP: return "ndarray not compiled with OpenMP";
  case ERR_NOT_BUILT_WITH_PARAVIEW: return "ndarray not compiled with ParaView";
  case ERR_NOT_BUILT_WITH_PNETCDF: return "ndarray not compiled with Parallel-NetCDF";
  case ERR_NOT_BUILT_WITH_PNG: return "ndarray not compiled with PNG";
  case ERR_NOT_BUILT_WITH_PYBIND11: return "ndarray not compiled with PyBind11";
  case ERR_NOT_BUILT_WITH_ROCKSDB: return "ndarray not compiled with RocksDB";
  case ERR_NOT_BUILT_WITH_QT5: return "ndarray not compiled with Qt5";
  case ERR_NOT_BUILT_WITH_QT: return "ndarray not compiled with Qt";
  case ERR_NOT_BUILT_WITH_TBB: return "ndarray not compiled with TBB";
  case ERR_NOT_BUILT_WITH_VTK: return "ndarray not compiled with VTK";
  case ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS: return "ndarray only supports one dim for components";
  case ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY: return "unsupported data dimensionality";
  case ERR_NDARRAY_RESHAPE_EMPTY: return "unable to reshape empty array";
  case ERR_NDARRAY_RESHAPE_DEVICE: return "reshaping a device ndarray is not supported";
  case ERR_NDARRAY_UNKNOWN_DEVICE: return "unknown device for ndarray";
  case ERR_ACCELERATOR_UNSUPPORTED: return "unsupported accelerator";
  case ERR_THREAD_BACKEND_UNSUPPORTED: return "unsupported thread backend";
  case ERR_VTK_VARIABLE_NOT_FOUND: return "VTK variable not found";
  case ERR_VTK_UNSUPPORTED_OUTPUT_FORMAT: return "unsupported vtk output format";
  case ERR_NETCDF_MISSING_VARIABLE: return "missing netcdf variable name(s)";
  case ERR_NETCDF_IO: return "NetCDF I/O error";
  case ERR_PNETCDF_IO: return "Parallel-NetCDF I/O error";
  case ERR_ADIOS2: return "adios2 error";
  case ERR_ADIOS2_VARIABLE_NOT_FOUND: return "adios2 variable not found";
  case ERR_MESH_UNSUPPORTED_FORMAT: return "unsupported mesh format";
  case ERR_MESH_NONSIMPLICIAL: return "unsupported nonsimplicial mesh";
  case ERR_MESH_EMPTY: return "empty mesh";
  case ERR_STREAM_FORMAT: return "wrong stream format";
  default: return "unknown error: " + std::to_string(e);
  }
}

////
// Exception Classes
////

/**
 * @brief Base exception class for all ndarray errors
 *
 * All ndarray exceptions derive from this class, which itself derives
 * from std::exception. This allows catching all ndarray errors with:
 *
 * @code
 * try {
 *   arr.read_netcdf("file.nc", "var");
 * } catch (const ftk::nd::exception& e) {
 *   std::cerr << "ndarray error: " << e.what() << std::endl;
 *   std::cerr << "Error code: " << e.error_code() << std::endl;
 * }
 * @endcode
 *
 * @see error codes defined above
 */
class exception : public std::exception {
public:
  /**
   * @brief Construct exception with error code and optional message
   * @param code Error code from enum above
   * @param msg Additional context message (optional)
   */
  explicit exception(int code, const std::string& msg = "")
    : err_code(code), user_msg(msg)
  {
    full_msg = "[NDARRAY ERROR] " + err2str(code);
    if (!msg.empty()) {
      full_msg += ": " + msg;
    }
  }

  /**
   * @brief Construct exception with message only (no error code)
   * @param msg Error message
   */
  explicit exception(const std::string& msg)
    : err_code(0), user_msg(msg)
  {
    full_msg = "[NDARRAY ERROR] " + msg;
  }

  /**
   * @brief Get full error message
   * @return C-string with complete error description
   */
  virtual const char* what() const noexcept override {
    return full_msg.c_str();
  }

  /**
   * @brief Get error code
   * @return Error code from enum, or 0 if constructed with message only
   */
  int error_code() const noexcept {
    return err_code;
  }

  /**
   * @brief Get user-provided context message
   * @return Additional message provided when exception was thrown
   */
  const std::string& message() const noexcept {
    return user_msg;
  }

protected:
  int err_code;
  std::string user_msg;
  std::string full_msg;
};

/**
 * @brief Exception for file I/O errors
 *
 * Thrown when file operations fail (open, read, write, format errors).
 */
class file_error : public exception {
public:
  explicit file_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit file_error(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for missing library/feature
 *
 * Thrown when attempting to use a feature that was not compiled in.
 */
class feature_not_available : public exception {
public:
  explicit feature_not_available(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit feature_not_available(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for invalid array operations
 *
 * Thrown for invalid array operations like reshaping empty arrays.
 */
class invalid_operation : public exception {
public:
  explicit invalid_operation(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit invalid_operation(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for not-yet-implemented features
 */
class not_implemented : public exception {
public:
  explicit not_implemented(const std::string& msg = "")
    : exception(ERR_NOT_IMPLEMENTED, msg) {}
};

/**
 * @brief Exception for device/accelerator errors
 */
class device_error : public exception {
public:
  explicit device_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit device_error(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for VTK-related errors
 */
class vtk_error : public exception {
public:
  explicit vtk_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit vtk_error(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for NetCDF-related errors
 */
class netcdf_error : public exception {
public:
  explicit netcdf_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit netcdf_error(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for ADIOS2-related errors
 */
class adios2_error : public exception {
public:
  explicit adios2_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit adios2_error(const std::string& msg)
    : exception(msg) {}
};

/**
 * @brief Exception for stream-related errors
 */
class stream_error : public exception {
public:
  explicit stream_error(int code, const std::string& msg = "")
    : exception(code, msg) {}
  explicit stream_error(const std::string& msg)
    : exception(msg) {}
};

inline void print_backtrace()
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  printf ("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
    printf ("%s\n", strings[i]);

  free (strings);
}

/**
 * @brief Throw appropriate exception based on error code
 *
 * This function examines the error code and throws the most specific
 * exception type available. Falls back to generic exception if no
 * specific type matches.
 *
 * @param err Error code from enum
 * @param str Optional context message
 * @throws Specific exception subclass based on error code
 */
[[noreturn]] inline void fatal(int err, const std::string& str = "")
{
  // Determine which exception type to throw based on error code

  // File errors
  if (err >= ERR_FILE_NOT_FOUND && err < ERR_NOT_BUILT_WITH_ADIOS2) {
    throw file_error(err, str);
  }

  // Feature not available errors
  else if (err >= ERR_NOT_BUILT_WITH_ADIOS2 && err < ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS) {
    throw feature_not_available(err, str);
  }

  // Array operation errors
  else if (err >= ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS && err < ERR_ACCELERATOR_UNSUPPORTED) {
    throw invalid_operation(err, str);
  }

  // Device/accelerator errors
  else if (err >= ERR_ACCELERATOR_UNSUPPORTED && err < ERR_THREAD_BACKEND_UNSUPPORTED) {
    throw device_error(err, str);
  }

  // Thread backend errors (treat as device errors)
  else if (err >= ERR_THREAD_BACKEND_UNSUPPORTED && err < ERR_VTK_VARIABLE_NOT_FOUND) {
    throw device_error(err, str);
  }

  // VTK errors
  else if (err >= ERR_VTK_VARIABLE_NOT_FOUND && err < ERR_NETCDF_MISSING_VARIABLE) {
    throw vtk_error(err, str);
  }

  // NetCDF errors
  else if (err >= ERR_NETCDF_MISSING_VARIABLE && err < ERR_ADIOS2) {
    throw netcdf_error(err, str);
  }

  // ADIOS2 errors
  else if (err >= ERR_ADIOS2 && err < ERR_MESH_UNSUPPORTED_FORMAT) {
    throw adios2_error(err, str);
  }

  // Stream errors
  else if (err >= ERR_STREAM_FORMAT && err < ERR_UNKNOWN_OPTIONS) {
    throw stream_error(err, str);
  }

  // Not implemented
  else if (err == ERR_NOT_IMPLEMENTED) {
    throw not_implemented(str);
  }

  // Generic fallback
  else {
    throw exception(err, str);
  }
}

/**
 * @brief Throw exception with custom message (no error code)
 *
 * @param str Error message
 * @throws exception Generic exception with provided message
 */
[[noreturn]] inline void fatal(const std::string& str) {
  throw exception(str);
}

/**
 * @brief Print warning message (non-fatal)
 *
 * Warnings are printed to stderr but do not throw exceptions.
 *
 * @param err Error code from enum
 * @param str Optional context message
 */
inline void warn(int err, const std::string& str = "")
{
  std::cerr << "[NDARRAY WARN] " << err2str(err);
  if (str.length()) std::cerr << ": " << str;
  std::cerr << std::endl;
}

/**
 * @brief Print warning message (non-fatal)
 *
 * @param str Warning message
 */
inline void warn(const std::string& str) {
  std::cerr << "[NDARRAY WARN] " << str << std::endl;
}

}

}

#endif
