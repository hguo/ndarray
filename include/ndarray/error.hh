#ifndef __NDARRAY_ERROR_HH
#define __NDARRAY_ERROR_HH

#include <ndarray/config.hh>
#include <string>
#include <iostream>
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
  case ERR_ADIOS2: return "adios2 error";
  case ERR_ADIOS2_VARIABLE_NOT_FOUND: return "adios2 variable not found";
  case ERR_MESH_UNSUPPORTED_FORMAT: return "unsupported mesh format";
  case ERR_MESH_NONSIMPLICIAL: return "unsupported nonsimplicial mesh";
  case ERR_MESH_EMPTY: return "empty mesh";
  default: return "unknown error: " + std::to_string(e);
  }
}

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

inline void fatal(int err, std::string str = "")
{
  std::cerr << "[NDARRAY FATAL] " << err2str(err);
  if (str.length()) std::cerr << ": " << str;
  std::cerr << std::endl;
  
  print_backtrace();
  exit(1);
}

inline void warn(int err, std::string str = "")
{
  std::cerr << "[NDARRAY WARN] " << err2str(err);
  if (str.length()) std::cerr << ": " << str;
  std::cerr << std::endl;
}

inline void fatal(const std::string& str) {
  std::cerr << "[NDARRAY FATAL] " << str << std::endl;
  
  print_backtrace();
  exit(1);
}

inline void warn(const std::string& str) {
  std::cerr << "[NDARRAY WARN] " << str << std::endl;
}

}

}

#endif
