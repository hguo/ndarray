#ifndef __NDARRAY_ERROR_HH
#define __NDARRAY_ERROR_HH

#include <ndarray/config.hh>
#include <execinfo.h>

namespace ndarray {

enum {
  NDARRAY_ERR_NOT_IMPLEMENTED = 1,
  NDARRAY_ERR_FILE_NOT_FOUND = 1000,
  NDARRAY_ERR_FILE_CANNOT_OPEN,
  NDARRAY_ERR_FILE_CANNOT_WRITE,
  NDARRAY_ERR_FILE_CANNOT_READ_EXPECTED_BYTES,
  NDARRAY_ERR_FILE_UNRECOGNIZED_EXTENSION,
  NDARRAY_ERR_FILE_FORMAT,
  NDARRAY_ERR_FILE_FORMAT_AMIRA,
  NDARRAY_ERR_NOT_BUILT_WITH_ADIOS2 = 2000,
  NDARRAY_ERR_NOT_BUILT_WITH_ADIOS1,
  NDARRAY_ERR_NOT_BUILT_WITH_BOOST,
  NDARRAY_ERR_NOT_BUILT_WITH_CGAL,
  NDARRAY_ERR_NOT_BUILT_WITH_CUDA,
  NDARRAY_ERR_NOT_BUILT_WITH_GMP,
  NDARRAY_ERR_NOT_BUILT_WITH_HDF5,
  NDARRAY_ERR_NOT_BUILT_WITH_HIPSYCL,
  NDARRAY_ERR_NOT_BUILT_WITH_SYCL,
  NDARRAY_ERR_NOT_BUILT_WITH_KOKKOS,
  NDARRAY_ERR_NOT_BUILT_WITH_LEVELDB,
  NDARRAY_ERR_NOT_BUILT_WITH_METIS,
  NDARRAY_ERR_NOT_BUILT_WITH_MPI,
  NDARRAY_ERR_NOT_BUILT_WITH_MPSOLVE,
  NDARRAY_ERR_NOT_BUILT_WITH_NETCDF,
  NDARRAY_ERR_NOT_BUILT_WITH_OPENMP,
  NDARRAY_ERR_NOT_BUILT_WITH_PARAVIEW,
  NDARRAY_ERR_NOT_BUILT_WITH_PNETCDF,
  NDARRAY_ERR_NOT_BUILT_WITH_PNG,
  NDARRAY_ERR_NOT_BUILT_WITH_PYBIND11,
  NDARRAY_ERR_NOT_BUILT_WITH_ROCKSDB,
  NDARRAY_ERR_NOT_BUILT_WITH_QT5,
  NDARRAY_ERR_NOT_BUILT_WITH_QT,
  NDARRAY_ERR_NOT_BUILT_WITH_TBB,
  NDARRAY_ERR_NOT_BUILT_WITH_VTK,
  NDARRAY_ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS = 3000, // only support one dim for components
  NDARRAY_ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY,
  NDARRAY_ERR_NDARRAY_RESHAPE_EMPTY,
  NDARRAY_ERR_NDARRAY_RESHAPE_DEVICE,
  NDARRAY_ERR_NDARRAY_UNKNOWN_DEVICE,
  NDARRAY_ERR_ACCELERATOR_UNSUPPORTED = 4000,
  NDARRAY_ERR_THREAD_BACKEND_UNSUPPORTED = 5000,
  NDARRAY_ERR_VTK_VARIABLE_NOT_FOUND = 6000,
  NDARRAY_ERR_VTK_UNSUPPORTED_OUTPUT_FORMAT,
  NDARRAY_ERR_NETCDF_MISSING_VARIABLE = 6500,
  NDARRAY_ERR_NETCDF_FILE_NOT_OPEN,
  NDARRAY_ERR_ADIOS2 = 7000,
  NDARRAY_ERR_ADIOS2_VARIABLE_NOT_FOUND,
  NDARRAY_ERR_MESH_UNSUPPORTED_FORMAT = 8000,
  NDARRAY_ERR_MESH_NONSIMPLICIAL, 
  NDARRAY_ERR_MESH_EMPTY,
  NDARRAY_ERR_UNKNOWN_OPTIONS = 10000
};

inline std::string err2str(int e)
{
  switch (e) {
  case NDARRAY_ERR_NOT_IMPLEMENTED: return "not implemented yet";
  case NDARRAY_ERR_FILE_NOT_FOUND: return "file not found";
  case NDARRAY_ERR_FILE_CANNOT_OPEN: return "cannot open file";
  case NDARRAY_ERR_FILE_CANNOT_WRITE: return "cannot write file";
  case NDARRAY_ERR_FILE_CANNOT_READ_EXPECTED_BYTES: return "cannot read expected number of bytes";
  case NDARRAY_ERR_FILE_FORMAT: return "file format error";
  case NDARRAY_ERR_FILE_FORMAT_AMIRA: return "file format error with AmiraMesh data";
  case NDARRAY_ERR_FILE_UNRECOGNIZED_EXTENSION: return "unrecognized file extension";
  case NDARRAY_ERR_NOT_BUILT_WITH_ADIOS2: return "FTK not compiled with ADIOS2";
  case NDARRAY_ERR_NOT_BUILT_WITH_ADIOS1: return "FTK not compiled with ADIOS1";
  case NDARRAY_ERR_NOT_BUILT_WITH_BOOST: return "FTK not compiled with Boost";
  case NDARRAY_ERR_NOT_BUILT_WITH_CGAL: return "FTK not compiled with CGAL";
  case NDARRAY_ERR_NOT_BUILT_WITH_CUDA: return "FTK not compiled with CUDA";
  case NDARRAY_ERR_NOT_BUILT_WITH_GMP: return "FTK not compiled with GMP";
  case NDARRAY_ERR_NOT_BUILT_WITH_HDF5: return "FTK not compiled with HDF5";
  case NDARRAY_ERR_NOT_BUILT_WITH_HIPSYCL: return "FTK not compiled with hipSYCL";
  case NDARRAY_ERR_NOT_BUILT_WITH_KOKKOS: return "FTK not compiled with Kokkos";
  case NDARRAY_ERR_NOT_BUILT_WITH_LEVELDB: return "FTK not compiled with LevelDB";
  case NDARRAY_ERR_NOT_BUILT_WITH_METIS: return "FTK not compiled with Metis";
  case NDARRAY_ERR_NOT_BUILT_WITH_MPI: return "FTK not compiled with MPI";
  case NDARRAY_ERR_NOT_BUILT_WITH_MPSOLVE: return "FTK not compiled with MPSolve";
  case NDARRAY_ERR_NOT_BUILT_WITH_NETCDF: return "FTK not compiled with NetCDF";
  case NDARRAY_ERR_NOT_BUILT_WITH_OPENMP: return "FTK not compiled with OpenMP";
  case NDARRAY_ERR_NOT_BUILT_WITH_PARAVIEW: return "FTK not compiled with ParaView";
  case NDARRAY_ERR_NOT_BUILT_WITH_PNETCDF: return "FTK not compiled with Parallel-NetCDF";
  case NDARRAY_ERR_NOT_BUILT_WITH_PNG: return "FTK not compiled with PNG";
  case NDARRAY_ERR_NOT_BUILT_WITH_PYBIND11: return "FTK not compiled with PyBind11";
  case NDARRAY_ERR_NOT_BUILT_WITH_ROCKSDB: return "FTK not compiled with RocksDB";
  case NDARRAY_ERR_NOT_BUILT_WITH_QT5: return "FTK not compiled with Qt5";
  case NDARRAY_ERR_NOT_BUILT_WITH_QT: return "FTK not compiled with Qt";
  case NDARRAY_ERR_NOT_BUILT_WITH_TBB: return "FTK not compiled with TBB";
  case NDARRAY_ERR_NOT_BUILT_WITH_VTK: return "FTK not compiled with VTK";
  case NDARRAY_ERR_NDARRAY_MULTIDIMENSIONAL_COMPONENTS: return "FTK only supports one dim for components";
  case NDARRAY_ERR_NDARRAY_UNSUPPORTED_DIMENSIONALITY: return "unsupported data dimensionality";
  case NDARRAY_ERR_NDARRAY_RESHAPE_EMPTY: return "unable to reshape empty array";
  case NDARRAY_ERR_NDARRAY_RESHAPE_DEVICE: return "reshaping a device ndarray is not supported";
  case NDARRAY_ERR_NDARRAY_UNKNOWN_DEVICE: return "unknown device for ndarray";
  case NDARRAY_ERR_ACCELERATOR_UNSUPPORTED: return "unsupported accelerator";
  case NDARRAY_ERR_THREAD_BACKEND_UNSUPPORTED: return "unsupported thread backend";
  case NDARRAY_ERR_VTK_VARIABLE_NOT_FOUND: return "VTK variable not found";
  case NDARRAY_ERR_VTK_UNSUPPORTED_OUTPUT_FORMAT: return "unsupported vtk output format";
  case NDARRAY_ERR_NETCDF_MISSING_VARIABLE: return "missing netcdf variable name(s)";
  case NDARRAY_ERR_ADIOS2: return "adios2 error";
  case NDARRAY_ERR_ADIOS2_VARIABLE_NOT_FOUND: return "adios2 variable not found";
  case NDARRAY_ERR_MESH_UNSUPPORTED_FORMAT: return "unsupported mesh format";
  case NDARRAY_ERR_MESH_NONSIMPLICIAL: return "unsupported nonsimplicial mesh";
  case NDARRAY_ERR_MESH_EMPTY: return "empty mesh";
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

#endif
