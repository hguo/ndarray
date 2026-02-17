#ifndef _NDARRAY_CONFIG_HH
#define _NDARRAY_CONFIG_HH

#define NDARRAY_VERSION ""

/* #undef NDARRAY_HAVE_ADIOS2 */
/* #undef NDARRAY_HAVE_CUDA */
/* #undef NDARRAY_HAVE_EIGEN */
/* #undef NDARRAY_HAVE_HDF5 */
/* #undef NDARRAY_HAVE_HIP */
#define NDARRAY_HAVE_MPI 1
/* #undef NDARRAY_HAVE_NETCDF */
/* #undef NDARRAY_HAVE_OPENMP */
/* #undef NDARRAY_HAVE_PNETCDF */
/* #undef NDARRAY_HAVE_PNG */
/* #undef NDARRAY_HAVE_VTK */
/* #undef NDARRAY_HAVE_XTENSOR */
/* #undef NDARRAY_HAVE_YAML */

/* #undef NDARRAY_USE_BIG_ENDIAN */
#define NDARRAY_USE_LITTLE_ENDIAN 1

#if NDARRAY_HAVE_MPI
#else
  typedef int MPI_Comm;
#define MPI_COMM_WORLD 0x44000000
#define MPI_COMM_SELF 0x44000001
#endif

#ifdef __CUDACC__
// #define NDARRAY_NUMERIC_FUNC __device__ __host__
#else
// #define NDARRAY_NUMERIC_FUNC
#define __device__
#define __host__
#endif

// Error handling macros for NetCDF/PNetCDF
// Note: These macros require <ndarray/error.hh> to be included first
// Most users should include <ndarray/ndarray.hh> which includes everything

#define NC_SAFE_CALL(call) \
  do { \
    int retval = (call); \
    if (retval != 0) { \
      std::string error_msg = std::string(nc_strerror(retval)) + \
                              " at " + __FILE__ + ":" + std::to_string(__LINE__); \
      throw ftk::nd::netcdf_error(ftk::nd::ERR_NETCDF_IO, error_msg); \
    } \
  } while (0)

#define PNC_SAFE_CALL(call) \
  do { \
    int retval = (call); \
    if (retval != 0) { \
      std::string error_msg = std::string(ncmpi_strerror(retval)) + \
                              " at " + __FILE__ + ":" + std::to_string(__LINE__); \
      throw ftk::nd::netcdf_error(ftk::nd::ERR_PNETCDF_IO, error_msg); \
    } \
  } while (0)

#endif
