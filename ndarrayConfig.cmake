# get_filename_component(NDARRAY_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ndarrayConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
include("${CMAKE_CURRENT_LIST_DIR}/ndarrayTargets.cmake")

list (INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}")

set (NDARRAY_HAVE_ADIOS2 )
set (NDARRAY_HAVE_CUDA )
set (NDARRAY_HAVE_GMP )
set (NDARRAY_HAVE_HDF5 )
set (NDARRAY_HAVE_METIS )
set (NDARRAY_HAVE_MPI TRUE)
set (NDARRAY_HAVE_NETCDF )
set (NDARRAY_HAVE_VTK )

find_dependency (yaml-cpp REQUIRED)

if (NDARRAY_HAVE_ADIOS2)
  find_dependency (ADIOS2 REQUIRED)
endif ()

if (NDARRAY_HAVE_CUDA)
  enable_language (CUDA)
endif ()

if (NDARRAY_HAVE_VTK)
  find_dependency (VTK . REQUIRED)
endif ()

if (NDARRAY_HAVE_METIS)
  find_dependency (METIS REQUIRED)
endif ()

if (NDARRAY_HAVE_MPI)
  find_dependency (MPI REQUIRED)
  include_directories (${MPI_C_INCLUDE_PATH})
endif ()

if (NDARRAY_HAVE_HDF5)
  find_dependency (HDF5 REQUIRED)
  include_directories (${HDF5_INCLUDE_DIRS})
endif ()

if (NDARRAY_HAVE_NETCDF)
  find_dependency (netCDF REQUIRED)
  include_directories (${netCDF_INCLUDE_DIR})
endif ()

set (NDARRAY_INCLUDE_DIR "/usr/local/include")
include_directories (${NDARRAY_INCLUDE_DIR})

set (NDARRAY_FOUND 1)
set (NDARRAY_LIBRARY "")
