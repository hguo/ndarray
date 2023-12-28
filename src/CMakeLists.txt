add_library (libndarray SHARED ndarray.cpp)
target_link_libraries (libndarray yaml-cpp::yaml-cpp)

if (NDARRAY_HAVE_ADIOS2)
  target_link_libraries (libndarray adios2::adios2)
endif ()

if (NDARRAY_HAVE_VTK)
  target_link_libraries (libndarray ${VTK_LIBRARIES})
endif ()

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries (libndarray netCDF::netcdf)
endif ()

if (NDARRAY_HAVE_PNETCDF)
  target_link_libraries (libndarray ${PNETCDF_LIBRARY})
endif ()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries (libndarray ${HDF5_LIBRARIES})
endif ()

if (NDARRAY_HAVE_MPI)
  # add_subdirectory (bil)
  target_link_libraries (libndarray ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif ()

if (NDARRAY_HAVE_PNG)
  target_link_libraries (libndarray ${PNG_LIBRARIES})
endif ()

if (NDARRAY_HAVE_OPENMP)
  target_link_libraries (libndarray OpenMP::OpenMP_CXX)
endif ()

if (NDARRAY_HAVE_CUDA)
  target_link_libraries (libndarray CUDA::cudart)
endif ()


add_executable (dryrun dryrun.cpp)
target_link_libraries (dryrun libndarray)
