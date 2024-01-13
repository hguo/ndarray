add_library (ndarray SHARED ndarray.cpp)
target_link_libraries (ndarray yaml-cpp::yaml-cpp)

if (NDARRAY_HAVE_ADIOS2)
  target_link_libraries (ndarray adios2::adios2)
endif ()

if (NDARRAY_HAVE_VTK)
  target_link_libraries (ndarray ${VTK_LIBRARIES})
endif ()

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries (ndarray netCDF::netcdf)
endif ()

if (NDARRAY_HAVE_PNETCDF)
  target_link_libraries (ndarray ${PNETCDF_LIBRARY})
endif ()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries (ndarray ${HDF5_LIBRARIES})
endif ()

if (NDARRAY_HAVE_MPI)
  # add_subdirectory (bil)
  target_link_libraries (ndarray ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif ()

if (NDARRAY_HAVE_PNG)
  target_link_libraries (ndarray ${PNG_LIBRARIES})
endif ()

if (NDARRAY_HAVE_OPENMP)
  target_link_libraries (ndarray OpenMP::OpenMP_CXX)
endif ()

if (NDARRAY_HAVE_CUDA)
  target_link_libraries (ndarray CUDA::cudart)
endif ()


add_executable (dryrun dryrun.cpp)
target_link_libraries (dryrun ndarray)


install (TARGETS ndarray
  EXPORT ndarrayTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install (EXPORT ndarrayTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndarray
  NAMESPACE ndarray::)
