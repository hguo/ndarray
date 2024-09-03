add_library (ndarray SHARED ndarray.cpp)
target_link_libraries (ndarray yaml-cpp::yaml-cpp)

if (NDARRAY_HAVE_ADIOS2)
  target_link_libraries (ndarray adios2::adios2)
endif ()

if (NDARRAY_HAVE_VTK)
  target_link_libraries (ndarray VTK::CommonCore VTK::CommonDataModel VTK::IOLegacy VTK::IOXML VTK::IOParallelXML VTK::FiltersSources VTK::FiltersCore)
endif ()

if (NDARRAY_HAVE_NETCDF)
  target_link_libraries (ndarray netCDF::netcdf)
endif ()

if (NDARRAY_HAVE_PNETCDF)
  target_link_libraries (ndarray ${PNETCDF_LIBRARY})
endif ()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries (ndarray HDF5::HDF5)
endif ()

if (NDARRAY_HAVE_MPI)
  target_link_libraries (ndarray MPI::MPI_C MPI::MPI_CXX)
endif ()

if (NDARRAY_HAVE_PNG)
  target_link_libraries (ndarray PNG::PNG)
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
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/ndarray
  NAMESPACE ndarray::)
