if (NDARRAY_HAVE_HENSON)
  if                          (NOT APPLE)
     set                      (linker_flags "-pie -Wl,--export-dynamic")
     set                      (linker_flags "${linker_flags} -Wl,-u,henson_set_contexts,-u,henson_set_namemap")
  else                        ()
     set                      (linker_flags "-Wl,-u,_henson_set_contexts,-u,_henson_set_namemap")
  endif                       (NOT APPLE)
endif ()

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

if (NDARRAY_HAVE_HENSON)
  target_link_libraries (ndarray ${HENSON_PMPI_LIBRARY} ${HENSON_LIBRARY})
endif ()

if (NDARRAY_HAVE_MPI AND TARGET MPI::MPI_CXX)
  target_link_libraries (ndarray MPI::MPI_CXX)
  if (TARGET MPI::MPI_C)
    target_link_libraries (ndarray MPI::MPI_C)
  endif ()
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


add_executable (ndarray-dryrun dryrun.cpp)
target_link_libraries (ndarray-dryrun PRIVATE ndarray)


install (TARGETS ndarray ndarray-dryrun
  EXPORT ndarrayTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install (EXPORT ndarrayTargets
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/ndarray
  NAMESPACE ndarray::)
