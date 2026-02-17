if (NDARRAY_HAVE_HENSON)
  if                          (NOT APPLE)
     set                      (linker_flags "-pie -Wl,--export-dynamic")
     set                      (linker_flags "${linker_flags} -Wl,-u,henson_set_contexts,-u,henson_set_namemap")
  else                        ()
     set                      (linker_flags "-Wl,-u,_henson_set_contexts,-u,_henson_set_namemap")
  endif                       (NOT APPLE)
endif ()

add_library (ndarray SHARED ndarray.cpp)

# Link all dependencies with PUBLIC visibility so they propagate to consumers
# This allows examples/tests to simply link against ndarray without manually
# specifying each optional dependency

if (NDARRAY_HAVE_YAML)
  target_link_libraries (ndarray PUBLIC yaml-cpp::yaml-cpp)
endif ()

if (NDARRAY_HAVE_ADIOS2)
  target_link_libraries (ndarray PUBLIC adios2::adios2)
endif ()

if (NDARRAY_HAVE_VTK)
  # Core modules (always required)
  target_link_libraries (ndarray PUBLIC VTK::CommonCore VTK::CommonDataModel VTK::IOLegacy VTK::IOXML)

  # Optional modules - only link if available
  if (TARGET VTK::IOParallelXML)
    target_link_libraries (ndarray PUBLIC VTK::IOParallelXML)
  endif ()
  if (TARGET VTK::FiltersSources)
    target_link_libraries (ndarray PUBLIC VTK::FiltersSources)
  endif ()
  if (TARGET VTK::FiltersCore)
    target_link_libraries (ndarray PUBLIC VTK::FiltersCore)
  endif ()
endif ()

if (NDARRAY_HAVE_NETCDF)
  # Use target if available (CMake build), otherwise use libraries directly (autotools build)
  if (TARGET netCDF::netcdf)
    target_link_libraries (ndarray PUBLIC netCDF::netcdf)
  else()
    target_link_libraries (ndarray PUBLIC ${netCDF_LIBRARIES})
    if (netCDF_LIBRARY_DIRS)
      target_link_directories (ndarray PUBLIC ${netCDF_LIBRARY_DIRS})
    endif()
  endif()
endif ()

if (NDARRAY_HAVE_PNETCDF)
  target_link_libraries (ndarray PUBLIC ${PNETCDF_LIBRARIES})
  if (PNETCDF_LIBRARY_DIRS)
    target_link_directories (ndarray PUBLIC ${PNETCDF_LIBRARY_DIRS})
  endif()
endif ()

if (NDARRAY_HAVE_HDF5)
  target_link_libraries (ndarray PUBLIC ${HDF5_LIBRARIES})
endif ()

if (NDARRAY_HAVE_HENSON)
  target_link_libraries (ndarray PUBLIC ${HENSON_PMPI_LIBRARY} ${HENSON_LIBRARY})
endif ()

if (NDARRAY_HAVE_MPI AND TARGET MPI::MPI_CXX)
  target_link_libraries (ndarray PUBLIC MPI::MPI_CXX)
  if (TARGET MPI::MPI_C)
    target_link_libraries (ndarray PUBLIC MPI::MPI_C)
  endif ()
endif ()

if (NDARRAY_HAVE_PNG)
  target_link_libraries (ndarray PUBLIC PNG::PNG)
endif ()

if (NDARRAY_HAVE_OPENMP)
  target_link_libraries (ndarray PUBLIC OpenMP::OpenMP_CXX)
endif ()

if (NDARRAY_HAVE_CUDA)
  target_link_libraries (ndarray PUBLIC CUDA::cudart)
endif ()


if (NDARRAY_HAVE_YAML)
  add_executable (ndarray-dryrun dryrun.cpp)
  target_link_libraries (ndarray-dryrun PRIVATE ndarray)
endif ()


if (NDARRAY_HAVE_YAML)
  install (TARGETS ndarray ndarray-dryrun
    EXPORT ndarrayTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
else ()
  install (TARGETS ndarray
    EXPORT ndarrayTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif ()

install (EXPORT ndarrayTargets
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/ndarray
  NAMESPACE ndarray::)
