add_executable (dryrun dryrun.cpp)

if (FTK_HAVE_ADIOS2)
  target_link_libraries (libftk adios2::adios2)
endif ()

if (FTK_HAVE_VTK)
  target_link_libraries (libftk ${VTK_LIBRARIES})
endif ()

if (FTK_HAVE_NETCDF)
  target_link_libraries (libftk ${netCDF_LIBRARIES})
endif ()

if (FTK_HAVE_PNETCDF)
  target_link_libraries (libftk ${PNETCDF_LIBRARY})
endif ()

if (FTK_HAVE_HDF5)
  target_link_libraries (libftk ${HDF5_LIBRARIES})
endif ()

if (FTK_HAVE_MPI)
  add_subdirectory (bil)
  target_link_libraries (libftk bil ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif ()

if (FTK_HAVE_PNG)
  target_link_libraries (libftk ${PNG_LIBRARIES})
endif ()

if (FTK_HAVE_OPENMP)
  target_link_libraries (libftk OpenMP::OpenMP_CXX)
endif ()

if (FTK_HAVE_CUDA)
  target_link_libraries (libftk CUDA::cudart)
endif ()
