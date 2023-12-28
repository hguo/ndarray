add_executable (dryrun dryrun.cpp)
target_link_libraries (dryrun yaml-cpp::yaml-cpp)

if (FTK_HAVE_ADIOS2)
  target_link_libraries (libndarray adios2::adios2)
endif ()

if (FTK_HAVE_VTK)
  target_link_libraries (libndarray ${VTK_LIBRARIES})
endif ()

if (FTK_HAVE_NETCDF)
  target_link_libraries (libndarray ${netCDF_LIBRARIES})
endif ()

if (FTK_HAVE_PNETCDF)
  target_link_libraries (libndarray ${PNETCDF_LIBRARY})
endif ()

if (FTK_HAVE_HDF5)
  target_link_libraries (libndarray ${HDF5_LIBRARIES})
endif ()

if (FTK_HAVE_MPI)
  add_subdirectory (bil)
  target_link_libraries (libndarray bil ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif ()

if (FTK_HAVE_PNG)
  target_link_libraries (libndarray ${PNG_LIBRARIES})
endif ()

if (FTK_HAVE_OPENMP)
  target_link_libraries (libndarray OpenMP::OpenMP_CXX)
endif ()

if (FTK_HAVE_CUDA)
  target_link_libraries (libndarray CUDA::cudart)
endif ()
