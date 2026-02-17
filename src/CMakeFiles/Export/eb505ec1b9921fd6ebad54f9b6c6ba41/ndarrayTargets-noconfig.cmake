#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ndarray::ndarray" for configuration ""
set_property(TARGET ndarray::ndarray APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ndarray::ndarray PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libndarray.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libndarray.dylib"
  )

list(APPEND _cmake_import_check_targets ndarray::ndarray )
list(APPEND _cmake_import_check_files_for_ndarray::ndarray "${_IMPORT_PREFIX}/lib/libndarray.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
