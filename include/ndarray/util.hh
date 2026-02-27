#ifndef _NDARRAY_IO_UTIL_HH
#define _NDARRAY_IO_UTIL_HH

#include <ndarray/config.hh>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif
#if NDARRAY_HAVE_NETCDF
#include <ndarray/fdpool.hh>
#endif
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <regex>
#include <glob.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ftk {

enum {
  FILE_EXT_NULL = 0,
  FILE_EXT_BIN = 1,
  FILE_EXT_JSON,
  FILE_EXT_NETCDF,
  FILE_EXT_HDF5,
  FILE_EXT_BP,  // adios2
  FILE_EXT_NUMPY, // numpy
  FILE_EXT_PNG,
  FILE_EXT_VTI, // vtk xml image data
  FILE_EXT_VTP, // vtk xml poly data
  FILE_EXT_VTU, // vtk xml unstructured grid data
  FILE_EXT_PVTU, // vtk xml parallel unstructured grid data
  FILE_EXT_VTK, // legacy vtk format
  FILE_EXT_PLY, // surface
  FILE_EXT_STL  // surface
};

#if NDARRAY_HAVE_MPI
/**
 * @brief Initialize ndarray library (no-op currently)
 * @param comm MPI communicator (default: MPI_COMM_WORLD)
 * @note Reserved for future initialization needs
 */
static void ndarray_init(MPI_Comm = MPI_COMM_WORLD) { }
#else
/**
 * @brief Initialize ndarray library (no-op currently)
 * @note Reserved for future initialization needs
 */
static void ndarray_init() { }
#endif

/**
 * @brief Finalize ndarray library and clean up resources
 *
 * This function performs cleanup of global resources managed by ndarray:
 *
 * 1. NetCDF File Descriptor Pool (fdpool_nc):
 *    - Closes all open NetCDF files cached in the fdpool
 *    - Prevents file descriptor leaks
 *    - Required when using ndarray_group_stream with NetCDF files
 *
 * IMPORTANT: Call this function before program exit, especially when:
 * - Using ndarray_group_stream to read NetCDF files
 * - Opening multiple NetCDF files during program lifetime
 * - Running in MPI parallel mode
 *
 * Typical usage:
 * @code
 * int main(int argc, char** argv) {
 *   #if NDARRAY_HAVE_MPI
 *   MPI_Init(&argc, &argv);
 *   #endif
 *
 *   // ... ndarray operations ...
 *   // ... stream processing ...
 *
 *   // Cleanup before exit
 *   ftk::ndarray_finalize();  // Closes all NetCDF files in fdpool
 *
 *   #if NDARRAY_HAVE_MPI
 *   MPI_Finalize();
 *   #endif
 *
 *   return 0;
 * }
 * @endcode
 *
 * Thread Safety: NOT thread-safe. Call only from main thread after all
 * ndarray operations complete.
 *
 * @see fdpool_nc in fdpool.hh for file descriptor pool details
 */
static void ndarray_finalize()
{
#if NDARRAY_HAVE_NETCDF
  // Close all NetCDF files in the file descriptor pool
  // This prevents file descriptor leaks and ensures clean shutdown
  auto &ncpool = fdpool_nc::get_instance();
  ncpool.close_all();
#endif

  // Future: Add cleanup for other singleton resources here
}

static inline std::string series_filename(
    const std::string& pattern, int k)
{
  ssize_t size = snprintf(NULL, 0, pattern.c_str(), k);
  std::string filename(size + 1, '\0');
  snprintf(&filename[0], size + 1, pattern.c_str(), k);
  filename.resize(size);
  return filename;
}

static inline std::vector<std::string> glob(const std::string &pattern)
{
  std::vector<std::string> filenames;
  glob_t results;
  ::glob(pattern.c_str(), 0, NULL, &results);
  for (int i=0; i<results.gl_pathc; i++)
    filenames.push_back(results.gl_pathv[i]);
  globfree(&results);
  return filenames;
}

static bool is_directory(const std::string& filename) {
  struct stat s;
  if ( stat(filename.c_str(), &s) == 0 ) {
    if (s.st_mode & S_IFDIR) return true;
    else return false;
  } else return false;
}

static bool file_exists(const std::string& filename) {
  return access( filename.c_str(), F_OK ) == 0;
  // std::ifstream f(filename);
  // return f.good();
}

#if 0
static bool is_directory_all(const std::string& filename, diy::mpi::communicator comm = MPI_COMM_WORLD)
{
  bool b = false;
  if (comm.rank() == 0)
    b = is_directory(filename);
  diy::mpi::all_reduce(comm, b, b, std::logical_or<bool>());
  return b;
}

static bool file_exists_all(const std::string& filename, diy::mpi::communicator comm = MPI_COMM_WORLD)
{
  bool b = false;
  if (comm.rank() == 0)
    b = file_exists(filename);
  diy::mpi::all_reduce(comm, b, b, std::logical_or<bool>());
  return b;
}

static bool file_not_exists_all(const std::string& filename, diy::mpi::communicator comm = MPI_COMM_WORLD)
{
  return !file_exists_all(filename, comm);
}
#endif

static bool file_not_exists(const std::string& filename) {
  return !file_exists(filename);
}

static std::string remove_file_extension(const std::string& f)
{
  size_t lastindex = f.find_last_of(".");
  return f.substr(0, lastindex);
}

static inline bool starts_with(std::string const & value, std::string const & starting)
{
  if (value.find(starting) == 0) return true;
  else return false;
}

static inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static inline std::string to_lower_cases(std::string str)
{
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
      return std::tolower(c);
  });
  return str;
}

static inline bool ends_with_lower(const std::string& str, const std::string& ending)
{
  return ends_with(to_lower_cases(str), to_lower_cases(ending));
}

// https://stackoverflow.com/questions/9435385/split-a-string-using-c11
static inline std::vector<std::string> split(const std::string& input, const std::string& regex) {
    // passing -1 as the submatch index parameter performs splitting
    std::regex re(regex);
    std::sregex_token_iterator
        first{input.begin(), input.end(), re, -1},
        last;
    return {first, last};
}

static inline int file_extension(const std::string& f)
{
  auto m = [f](std::string e) { return ends_with_lower(f, e); };

  if (m("bin") || m("binary"))
    return FILE_EXT_BIN;
  else if (m("nc") || m("netcdf"))
    return FILE_EXT_NETCDF;
  else if (m("h5") || m("hdf5"))
    return FILE_EXT_HDF5;
  else if (m("bp") || m("adios2"))
    return FILE_EXT_BP;
  else if (m("npy") || m("numpy"))
    return FILE_EXT_NUMPY;
  else if (m("png"))
    return FILE_EXT_PNG;
  else if (m("vti"))
    return FILE_EXT_VTI;
  else if (m("vtp"))
    return FILE_EXT_VTP;
  else if (m("pvtu")) // must check pvtu before vtu
    return FILE_EXT_PVTU;
  else if (m("vtu"))
    return FILE_EXT_VTU;
  else if (m("vtk"))
    return FILE_EXT_VTK;
  else if (m("ply"))
    return FILE_EXT_PLY;
  else if (m("stl"))
    return FILE_EXT_STL;
  else
    return FILE_EXT_NULL;
}

static inline int file_extension(const std::string& filename, const std::string& format)
{
  if (format == "auto" || format.empty())
    return file_extension(filename);
  else
    return file_extension(format);
}

} // namespace ftk

#endif
