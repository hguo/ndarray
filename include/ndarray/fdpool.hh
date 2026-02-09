#ifndef _NDARRAY_FDPOOL_H
#define _NDARRAY_FDPOOL_H

#include <ndarray/config.hh>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif
#if NDARRAY_HAVE_NETCDF
#include <netcdf.h>
#endif

#include <map>

namespace ftk {

struct fdpool_nc {
  fdpool_nc(fdpool_nc const&) = delete; // singleton
  void operator=(fdpool_nc const&) = delete;

  static fdpool_nc& get_instance() {
    static fdpool_nc instance; // singleton
    return instance;
  }

#if NDARRAY_HAVE_MPI
  int open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
#else
  int open(const std::string& filename);
#endif
  void close_all();

private:
  fdpool_nc() {} // singleton

  std::map<std::string, int> pool; // descriptors
};

////
#if NDARRAY_HAVE_MPI
inline int fdpool_nc::open(const std::string& f, MPI_Comm comm)
#else
inline int fdpool_nc::open(const std::string& f)
#endif
{
  auto it = pool.find(f);
  if (it == pool.end()) { // never opened
    int ncid;

#if NDARRAY_HAVE_NETCDF
#if NDARRAY_HAVE_MPI && NC_HAS_PARALLEL
    int rtn;
    rtn = nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
    if (rtn != NC_NOERR)
      NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#else
    NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#endif
#endif

    // fprintf(stderr, "[fdpool] opened netcdf file %s, ncid=%d.\n", f.c_str(), ncid);

    pool[f] = ncid;
    return ncid;
  } else
    return it->second;
}

inline void fdpool_nc::close_all()
{
  for (const auto &kv : pool) {
#if NDARRAY_HAVE_NETCDF
    // fprintf(stderr, "[fdpool] closing netcdf file %s, ncid=%d.\n", kv.first.c_str(), kv.second);
    
    NC_SAFE_CALL( nc_close(kv.second) );
#endif
  }

  pool.clear();
}

} // namespace

#endif
