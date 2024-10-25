#ifndef _NDARRAY_FDPOOL_H
#define _NDARRAY_FDPOOL_H

#include <ndarray/config.hh>

namespace ftk {

struct fdpool_nc {
  fdpool_nc(fdpool_nc const&) = delete; // singleton
  void operator=(fdpool_nc const&) = delete;

  static fdpool_nc& get_instance() {
    static fdpool_nc instance; // singleton
    return instance;
  }

  int open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
  void close_all();

private:
  fdpool_nc() {} // singleton

  std::map<std::string, int> pool; // descriptors
};

////
inline int fdpool_nc::open(const std::string& f, MPI_Comm comm)
{
  auto it = pool.find(f);
  if (it == pool.end()) { // never opened
    int ncid, rtn;
    
#if NDARRAY_HAVE_NETCDF
#if NC_HAS_PARALLEL
    rtn = nc_open_par(f.c_str(), NC_NOWRITE, comm, MPI_INFO_NULL, &ncid);
    if (rtn != NC_NOERR)
      NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#else
    NC_SAFE_CALL( nc_open(f.c_str(), NC_NOWRITE, &ncid) );
#endif
#endif
    
    fprintf(stderr, "[fdpool] opened netcdf file %s, ncid=%d.\n", 
        f.c_str(), ncid);

    pool[f] = ncid;
    return ncid;
  } else 
    return it->second;
}

inline void fdpool_nc::close_all()
{
  for (const auto &kv : pool) {
#if NDARRAY_HAVE_NETCDF
    fprintf(stderr, "[fdpool] closing netcdf file %s, ncid=%d.\n", 
        kv.first.c_str(), kv.second);
    
    NC_SAFE_CALL( nc_close(kv.second) );
#endif
  }

  pool.clear();
}

} // namespace

#endif
