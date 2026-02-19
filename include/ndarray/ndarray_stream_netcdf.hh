#ifndef _NDARRAY_STREAM_NETCDF_HH
#define _NDARRAY_STREAM_NETCDF_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML && NDARRAY_HAVE_NETCDF

#include <ndarray/ndarray_stream.hh>
#include <ndarray/fdpool.hh>
#include <ndarray/variable_name_utils.hh>

namespace ftk {

/**
 * @brief NetCDF substream for time-varying data
 *
 * Reads variables from NetCDF files with support for:
 * - Unlimited time dimension
 * - Variable name aliasing (possible_names)
 * - Wildcard patterns for variable names
 * - Multi-file timesteps
 * - File descriptor pooling (prevents double-opening)
 */
template <typename StoragePolicy = native_storage>
struct substream_netcdf : public substream<StoragePolicy> {
  using stream_type = stream<StoragePolicy>;
  using group_type = ndarray_group<StoragePolicy>;
  substream_netcdf(stream_type& s) : substream<StoragePolicy>(s) {}
  bool require_input_files() { return true; }
  bool require_dimensions() { return false; }
  int direction() { return SUBSTREAM_DIR_INPUT;}

  void initialize(YAML::Node);
  void read(int, std::shared_ptr<group_type>);

  bool has_unlimited_time_dimension = false;
};

///////////
// Implementation
///////////

template <typename StoragePolicy>
inline void substream_netcdf<StoragePolicy>::read(int i, std::shared_ptr<group_type> g)
{
  int fi = this->locate_timestep_file_index(i);

  if (this->is_static) fi = 0;
  if (fi < 0) return;

  const std::string f = this->filenames[fi];

  fprintf(stderr, "static=%d, filename=%s, i=%d, fi=%d, filenames.size=%zu\n", this->is_static, f.c_str(), i, fi, this->filenames.size());

  auto &pool = fdpool_nc::get_instance();
  int ncid = pool.open(f, this->comm);

  for (const auto &var : this->variables) {
    int varid = -1;
    std::string found_varname;

    // Step 1: Try exact names from possible_names (fast path)
    for (const auto varname : var.possible_names) {
      int rtn = nc_inq_varid(ncid, varname.c_str(), &varid);

      if (rtn == NC_NOERR) {
        found_varname = varname;
        break;
      }
    }

    // Step 2: If not found, try wildcard patterns (slower but flexible)
    if (varid < 0 && !var.name_patterns.empty()) {
      auto available_vars = list_netcdf_variables(ncid);

      for (const auto& pattern : var.name_patterns) {
        for (const auto& available : available_vars) {
          if (matches_pattern(available, pattern)) {
            int rtn = nc_inq_varid(ncid, available.c_str(), &varid);
            if (rtn == NC_NOERR) {
              found_varname = available;
              break;
            }
          }
        }
        if (varid >= 0) break;
      }
    }

    if (varid >= 0) { // succ
      // Optionally log which name was actually used
      if (found_varname != var.name && var.possible_names.size() > 1) {
        // fprintf(stderr, "[ndarray] Variable '%s' found as '%s'\n",
        //         var.name.c_str(), found_varname.c_str());
      }
      // create a new array
      int type;
      NC_SAFE_CALL( nc_inq_vartype(ncid, varid, &type) );

      std::shared_ptr<ndarray_base> p;
      if (var.is_dtype_auto)
        p = ndarray_base::new_by_nc_dtype(type);
      else
        p = ndarray_base::new_by_dtype(var.dtype);

      // check if the variable has unlimited dimension
      int unlimited_recid;
      NC_SAFE_CALL( nc_inq_unlimdim(ncid, &unlimited_recid) );

      int ndims, dimids[4];
      NC_SAFE_CALL( nc_inq_varndims(ncid, varid, &ndims) );
      NC_SAFE_CALL( nc_inq_vardimid(ncid, varid, dimids) );

      bool time_varying = false;
      if (unlimited_recid >= 0) // has unlimied dimension
        for (int i = 0; i < ndims; i ++)
          if (dimids[i] == unlimited_recid)
            time_varying = true;

#if NDARRAY_HAVE_MPI
      // Configure distribution
      if (var.dist_type == VariableDistType::DISTRIBUTED) {
        // Find global dimensions
        std::vector<size_t> gdims(ndims);
        for (int d = 0; d < ndims; d++) {
          size_t len;
          NC_SAFE_CALL(nc_inq_dimlen(ncid, dimids[d], &len));
          gdims[d] = len;
        }
        
        // Remove time dimension from global dims if present (I/O handles it)
        if (time_varying) {
          // Assuming time is dim 0 in NetCDF
          gdims.erase(gdims.begin());
        }

        if (var.has_custom_decomposition) {
          p->decompose(this->comm, gdims, 0, var.custom_decomp.dims, var.custom_decomp.ghost);
        } else {
          // Use stream's default decomposition if available
          p->decompose(this->comm, gdims); 
        }
      } else {
        p->set_replicated(this->comm);
      }
#endif

      if (time_varying)
        p->read_netcdf_timestep(ncid, varid, i - this->first_timestep_per_file[fi], this->comm);
      else
        p->read_netcdf(ncid, varid, this->comm);

      if (var.multicomponents)
        p->make_multicomponents();

      g->set(var.name, p);

    } else { // failed
      if (!var.is_optional) {
        // Variable is required but not found - provide helpful error
        std::string error_msg = create_variable_not_found_message(
            var.possible_names, ncid);
        fprintf(stderr, "[NDARRAY ERROR] %s", error_msg.c_str());

        // For now, just warn instead of fatal error to maintain backward compatibility
        // In future versions, this should throw or fatal
        // nd::fatal(nd::ERR_NETCDF);
      }
    }
  }
}

template <typename StoragePolicy>
inline void substream_netcdf<StoragePolicy>::initialize(YAML::Node y)
{
  auto &pool = fdpool_nc::get_instance();
  for (const auto f : this->filenames) {
    int ncid = pool.open(f, this->comm);

    size_t nt = 0;
    int unlimited_recid;
    nc_inq_unlimdim(ncid, &unlimited_recid);

    if (unlimited_recid >= 0) {
      this->has_unlimited_time_dimension = true;

      NC_SAFE_CALL( nc_inq_dimlen(ncid, unlimited_recid, &nt) );
    } else {
      this->has_unlimited_time_dimension = false;
    }

    this->timesteps_per_file.push_back(nt);
    this->first_timestep_per_file.push_back( this->total_timesteps );
    this->total_timesteps += nt;
    fprintf(stderr, "filename=%s, nt=%zu\n", f.c_str(), nt);
  }
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML && NDARRAY_HAVE_NETCDF

#endif // _NDARRAY_STREAM_NETCDF_HH
