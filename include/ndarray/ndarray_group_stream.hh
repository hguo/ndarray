#ifndef _NDARRAY_NDARRAY_GROUP_STREAM_HH
#define _NDARRAY_NDARRAY_GROUP_STREAM_HH

/**
 * @file ndarray_group_stream.hh
 * @brief Unified header for time-varying scientific data streams
 *
 * This is a convenience header that includes all stream-related functionality.
 * For faster compilation, you can include only the specific format headers you need:
 *
 * - ndarray_stream.hh - Core stream classes
 * - ndarray_stream_netcdf.hh - NetCDF support
 * - ndarray_stream_hdf5.hh - HDF5 support
 * - ndarray_stream_adios2.hh - ADIOS2 support
 * - ndarray_stream_vtk.hh - VTK support
 * - ndarray_stream_binary.hh - Binary file support
 * - ndarray_stream_synthetic.hh - Synthetic data generation
 */

#include <ndarray/config.hh>

#if NDARRAY_HAVE_YAML

// Core stream classes
#include <ndarray/ndarray_stream.hh>

// Format-specific substreams
#include <ndarray/ndarray_stream_synthetic.hh>
#include <ndarray/ndarray_stream_binary.hh>

#if NDARRAY_HAVE_NETCDF
#include <ndarray/ndarray_stream_netcdf.hh>
#endif

#if NDARRAY_HAVE_HDF5
#include <ndarray/ndarray_stream_hdf5.hh>
#endif

#if NDARRAY_HAVE_ADIOS2
#include <ndarray/ndarray_stream_adios2.hh>
#endif

#if NDARRAY_HAVE_VTK
#include <ndarray/ndarray_stream_vtk.hh>
#endif

namespace ftk {

/**
 * @brief Create substream from YAML configuration
 *
 * Factory function that creates the appropriate substream type based on
 * the "format" field in the YAML configuration.
 *
 * Supported formats:
 * - "synthetic" - Synthetic data generation (name: "woven")
 * - "binary" - Raw binary files
 * - "netcdf" - NetCDF files (requires NDARRAY_HAVE_NETCDF)
 * - "h5" - HDF5 files (requires NDARRAY_HAVE_HDF5)
 * - "adios2" - ADIOS2 BP files (requires NDARRAY_HAVE_ADIOS2)
 * - "vti" - VTK ImageData files (requires NDARRAY_HAVE_VTK)
 * - "vtu_resample" - VTU resampled to regular grid (requires NDARRAY_HAVE_VTK)
 * - "vti_output" - VTI output (requires NDARRAY_HAVE_VTK)
 */
template <typename StoragePolicy>
inline void stream<StoragePolicy>::new_substream_from_yaml(YAML::Node y)
{
  std::shared_ptr<substream<StoragePolicy>> sub;

  std::string name;
  if (auto yname = y["name"])
    name = yname.as<std::string>();

  if (auto yformat = y["format"]) {
    std::string format = yformat.as<std::string>();
    if (format == "synthetic") {
      if (name == "woven")
        sub.reset(new substream_synthetic_woven<StoragePolicy>(*this));
      else
        fatal(nd::ERR_STREAM_FORMAT);
    }
    else if (format == "binary")
      sub.reset(new substream_binary<StoragePolicy>(*this));
#if NDARRAY_HAVE_NETCDF
    else if (format == "netcdf")
      sub.reset(new substream_netcdf<StoragePolicy>(*this));
#endif
#if NDARRAY_HAVE_HDF5
    else if (format == "h5")
      sub.reset(new substream_h5<StoragePolicy>(*this));
#endif
#if NDARRAY_HAVE_ADIOS2
    else if (format == "adios2")
      sub.reset(new substream_adios2<StoragePolicy>(*this));
#endif
#if NDARRAY_HAVE_VTK
    else if (format == "vti")
      sub.reset(new substream_vti<StoragePolicy>(*this));
    else if (format == "vtu_resample")
      sub.reset(new substream_vtu_resample<StoragePolicy>(*this));
    else if (format == "vti_output")
      sub.reset(new substream_vti_o<StoragePolicy>(*this));
#endif
    else
      nd::fatal(nd::ERR_STREAM_FORMAT);
  }

  sub->name = name;

  if (auto yvars = y["vars"]) { // has variable list
    for (auto i = 0; i < yvars.size(); i ++) {
      variable var;
      var.parse_yaml(yvars[i]);

      sub->variables.push_back(var);
    }
  }

  if (auto yfilenames = y["filenames"]) {
    if (yfilenames.IsScalar()) { // file name pattern
      sub->filename_pattern = path_prefix.empty() ? yfilenames.as<std::string>() :
        path_prefix + "/" + yfilenames.as<std::string>();

      if (sub->direction() == SUBSTREAM_DIR_INPUT) {
        sub->filenames = glob(sub->filename_pattern);
        fprintf(stderr, "input substream '%s', filename_pattern=%s, found %zu files.\n",
            sub->name.c_str(), sub->filename_pattern.c_str(), sub->filenames.size());
      }
    }
    else if (yfilenames.IsSequence()) {
      for (auto i = 0; i < yfilenames.size(); i ++) { // still, apply a pattern search
        const std::string mypattern = path_prefix.empty() ? yfilenames[i].as<std::string>() :
          path_prefix + "/" + yfilenames.as<std::string>();

        if (sub->direction() == SUBSTREAM_DIR_INPUT) {
          const auto filenames = glob(mypattern);
          sub->filenames.insert(sub->filenames.end(), filenames.begin(), filenames.end());
        }

        fprintf(stderr, "input substream '%s', found %zu files.\n",
            sub->name.c_str(), sub->filenames.size());
      }
    }
  }

  if (auto yopt = y["optional"])
    sub->is_optional = yopt.as<bool>();

  if (auto yenabled = y["enabled"])
    sub->is_enabled = yenabled.as<bool>();

  if (auto ystatic = y["static"])
    sub->is_static = ystatic.as<bool>();

  if (sub->require_input_files() && sub->filenames.empty()) {
    const std::string msg = "cannot find any files associated with substream " + sub->name + ", pattern=" + sub->filename_pattern;

    if (sub->is_optional) {
      nd::warn(msg);
      sub->is_enabled = false;
    } else
      nd::fatal(msg);
  }

  if (auto ydimensions = y["dimensions"]) {
      for (auto i = 0; i < ydimensions.size(); i ++)
        sub->dimensions.push_back(ydimensions[i].as<int>());

      for (auto &var : sub->variables) // overriding variables as well
        var.dimensions = this->dimensions;
  }

  if (this->has_dimensions()) {
    nd::warn("Overriding substream's dimensions with the stream's dimensions");
    sub->dimensions = this->dimensions;
    for (auto &var : sub->variables)
      var.dimensions = this->dimensions; // overriding variables as well
  }

  if (sub->is_enabled)
    sub->initialize(y);

  substreams.push_back(sub);
}

} // namespace ftk

#endif // NDARRAY_HAVE_YAML

// =============================================================================
// Distributed Memory Support (when MPI is enabled)
// =============================================================================

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

/**
 * Include distributed memory stream functionality when MPI is available.
 * This provides:
 * - distributed_ndarray: MPI-aware arrays with decomposition
 * - distributed_ndarray_group: Container for multiple distributed arrays
 * - distributed_stream: YAML-configured parallel I/O streams
 *
 * Usage is identical to regular streams, just use distributed_stream instead:
 * @code
 * ftk::distributed_stream<> stream(MPI_COMM_WORLD);
 * stream.parse_yaml("config.yaml");  // Add "decomposition" section to YAML
 * auto group = stream.read(timestep);
 * @endcode
 */

#include <ndarray/distributed_ndarray.hh>
#include <ndarray/distributed_ndarray_group.hh>
#include <ndarray/distributed_ndarray_stream.hh>

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_YAML

#endif // _NDARRAY_NDARRAY_GROUP_STREAM_HH
