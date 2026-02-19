#ifndef _NDARRAY_HIP_HH
#define _NDARRAY_HIP_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_HIP

#include <hip/hip_runtime.h>
#include <ndarray/error.hh>

namespace ftk {

/**
 * @brief HIP error checking macro
 *
 * Wraps HIP API calls and throws exception on error.
 *
 * @code
 * HIP_CHECK(hipMalloc(&ptr, size));
 * HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
 * @endcode
 */
#define HIP_CHECK(call) \
  do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
      throw ftk::device_error( \
          ftk::ERR_ACCELERATOR_UNSUPPORTED, \
          std::string("HIP error: ") + hipGetErrorString(err) + \
          " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while (0)

/**
 * @brief Get HIP device properties
 *
 * @param device_id HIP device ID
 * @return hipDeviceProp_t Device properties
 */
inline hipDeviceProp_t get_hip_device_properties(int device_id = 0)
{
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
  return prop;
}

/**
 * @brief Get number of available HIP devices
 *
 * @return Number of HIP devices
 */
inline int get_hip_device_count()
{
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  return count;
}

/**
 * @brief Check if HIP device is available
 *
 * @param device_id Device ID to check
 * @return true if device exists
 */
inline bool hip_device_available(int device_id = 0)
{
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) return false;
  return device_id < count;
}

/**
 * @brief Print HIP device information
 *
 * @param device_id Device ID
 */
inline void print_hip_device_info(int device_id = 0)
{
  if (!hip_device_available(device_id)) {
    std::cerr << "HIP device " << device_id << " not available" << std::endl;
    return;
  }

  hipDeviceProp_t prop = get_hip_device_properties(device_id);
  std::cout << "HIP Device " << device_id << ": " << prop.name << std::endl;
  std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
  std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "  Warp size: " << prop.warpSize << std::endl;
  std::cout << "  Memory clock rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
  std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
  std::cout << "  Peak memory bandwidth: " <<
    (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << " GB/s" << std::endl;
}

} // namespace ftk

#endif // NDARRAY_HAVE_HIP

#endif // _NDARRAY_HIP_HH
