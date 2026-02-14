#ifndef _NDARRAY_CUDA_HH
#define _NDARRAY_CUDA_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_CUDA

#include <cuda_runtime.h>
#include <ndarray/error.hh>

namespace ftk {
namespace nd {

/**
 * @brief CUDA error checking macro
 *
 * Wraps CUDA API calls and throws exception on error.
 *
 * @code
 * CUDA_CHECK(cudaMalloc(&ptr, size));
 * CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 * @endcode
 */
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw ftk::nd::device_error( \
          ftk::nd::ERR_NOT_BUILT_WITH_CUDA, \
          std::string("CUDA error: ") + cudaGetErrorString(err) + \
          " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while (0)

/**
 * @brief Get CUDA device properties
 *
 * @param device_id CUDA device ID
 * @return cudaDeviceProp Device properties
 */
inline cudaDeviceProp get_cuda_device_properties(int device_id = 0)
{
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop;
}

/**
 * @brief Get number of available CUDA devices
 *
 * @return Number of CUDA devices
 */
inline int get_cuda_device_count()
{
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

/**
 * @brief Check if CUDA device is available
 *
 * @param device_id Device ID to check
 * @return true if device exists
 */
inline bool cuda_device_available(int device_id = 0)
{
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) return false;
  return device_id < count;
}

/**
 * @brief Print CUDA device information
 *
 * @param device_id Device ID
 */
inline void print_cuda_device_info(int device_id = 0)
{
  if (!cuda_device_available(device_id)) {
    std::cerr << "CUDA device " << device_id << " not available" << std::endl;
    return;
  }

  cudaDeviceProp prop = get_cuda_device_properties(device_id);
  std::cout << "CUDA Device " << device_id << ": " << prop.name << std::endl;
  std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
  std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "  Warp size: " << prop.warpSize << std::endl;
}

} // namespace nd
} // namespace ftk

#endif // NDARRAY_HAVE_CUDA

#endif // _NDARRAY_CUDA_HH
