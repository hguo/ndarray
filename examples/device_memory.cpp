#include <ndarray/ndarray.hh>
#include <iostream>
#include <chrono>

/**
 * Device Memory Management Example for ndarray
 *
 * This example demonstrates:
 * - Moving data between host and device
 * - CUDA device memory management (if available)
 * - SYCL device memory management (if available)
 * - Copy vs move semantics for device transfers
 * - Querying device status
 *
 * Compile with device support:
 * - CUDA: -DNDARRAY_USE_CUDA=ON
 * - SYCL: -DNDARRAY_USE_SYCL=ON
 */

int main() {
  std::cout << "=== ndarray Device Memory Management Example ===" << std::endl << std::endl;

  // Create a sample array
  const size_t N = 1000000; // 1 million elements
  std::cout << "Creating host array with " << N << " elements..." << std::endl;

  ftk::ndarray<float> arr;
  arr.reshapef(N);

  // Initialize with data
  for (size_t i = 0; i < N; i++) {
    arr[i] = static_cast<float>(i) * 0.1f;
  }

  std::cout << "  Array size: " << arr.size() << " elements" << std::endl;
  std::cout << "  Memory usage: " << (arr.size() * arr.elem_size()) / (1024*1024) << " MB" << std::endl;
  std::cout << "  First element: " << arr[0] << std::endl;
  std::cout << "  Last element: " << arr[N-1] << std::endl;
  std::cout << std::endl;

  // Check initial device status
  std::cout << "Initial device status:" << std::endl;
  std::cout << "  is_on_host(): " << (arr.is_on_host() ? "true" : "false") << std::endl;
  std::cout << "  is_on_device(): " << (arr.is_on_device() ? "true" : "false") << std::endl;
  std::cout << "  device_type: " << arr.get_device_type() << " (0=HOST, 1=CUDA, 3=SYCL)" << std::endl;
  std::cout << std::endl;

#if NDARRAY_HAVE_CUDA || NDARRAY_HAVE_SYCL

  // Demonstrate moving data to device (clears host memory)
  std::cout << "=== Moving data to device (host memory will be cleared) ===" << std::endl;

#if NDARRAY_HAVE_CUDA
  std::cout << "Moving to CUDA device..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  arr.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "  Transfer time: " << duration.count() << " ms" << std::endl;
#elif NDARRAY_HAVE_SYCL
  std::cout << "Moving to SYCL device..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  arr.to_device(ftk::NDARRAY_DEVICE_SYCL, 0);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "  Transfer time: " << duration.count() << " ms" << std::endl;
#endif

  std::cout << "Device status after to_device():" << std::endl;
  std::cout << "  is_on_host(): " << (arr.is_on_host() ? "true" : "false") << std::endl;
  std::cout << "  is_on_device(): " << (arr.is_on_device() ? "true" : "false") << std::endl;
  std::cout << "  device_type: " << arr.get_device_type() << std::endl;
  std::cout << "  device_id: " << arr.get_device_id() << std::endl;
  std::cout << "  devptr: " << arr.get_devptr() << std::endl;
  std::cout << "  Host data size: " << arr.size() << " (should be 0 after move)" << std::endl;
  std::cout << std::endl;

  // Move data back to host
  std::cout << "=== Moving data back to host (device memory will be freed) ===" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  arr.to_host();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "  Transfer time: " << duration.count() << " ms" << std::endl;

  std::cout << "Device status after to_host():" << std::endl;
  std::cout << "  is_on_host(): " << (arr.is_on_host() ? "true" : "false") << std::endl;
  std::cout << "  is_on_device(): " << (arr.is_on_device() ? "true" : "false") << std::endl;
  std::cout << "  Host data size: " << arr.size() << std::endl;
  std::cout << "  First element: " << arr[0] << std::endl;
  std::cout << "  Last element: " << arr[N-1] << std::endl;
  std::cout << std::endl;

  // Demonstrate copying data to device (keeps host memory)
  std::cout << "=== Copying data to device (host memory preserved) ===" << std::endl;

#if NDARRAY_HAVE_CUDA
  std::cout << "Copying to CUDA device..." << std::endl;
  arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
#elif NDARRAY_HAVE_SYCL
  std::cout << "Copying to SYCL device..." << std::endl;
  arr.copy_to_device(ftk::NDARRAY_DEVICE_SYCL, 0);
#endif

  std::cout << "Device status after copy_to_device():" << std::endl;
  std::cout << "  is_on_device(): " << (arr.is_on_device() ? "true" : "false") << std::endl;
  std::cout << "  Host data size: " << arr.size() << " (should still be " << N << ")" << std::endl;
  std::cout << "  Host data accessible: " << (arr.size() > 0 ? "YES" : "NO") << std::endl;
  std::cout << "  First element on host: " << arr[0] << std::endl;
  std::cout << std::endl;

  // Simulate modifying device data (in real usage, you'd run a kernel here)
  std::cout << "=== Simulating device computation ===" << std::endl;
  std::cout << "  (In real usage, you would run CUDA/SYCL kernels here)" << std::endl;
  std::cout << "  For demonstration, we'll just copy data back..." << std::endl;
  std::cout << std::endl;

  // Copy data from device back to host (keeps device memory)
  std::cout << "=== Copying data from device to host (device memory preserved) ===" << std::endl;
  arr.copy_from_device();

  std::cout << "Device status after copy_from_device():" << std::endl;
  std::cout << "  is_on_device(): " << (arr.is_on_device() ? "true" : "false") << std::endl;
  std::cout << "  Host data size: " << arr.size() << std::endl;
  std::cout << "  devptr: " << arr.get_devptr() << " (should still be valid)" << std::endl;
  std::cout << std::endl;

  // Clean up by moving back to host
  std::cout << "=== Final cleanup - moving to host ===" << std::endl;
  arr.to_host();
  std::cout << "  Final device status: " << (arr.is_on_host() ? "on host" : "on device") << std::endl;
  std::cout << std::endl;

  // Summary of memory management patterns
  std::cout << "=== Summary of Memory Management Patterns ===" << std::endl;
  std::cout << std::endl;
  std::cout << "1. to_device(device, id):" << std::endl;
  std::cout << "   - Moves data from host to device" << std::endl;
  std::cout << "   - Clears host memory (efficient for large data)" << std::endl;
  std::cout << "   - Use when you don't need host access during computation" << std::endl;
  std::cout << std::endl;

  std::cout << "2. to_host():" << std::endl;
  std::cout << "   - Moves data from device to host" << std::endl;
  std::cout << "   - Frees device memory" << std::endl;
  std::cout << "   - Use when computation is done and you need results on host" << std::endl;
  std::cout << std::endl;

  std::cout << "3. copy_to_device(device, id):" << std::endl;
  std::cout << "   - Copies data from host to device" << std::endl;
  std::cout << "   - Keeps host memory intact" << std::endl;
  std::cout << "   - Use when you need data on both host and device" << std::endl;
  std::cout << std::endl;

  std::cout << "4. copy_from_device():" << std::endl;
  std::cout << "   - Copies data from device to host" << std::endl;
  std::cout << "   - Keeps device memory intact" << std::endl;
  std::cout << "   - Use when you want to peek at results without stopping device work" << std::endl;
  std::cout << std::endl;

  std::cout << "Device Query Methods:" << std::endl;
  std::cout << "  - is_on_host(): Check if data is on host" << std::endl;
  std::cout << "  - is_on_device(): Check if data is on device" << std::endl;
  std::cout << "  - get_device_type(): Get current device type (HOST/CUDA/SYCL)" << std::endl;
  std::cout << "  - get_device_id(): Get device ID" << std::endl;
  std::cout << "  - get_devptr(): Get raw device pointer for custom kernels" << std::endl;
  std::cout << std::endl;

#else
  std::cout << "No device support enabled!" << std::endl;
  std::cout << "Please compile with -DNDARRAY_USE_CUDA=ON or -DNDARRAY_USE_SYCL=ON" << std::endl;
  std::cout << std::endl;
  std::cout << "Available device management methods:" << std::endl;
  std::cout << "  - to_device(device_type, device_id)" << std::endl;
  std::cout << "  - to_host()" << std::endl;
  std::cout << "  - copy_to_device(device_type, device_id)" << std::endl;
  std::cout << "  - copy_from_device()" << std::endl;
  std::cout << "  - is_on_host() / is_on_device()" << std::endl;
  std::cout << "  - get_device_type() / get_device_id()" << std::endl;
  std::cout << "  - get_devptr()" << std::endl;
#endif

  std::cout << std::endl;
  std::cout << "=== Example completed successfully ===" << std::endl;

  return 0;
}
