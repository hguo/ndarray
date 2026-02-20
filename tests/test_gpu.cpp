/**
 * GPU (CUDA/SYCL) functionality tests
 *
 * Tests device memory management:
 * - to_device() / to_host() - move semantics
 * - copy_to_device() / copy_from_device() - copy semantics
 * - Device queries
 * - Data integrity after transfers
 * - Multi-device support
 */

#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_cuda.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return 1; \
    } \
  } while (0)

#define TEST_SECTION(name) \
  std::cout << "  Testing: " << name << std::endl

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "=== Running GPU Tests ===" << std::endl << std::endl;

#if !NDARRAY_HAVE_CUDA && !NDARRAY_HAVE_SYCL
  std::cout << "No GPU support enabled (CUDA or SYCL required)" << std::endl;
  std::cout << "Compile with -DNDARRAY_USE_CUDA=ON or -DNDARRAY_USE_SYCL=ON" << std::endl;
  return 0; // Skip tests, not a failure
#endif

#if NDARRAY_HAVE_CUDA
  std::cout << "CUDA support: ENABLED" << std::endl;

  // Check for CUDA devices
  int device_count = ftk::get_cuda_device_count();
  std::cout << "CUDA devices found: " << device_count << std::endl;

  if (device_count == 0) {
    std::cout << "No CUDA devices available, skipping CUDA tests" << std::endl;
    return 0;
  }

  // Print device info
  for (int i = 0; i < device_count; i++) {
    ftk::print_cuda_device_info(i);
  }
  std::cout << std::endl;
#endif

#if NDARRAY_HAVE_SYCL
  std::cout << "SYCL support: ENABLED" << std::endl << std::endl;
#endif

  const int device_type =
#if NDARRAY_HAVE_CUDA
    ftk::NDARRAY_DEVICE_CUDA;
#elif NDARRAY_HAVE_SYCL
    ftk::NDARRAY_DEVICE_SYCL;
#endif

  // Test 1: Basic to_device/to_host (move semantics)
  {
    TEST_SECTION("to_device() and to_host() - move semantics");

    ftk::ndarray<float> arr;
    arr.reshapef(1000);

    // Fill with test data
    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = static_cast<float>(i) * 0.5f;
    }

    // Verify initial state
    TEST_ASSERT(arr.is_on_host(), "Array should initially be on host");
    TEST_ASSERT(!arr.is_on_device(), "Array should not be on device");
    TEST_ASSERT(arr.get_device_type() == ftk::NDARRAY_DEVICE_HOST, "Device type should be HOST");

    // Move to device
    arr.to_device(device_type, 0);

    TEST_ASSERT(!arr.is_on_host(), "Array should not be on host after to_device");
    TEST_ASSERT(arr.is_on_device(), "Array should be on device after to_device");
    TEST_ASSERT(arr.get_device_type() == device_type, "Device type mismatch");
    TEST_ASSERT(arr.get_device_id() == 0, "Device ID should be 0");
    TEST_ASSERT(arr.get_devptr() != nullptr, "Device pointer should be valid");
    TEST_ASSERT(arr.size() == 0, "Host data should be cleared after move");

    // Move back to host
    arr.to_host();

    TEST_ASSERT(arr.is_on_host(), "Array should be on host after to_host");
    TEST_ASSERT(!arr.is_on_device(), "Array should not be on device after to_host");
    TEST_ASSERT(arr.size() == 1000, "Host data should be restored");
    TEST_ASSERT(arr.get_devptr() == nullptr, "Device pointer should be null");

    // Verify data integrity
    for (size_t i = 0; i < arr.size(); i++) {
      TEST_ASSERT(std::abs(arr[i] - static_cast<float>(i) * 0.5f) < 1e-6,
                  "Data corrupted after round-trip");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: copy_to_device/copy_from_device (copy semantics)
  {
    TEST_SECTION("copy_to_device() and copy_from_device() - copy semantics");

    ftk::ndarray<double> arr;
    arr.reshapef(500);

    // Fill with test data
    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = static_cast<double>(i) * 0.25;
    }

    // Copy to device
    arr.copy_to_device(device_type, 0);

    TEST_ASSERT(arr.is_on_device(), "Array should be on device");
    TEST_ASSERT(arr.size() == 500, "Host data should still be present");
    TEST_ASSERT(arr.get_devptr() != nullptr, "Device pointer should be valid");

    // Verify host data still accessible
    TEST_ASSERT(std::abs(arr[0] - 0.0) < 1e-10, "Host data should be accessible");
    TEST_ASSERT(std::abs(arr[499] - 499 * 0.25) < 1e-10, "Host data should be correct");

    // Modify host data (simulating modification on device in real scenario)
    double old_value = arr[250];
    arr[250] = 999.0;

    // Copy from device (should overwrite host modification)
    arr.copy_from_device();

    TEST_ASSERT(arr.is_on_device(), "Device memory should still be allocated");
    TEST_ASSERT(arr.get_devptr() != nullptr, "Device pointer should still be valid");

    // Note: In a real scenario with kernel execution, device data would change.
    // Here, device data hasn't changed, so we expect the original value back.
    TEST_ASSERT(std::abs(arr[250] - old_value) < 1e-10,
                "Data should be copied from device");

    // Clean up
    arr.to_host();

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Data integrity with different data types
  {
    TEST_SECTION("Data integrity - multiple data types");

    // Test int
    {
      ftk::ndarray<int> arr_int;
      arr_int.reshapef(100);
      for (size_t i = 0; i < arr_int.size(); i++) {
        arr_int[i] = static_cast<int>(i * 17);
      }

      arr_int.to_device(device_type, 0);
      arr_int.to_host();

      for (size_t i = 0; i < arr_int.size(); i++) {
        TEST_ASSERT(arr_int[i] == static_cast<int>(i * 17),
                    "Int data corrupted");
      }
    }

    // Test float
    {
      ftk::ndarray<float> arr_float;
      arr_float.reshapef(100);
      for (size_t i = 0; i < arr_float.size(); i++) {
        arr_float[i] = static_cast<float>(i) * 3.14f;
      }

      arr_float.to_device(device_type, 0);
      arr_float.to_host();

      for (size_t i = 0; i < arr_float.size(); i++) {
        TEST_ASSERT(std::abs(arr_float[i] - static_cast<float>(i) * 3.14f) < 1e-5,
                    "Float data corrupted");
      }
    }

    // Test double
    {
      ftk::ndarray<double> arr_double;
      arr_double.reshapef(100);
      for (size_t i = 0; i < arr_double.size(); i++) {
        arr_double[i] = static_cast<double>(i) * 2.71828;
      }

      arr_double.to_device(device_type, 0);
      arr_double.to_host();

      for (size_t i = 0; i < arr_double.size(); i++) {
        TEST_ASSERT(std::abs(arr_double[i] - static_cast<double>(i) * 2.71828) < 1e-10,
                    "Double data corrupted");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Multi-dimensional arrays
  {
    TEST_SECTION("Multi-dimensional arrays on device");

    // 2D array
    {
      ftk::ndarray<float> arr2d;
      arr2d.reshapef(10, 20);

      for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 20; j++) {
          arr2d.f(i, j) = static_cast<float>(i * 20 + j);
        }
      }

      arr2d.to_device(device_type, 0);
      arr2d.to_host();

      for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 20; j++) {
          TEST_ASSERT(std::abs(arr2d.f(i, j) - static_cast<float>(i * 20 + j)) < 1e-5,
                      "2D array data corrupted");
        }
      }
    }

    // 3D array
    {
      ftk::ndarray<double> arr3d;
      arr3d.reshapef(5, 6, 7);

      for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 6; j++) {
          for (size_t k = 0; k < 7; k++) {
            arr3d.f(i, j, k) = static_cast<double>(i * 42 + j * 7 + k);
          }
        }
      }

      arr3d.to_device(device_type, 0);
      arr3d.to_host();

      for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 6; j++) {
          for (size_t k = 0; k < 7; k++) {
            TEST_ASSERT(std::abs(arr3d.f(i, j, k) - static_cast<double>(i * 42 + j * 7 + k)) < 1e-10,
                        "3D array data corrupted");
          }
        }
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: Large array performance test
  {
    TEST_SECTION("Large array transfer performance");

    const size_t large_size = 10000000; // 10 million elements
    ftk::ndarray<float> large_arr;
    large_arr.reshapef(large_size);

    // Fill with data
    for (size_t i = 0; i < large_size; i++) {
      large_arr[i] = static_cast<float>(i % 10000);
    }

    std::cout << "    Transferring " << (large_size * sizeof(float) / (1024*1024)) << " MB to device..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    large_arr.to_device(device_type, 0);
    auto end = std::chrono::high_resolution_clock::now();
    auto to_device_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "    to_device() time: " << to_device_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    large_arr.to_host();
    end = std::chrono::high_resolution_clock::now();
    auto to_host_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "    to_host() time: " << to_host_time << " ms" << std::endl;

    // Verify first and last elements
    TEST_ASSERT(std::abs(large_arr[0] - 0.0f) < 1e-5, "First element corrupted");
    TEST_ASSERT(std::abs(large_arr[large_size-1] - static_cast<float>((large_size-1) % 10000)) < 1e-5,
                "Last element corrupted");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Error handling
  {
    TEST_SECTION("Error handling");

    ftk::ndarray<float> arr;
    arr.reshapef(100);

    // Test double to_device() warning
    arr.to_device(device_type, 0);
    arr.to_device(device_type, 0); // Should warn but not fail
    TEST_ASSERT(arr.is_on_device(), "Should still be on device");

    // Test double to_host() warning
    arr.to_host();
    arr.to_host(); // Should warn but not fail
    TEST_ASSERT(arr.is_on_host(), "Should still be on host");

    std::cout << "    PASSED" << std::endl;
  }

#if NDARRAY_HAVE_CUDA
  // Test 7: Multi-device support (CUDA only, requires 2+ devices)
  if (ftk::get_cuda_device_count() >= 2) {
    TEST_SECTION("Multi-device support");

    ftk::ndarray<float> arr0, arr1;
    arr0.reshapef(1000);
    arr1.reshapef(1000);

    for (size_t i = 0; i < 1000; i++) {
      arr0[i] = static_cast<float>(i);
      arr1[i] = static_cast<float>(i) * 2.0f;
    }

    // Transfer to different devices
    arr0.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
    arr1.to_device(ftk::NDARRAY_DEVICE_CUDA, 1);

    TEST_ASSERT(arr0.get_device_id() == 0, "Device 0 mismatch");
    TEST_ASSERT(arr1.get_device_id() == 1, "Device 1 mismatch");

    // Transfer back
    arr0.to_host();
    arr1.to_host();

    // Verify data
    TEST_ASSERT(std::abs(arr0[500] - 500.0f) < 1e-5, "Device 0 data corrupted");
    TEST_ASSERT(std::abs(arr1[500] - 1000.0f) < 1e-5, "Device 1 data corrupted");

    std::cout << "    PASSED" << std::endl;
  } else {
    std::cout << "  Testing: Multi-device support - SKIPPED (requires 2+ GPUs)" << std::endl;
  }

  // Test 8: GPU kernel execution (CUDA only for now)
  if (device_type == ftk::NDARRAY_DEVICE_CUDA) {
    TEST_SECTION("GPU kernel execution (fill, scale, add)");

    ftk::ndarray<float> arr1, arr2;
    arr1.reshapef(100);
    arr2.reshapef(100);

    arr1.fill(1.0f);
    arr2.fill(2.0f);

    arr1.to_device(device_type, 0);
    arr2.to_device(device_type, 0);

    // Test GPU fill
    arr1.fill(10.0f); 
    
    // Test GPU scale
    arr2.scale(5.0f); // 2.0 * 5.0 = 10.0

    // Test GPU add
    arr1.add(arr2); // 10.0 + 10.0 = 20.0

    arr1.to_host();
    for (size_t i = 0; i < arr1.size(); i++) {
      TEST_ASSERT(std::abs(arr1[i] - 20.0f) < 1e-5, "GPU kernel results incorrect");
    }

    std::cout << "    PASSED" << std::endl;
  }
#endif

  std::cout << std::endl;
  std::cout << "=== All GPU tests passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
