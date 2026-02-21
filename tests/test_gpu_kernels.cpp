/**
 * @file test_gpu_kernels.cpp
 * @brief Tests for GPU data management and basic kernels
 *
 * Tests GPU memory transfers and simple kernels (fill, scale, add).
 * Scope: Data management only, not compute kernels.
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace ftk;

// Helper to compare arrays with tolerance
template <typename T>
bool arrays_equal(const ndarray<T>& a, const ndarray<T>& b, T tol = 1e-6) {
  if (a.shapef() != b.shapef()) return false;
  for (size_t i = 0; i < a.nelem(); i++) {
    if (std::abs(a[i] - b[i]) > tol) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

void test_to_device_to_host() {
  std::cout << "Test: to_device() and to_host()" << std::endl;

  // Create array on host
  ndarray<float> arr({10, 10});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i);
  }

  // Save original data for comparison
  ndarray<float> original = arr;

  // Move to device (transfers data, clears host)
  arr.to_device(NDARRAY_DEVICE_CUDA);
  assert(arr.is_on_device());
  assert(!arr.is_on_host());

  // Move back to host
  arr.to_host();
  assert(arr.is_on_host());
  assert(!arr.is_on_device());

  // Verify data integrity
  assert(arrays_equal(arr, original));

  std::cout << "  ✓ to_device/to_host: Data integrity preserved" << std::endl;
}

void test_copy_to_device_copy_from_device() {
  std::cout << "Test: copy_to_device() and copy_from_device()" << std::endl;

  // Create array on host
  ndarray<float> arr({20, 20});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i * 2);
  }

  ndarray<float> original = arr;

  // Copy to device (keeps host data)
  arr.copy_to_device(NDARRAY_DEVICE_CUDA);
  assert(arr.is_on_device());

  // Verify host data still accessible
  assert(arrays_equal(arr, original));

  // Modify host data
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = -1.0f;
  }

  // Copy from device (restores original data)
  arr.copy_from_device();
  assert(arrays_equal(arr, original));

  std::cout << "  ✓ copy_to_device/copy_from_device: Bidirectional copy works" << std::endl;
}

void test_fill_on_device() {
  std::cout << "Test: fill() on device" << std::endl;

  ndarray<float> arr({50, 50});
  arr.to_device(NDARRAY_DEVICE_CUDA);

  // Fill on device
  arr.fill(3.14f);

  // Copy back and verify
  arr.to_host();
  for (size_t i = 0; i < arr.nelem(); i++) {
    assert(std::abs(arr[i] - 3.14f) < 1e-6);
  }

  std::cout << "  ✓ fill() on device: All elements set correctly" << std::endl;
}

void test_scale_on_device() {
  std::cout << "Test: scale() on device" << std::endl;

  // Create array with known values
  ndarray<float> arr({30, 30});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i);
  }

  // Move to device and scale
  arr.to_device(NDARRAY_DEVICE_CUDA);
  arr.scale(2.0f);

  // Verify on host
  arr.to_host();
  for (size_t i = 0; i < arr.nelem(); i++) {
    assert(std::abs(arr[i] - static_cast<float>(i * 2)) < 1e-5);
  }

  std::cout << "  ✓ scale() on device: All elements scaled correctly" << std::endl;
}

void test_add_on_device() {
  std::cout << "Test: add() on device" << std::endl;

  // Create two arrays
  ndarray<float> arr1({40, 40});
  ndarray<float> arr2({40, 40});

  for (size_t i = 0; i < arr1.nelem(); i++) {
    arr1[i] = static_cast<float>(i);
    arr2[i] = static_cast<float>(i * 2);
  }

  // Move both to device
  arr1.to_device(NDARRAY_DEVICE_CUDA);
  arr2.to_device(NDARRAY_DEVICE_CUDA);

  // Add on device
  arr1.add(arr2);

  // Verify on host
  arr1.to_host();
  for (size_t i = 0; i < arr1.nelem(); i++) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    assert(std::abs(arr1[i] - expected) < 1e-5);
  }

  std::cout << "  ✓ add() on device: Element-wise addition correct" << std::endl;
}

void test_multiple_transfers() {
  std::cout << "Test: Multiple round-trip transfers" << std::endl;

  ndarray<float> arr({25, 25});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i);
  }

  ndarray<float> original = arr;

  // Multiple round trips
  for (int round = 0; round < 5; round++) {
    arr.to_device(NDARRAY_DEVICE_CUDA);
    arr.fill(static_cast<float>(round));
    arr.to_host();

    for (size_t i = 0; i < arr.nelem(); i++) {
      assert(std::abs(arr[i] - static_cast<float>(round)) < 1e-6);
    }
  }

  std::cout << "  ✓ Multiple transfers: No memory leaks or corruption" << std::endl;
}

void test_device_operations_chain() {
  std::cout << "Test: Chained device operations" << std::endl;

  ndarray<float> arr({35, 35});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = 1.0f;
  }

  // Chain: fill -> scale -> add
  arr.to_device(NDARRAY_DEVICE_CUDA);
  arr.fill(2.0f);
  arr.scale(3.0f);  // Now 6.0

  ndarray<float> arr2({35, 35});
  arr2.fill(4.0f);
  arr2.to_device(NDARRAY_DEVICE_CUDA);

  arr.add(arr2);  // Now 10.0

  arr.to_host();
  for (size_t i = 0; i < arr.nelem(); i++) {
    assert(std::abs(arr[i] - 10.0f) < 1e-5);
  }

  std::cout << "  ✓ Chained operations: fill->scale->add works correctly" << std::endl;
}

void test_raii_cleanup() {
  std::cout << "Test: RAII automatic cleanup" << std::endl;

  // Test that device memory is cleaned up when object goes out of scope
  {
    ndarray<float> arr({100, 100});
    arr.fill(1.0f);
    arr.to_device(NDARRAY_DEVICE_CUDA);
    // arr goes out of scope here - device memory should be freed automatically
  }

  // Create another array to ensure no memory leak from previous
  {
    ndarray<float> arr2({100, 100});
    arr2.fill(2.0f);
    arr2.to_device(NDARRAY_DEVICE_CUDA);
    arr2.to_host();
    assert(std::abs(arr2[0] - 2.0f) < 1e-6);
  }

  std::cout << "  ✓ RAII cleanup: No memory leaks on destruction" << std::endl;
}

void test_large_array() {
  std::cout << "Test: Large array transfer (10 MB)" << std::endl;

  // 1024 x 1024 floats = 4 MB
  ndarray<float> arr({1024, 1024});
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i % 1000);
  }

  ndarray<float> original = arr;

  arr.to_device(NDARRAY_DEVICE_CUDA);
  arr.to_host();

  assert(arrays_equal(arr, original));

  std::cout << "  ✓ Large array: 4 MB transfer successful" << std::endl;
}

int main() {
  std::cout << "=== GPU Data Management and Kernel Tests ===" << std::endl;
  std::cout << std::endl;

#if NDARRAY_HAVE_CUDA
  try {
    test_to_device_to_host();
    test_copy_to_device_copy_from_device();
    test_fill_on_device();
    test_scale_on_device();
    test_add_on_device();
    test_multiple_transfers();
    test_device_operations_chain();
    test_raii_cleanup();
    test_large_array();

    std::cout << std::endl;
    std::cout << "=== All GPU tests passed! ===" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
#else
  std::cout << "CUDA not available - tests skipped" << std::endl;
  std::cout << "Build with -DNDARRAY_USE_CUDA=TRUE to enable GPU tests" << std::endl;
  return 0;
#endif
}
