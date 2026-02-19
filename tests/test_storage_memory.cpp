/**
 * Storage Backend Memory Management Tests
 *
 * Verifies proper memory management across storage backends:
 * - No memory leaks on allocation/deallocation
 * - Proper cleanup on exceptions
 * - Move semantics work correctly
 * - No double-free errors
 * - Proper handling of large allocations
 *
 * These tests help ensure that all storage policies properly manage
 * memory lifecycle and follow RAII principles.
 */

#include <ndarray/ndarray.hh>
#include <ndarray/config.hh>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <cmath>
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
  std::cout << "  " << name << std::endl

// Memory usage tracking (simplified)
struct MemoryStats {
  size_t allocations = 0;
  size_t deallocations = 0;
  size_t bytes_allocated = 0;

  void reset() {
    allocations = 0;
    deallocations = 0;
    bytes_allocated = 0;
  }
};

// Test 1: Basic allocation and deallocation
template <typename StoragePolicy>
int test_basic_memory_lifecycle() {
  std::cout << "\n=== Test 1: Basic Memory Lifecycle ===" << std::endl;

  TEST_SECTION("Allocate and deallocate small array");
  {
    ftk::ndarray<double, StoragePolicy> arr;
    arr.reshapef(100);
    arr.fill(3.14);
    TEST_ASSERT(arr.size() == 100, "Should have 100 elements");
    // Array goes out of scope here - should deallocate cleanly
  }

  TEST_SECTION("Allocate and deallocate large array");
  {
    ftk::ndarray<float, StoragePolicy> arr;
    arr.reshapef(1000000);  // 1M floats = 4MB
    arr.fill(2.71f);
    TEST_ASSERT(arr.size() == 1000000, "Should have 1M elements");
    // Should deallocate cleanly
  }

  TEST_SECTION("Multiple allocations");
  {
    std::vector<ftk::ndarray<int, StoragePolicy>> arrays;
    for (int i = 0; i < 100; i++) {
      ftk::ndarray<int, StoragePolicy> arr;
      arr.reshapef(1000);
      arr.fill(i);
      arrays.push_back(std::move(arr));
    }
    TEST_ASSERT(arrays.size() == 100, "Should have 100 arrays");
    // All arrays deallocate when vector goes out of scope
  }

  std::cout << "  ✓ All basic memory lifecycle tests passed" << std::endl;
  return 0;
}

// Test 2: Reshape and reallocation
template <typename StoragePolicy>
int test_reshape_memory() {
  std::cout << "\n=== Test 2: Reshape Memory Management ===" << std::endl;

  TEST_SECTION("Reshape to larger size");
  {
    ftk::ndarray<double, StoragePolicy> arr;
    arr.reshapef(100);
    arr.fill(1.0);

    arr.reshapef(1000);  // Grow
    TEST_ASSERT(arr.size() == 1000, "Should grow to 1000 elements");
    arr.fill(2.0);
  }

  TEST_SECTION("Reshape to smaller size");
  {
    ftk::ndarray<double, StoragePolicy> arr;
    arr.reshapef(1000);
    arr.fill(1.0);

    arr.reshapef(100);  // Shrink
    TEST_ASSERT(arr.size() == 100, "Should shrink to 100 elements");
  }

  TEST_SECTION("Multiple reshapes");
  {
    ftk::ndarray<float, StoragePolicy> arr;
    for (size_t size : {10, 100, 50, 500, 25, 1000}) {
      arr.reshapef(size);
      TEST_ASSERT(arr.size() == size, "Size should match after reshape");
      arr.fill(static_cast<float>(size));
    }
  }

  TEST_SECTION("Reshape with different dimensions");
  {
    ftk::ndarray<int, StoragePolicy> arr;

    arr.reshapef(10, 10);  // 2D: 100 elements
    TEST_ASSERT(arr.size() == 100, "Should have 100 elements");

    arr.reshapef(5, 5, 4);  // 3D: 100 elements
    TEST_ASSERT(arr.size() == 100, "Should still have 100 elements");

    arr.reshapef(100);  // 1D: 100 elements
    TEST_ASSERT(arr.size() == 100, "Should still have 100 elements");
  }

  std::cout << "  ✓ All reshape memory tests passed" << std::endl;
  return 0;
}

// Test 3: Copy and move semantics
template <typename StoragePolicy>
int test_copy_move_semantics() {
  std::cout << "\n=== Test 3: Copy and Move Semantics ===" << std::endl;

  TEST_SECTION("Copy constructor");
  {
    ftk::ndarray<double, StoragePolicy> arr1;
    arr1.reshapef(100);
    for (size_t i = 0; i < arr1.size(); i++) {
      arr1[i] = static_cast<double>(i);
    }

    ftk::ndarray<double, StoragePolicy> arr2(arr1);  // Copy
    TEST_ASSERT(arr2.size() == arr1.size(), "Sizes should match");
    TEST_ASSERT(arr2[50] == arr1[50], "Data should be copied");

    // Modify arr2, arr1 should be unchanged
    arr2[50] = 999.0;
    TEST_ASSERT(arr1[50] == 50.0, "Original should be unchanged");
  }

  TEST_SECTION("Copy assignment");
  {
    ftk::ndarray<float, StoragePolicy> arr1, arr2;
    arr1.reshapef(100);
    arr1.fill(3.14f);

    arr2 = arr1;  // Copy assignment
    TEST_ASSERT(arr2.size() == arr1.size(), "Sizes should match");
    TEST_ASSERT(arr2[0] == 3.14f, "Data should be copied");
  }

  TEST_SECTION("Move constructor");
  {
    ftk::ndarray<int, StoragePolicy> arr1;
    arr1.reshapef(100);
    arr1.fill(42);

    ftk::ndarray<int, StoragePolicy> arr2(std::move(arr1));  // Move
    TEST_ASSERT(arr2.size() == 100, "Should have moved data");
    TEST_ASSERT(arr2[0] == 42, "Data should be moved");
    // arr1 is now in moved-from state (unspecified but valid)
  }

  TEST_SECTION("Move assignment");
  {
    ftk::ndarray<double, StoragePolicy> arr1, arr2;
    arr1.reshapef(100);
    arr1.fill(2.71);

    arr2 = std::move(arr1);  // Move assignment
    TEST_ASSERT(arr2.size() == 100, "Should have moved data");
    TEST_ASSERT(arr2[0] == 2.71, "Data should be moved");
  }

  std::cout << "  ✓ All copy/move semantics tests passed" << std::endl;
  return 0;
}

// Test 4: Exception safety
template <typename StoragePolicy>
int test_exception_safety() {
  std::cout << "\n=== Test 4: Exception Safety ===" << std::endl;

  TEST_SECTION("Array cleanup on exception");
  {
    bool exception_caught = false;
    try {
      ftk::ndarray<double, StoragePolicy> arr;
      arr.reshapef(1000);
      arr.fill(1.0);

      // Simulate an exception during processing
      throw std::runtime_error("Test exception");

    } catch (const std::exception& e) {
      exception_caught = true;
    }

    TEST_ASSERT(exception_caught, "Exception should be caught");
    // Array should have been properly cleaned up by destructor
  }

  TEST_SECTION("Multiple arrays cleanup on exception");
  {
    bool exception_caught = false;
    try {
      std::vector<ftk::ndarray<float, StoragePolicy>> arrays;

      for (int i = 0; i < 10; i++) {
        ftk::ndarray<float, StoragePolicy> arr;
        arr.reshapef(1000);
        arr.fill(static_cast<float>(i));
        arrays.push_back(std::move(arr));
      }

      throw std::runtime_error("Test exception");

    } catch (const std::exception& e) {
      exception_caught = true;
    }

    TEST_ASSERT(exception_caught, "Exception should be caught");
    // All arrays should have been cleaned up
  }

  std::cout << "  ✓ All exception safety tests passed" << std::endl;
  return 0;
}

// Test 5: Large allocation stress test
template <typename StoragePolicy>
int test_large_allocations() {
  std::cout << "\n=== Test 5: Large Allocation Stress Test ===" << std::endl;

  TEST_SECTION("Allocate and deallocate very large array");
  {
    ftk::ndarray<float, StoragePolicy> arr;
    // 10M floats = 40MB
    arr.reshapef(10000000);
    TEST_ASSERT(arr.size() == 10000000, "Should allocate 10M elements");

    // Write to ensure memory is actually allocated
    arr[0] = 1.0f;
    arr[arr.size() - 1] = 2.0f;

    TEST_ASSERT(arr[0] == 1.0f, "First element should be accessible");
    TEST_ASSERT(arr[arr.size() - 1] == 2.0f, "Last element should be accessible");
  }

  TEST_SECTION("Multiple large allocations");
  {
    std::vector<ftk::ndarray<double, StoragePolicy>> arrays;

    // Allocate 10 arrays of 1M elements each (80MB total)
    for (int i = 0; i < 10; i++) {
      ftk::ndarray<double, StoragePolicy> arr;
      arr.reshapef(1000000);
      arr.fill(static_cast<double>(i));
      arrays.push_back(std::move(arr));
    }

    TEST_ASSERT(arrays.size() == 10, "Should have 10 large arrays");

    // Verify data integrity
    for (size_t i = 0; i < arrays.size(); i++) {
      TEST_ASSERT(arrays[i][0] == static_cast<double>(i),
                  "Data should be preserved");
    }
  }

  std::cout << "  ✓ All large allocation tests passed" << std::endl;
  return 0;
}

// Test 6: Zero-size arrays
template <typename StoragePolicy>
int test_zero_size() {
  std::cout << "\n=== Test 6: Zero-Size Arrays ===" << std::endl;

  TEST_SECTION("Create zero-size array");
  {
    ftk::ndarray<int, StoragePolicy> arr;
    TEST_ASSERT(arr.size() == 0, "Default array should be empty");
    TEST_ASSERT(arr.empty(), "Should be empty");
  }

  TEST_SECTION("Reshape to zero size");
  {
    ftk::ndarray<float, StoragePolicy> arr;
    arr.reshapef(100);
    arr.fill(1.0f);

    arr.reshapef(0);  // Reshape to zero
    TEST_ASSERT(arr.size() == 0, "Should be zero size");
    TEST_ASSERT(arr.empty(), "Should be empty");
  }

  TEST_SECTION("Operations on zero-size array");
  {
    ftk::ndarray<double, StoragePolicy> arr;
    TEST_ASSERT(arr.empty(), "Should be empty");

    // These operations should not crash
    arr.fill(1.0);  // No-op on empty array

    // Copy empty array
    ftk::ndarray<double, StoragePolicy> arr2 = arr;
    TEST_ASSERT(arr2.empty(), "Copy should also be empty");
  }

  std::cout << "  ✓ All zero-size array tests passed" << std::endl;
  return 0;
}

// Test 7: Memory reuse
template <typename StoragePolicy>
int test_memory_reuse() {
  std::cout << "\n=== Test 7: Memory Reuse ===" << std::endl;

  TEST_SECTION("Reuse array multiple times");
  {
    ftk::ndarray<float, StoragePolicy> arr;

    for (int iteration = 0; iteration < 100; iteration++) {
      arr.reshapef(1000);
      arr.fill(static_cast<float>(iteration));

      TEST_ASSERT(arr.size() == 1000, "Size should be consistent");
      TEST_ASSERT(arr[0] == static_cast<float>(iteration),
                  "Data should be correct");
    }
  }

  TEST_SECTION("Reuse with different sizes");
  {
    ftk::ndarray<double, StoragePolicy> arr;

    std::vector<size_t> sizes = {100, 1000, 50, 5000, 10, 10000};
    for (size_t size : sizes) {
      arr.reshapef(size);
      arr.fill(static_cast<double>(size));

      TEST_ASSERT(arr.size() == size, "Size should match");
      TEST_ASSERT(arr[0] == static_cast<double>(size), "Data should be correct");
    }
  }

  std::cout << "  ✓ All memory reuse tests passed" << std::endl;
  return 0;
}

template <typename StoragePolicy>
int run_all_memory_tests(const std::string& backend_name) {
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║  Memory Tests: " << std::left << std::setw(43) << backend_name << "║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

  int result = 0;
  result |= test_basic_memory_lifecycle<StoragePolicy>();
  result |= test_reshape_memory<StoragePolicy>();
  result |= test_copy_move_semantics<StoragePolicy>();
  result |= test_exception_safety<StoragePolicy>();
  result |= test_large_allocations<StoragePolicy>();
  result |= test_zero_size<StoragePolicy>();
  result |= test_memory_reuse<StoragePolicy>();

  return result;
}

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     Storage Backend Memory Management Test Suite          ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

  int result = 0;

  // Test native storage
  result |= run_all_memory_tests<ftk::native_storage>("Native Storage");

#if NDARRAY_HAVE_EIGEN
  // Test Eigen storage
  result |= run_all_memory_tests<ftk::eigen_storage>("Eigen Storage");
#else
  std::cout << "\n⊘ Skipping Eigen storage memory tests (NDARRAY_HAVE_EIGEN not defined)" << std::endl;
#endif

#if NDARRAY_HAVE_XTENSOR
  // Test xtensor storage
  result |= run_all_memory_tests<ftk::xtensor_storage>("xtensor Storage");
#else
  std::cout << "\n⊘ Skipping xtensor storage memory tests (NDARRAY_HAVE_XTENSOR not defined)" << std::endl;
#endif

  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  if (result == 0) {
    std::cout << "║  ✓✓✓ ALL MEMORY MANAGEMENT TESTS PASSED ✓✓✓              ║" << std::endl;
  } else {
    std::cout << "║  ✗✗✗ SOME TESTS FAILED ✗✗✗                               ║" << std::endl;
  }
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";

  std::cout << "Note: This test suite verifies correct memory management behavior.\n";
  std::cout << "For detailed memory leak detection, run with valgrind:\n";
  std::cout << "  valgrind --leak-check=full --show-leak-kinds=all ./test_storage_memory\n";
  std::cout << "\n";

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return result;
}
