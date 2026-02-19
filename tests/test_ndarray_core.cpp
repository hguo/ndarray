/**
 * Core ndarray functionality tests
 *
 * Tests basic array operations:
 * - Construction and initialization
 * - Reshaping
 * - Element access
 * - Filling
 * - Slicing
 */

#include <ndarray/ndarray.hh>
#include <iostream>
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
  std::cout << "  Testing: " << name << std::endl

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "=== Running ndarray Core Tests ===" << std::endl << std::endl;

  int failed_tests = 0;

  // Test 1: Basic construction and sizing
  {
    TEST_SECTION("Basic construction");
    ftk::ndarray<double> arr;

    arr.reshapef(10);
    TEST_ASSERT(arr.nd() == 1, "1D array should have 1 dimension");
    TEST_ASSERT(arr.size() == 10, "1D array size should be 10");
    TEST_ASSERT(arr.dimf(0) == 10, "First dimension should be 10");

    arr.reshapef(5, 6);
    TEST_ASSERT(arr.nd() == 2, "2D array should have 2 dimensions");
    TEST_ASSERT(arr.size() == 30, "2D array size should be 30");
    TEST_ASSERT(arr.dimf(0) == 5, "First dimension should be 5");
    TEST_ASSERT(arr.dimf(1) == 6, "Second dimension should be 6");

    arr.reshapef(3, 4, 5);
    TEST_ASSERT(arr.nd() == 3, "3D array should have 3 dimensions");
    TEST_ASSERT(arr.size() == 60, "3D array size should be 60");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Vector constructor
  {
    TEST_SECTION("Vector constructor");
    std::vector<size_t> dims = {4, 5, 6};
    ftk::ndarray<double> arr(dims);

    TEST_ASSERT(arr.nd() == 3, "Array should have 3 dimensions");
    TEST_ASSERT(arr.size() == 120, "Array size should be 4*5*6=120");
    TEST_ASSERT(arr.dimf(0) == 4, "First dimension should be 4");
    TEST_ASSERT(arr.dimf(1) == 5, "Second dimension should be 5");
    TEST_ASSERT(arr.dimf(2) == 6, "Third dimension should be 6");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Fill operations
  {
    TEST_SECTION("Fill operations");
    ftk::ndarray<double> arr;
    arr.reshapef(100);
    arr.fill(3.14);

    for (size_t i = 0; i < arr.size(); i++) {
      TEST_ASSERT(arr[i] == 3.14, "All elements should be 3.14");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Element access and modification
  {
    TEST_SECTION("Element access and modification");
    ftk::ndarray<int> arr;
    arr.reshapef(10);

    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = static_cast<int>(i * 2);
    }

    for (size_t i = 0; i < arr.size(); i++) {
      TEST_ASSERT(arr[i] == static_cast<int>(i * 2), "Element value mismatch");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: Multi-dimensional access
  {
    TEST_SECTION("Multi-dimensional access");
    ftk::ndarray<double> arr;
    arr.reshapef(5, 10);

    // Set values
    arr.f(2, 3) = 42.0;
    arr.f(0, 0) = 1.0;
    arr.f(4, 9) = 99.0;

    // Check values
    TEST_ASSERT(arr.f(2, 3) == 42.0, "Value at (2,3) should be 42.0");
    TEST_ASSERT(arr.f(0, 0) == 1.0, "Value at (0,0) should be 1.0");
    TEST_ASSERT(arr.f(4, 9) == 99.0, "Value at (4,9) should be 99.0");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Reshaping preserves data (when possible)
  {
    TEST_SECTION("Reshaping operations");
    ftk::ndarray<double> arr;
    arr.reshapef(20);

    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = static_cast<double>(i);
    }

    arr.reshapef(4, 5);
    TEST_ASSERT(arr.nd() == 2, "Should be 2D after reshape");
    TEST_ASSERT(arr.size() == 20, "Size should remain 20");
    TEST_ASSERT(arr[0] == 0.0, "First element should be 0");
    TEST_ASSERT(arr[19] == 19.0, "Last element should be 19");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: Slicing
  {
    TEST_SECTION("Array slicing");
    ftk::ndarray<double> arr;
    arr.reshapef(10, 10);

    for (size_t i = 0; i < arr.size(); i++) {
      arr[i] = static_cast<double>(i);
    }

    std::vector<size_t> start = {2, 3};
    std::vector<size_t> sizes = {3, 4};
    auto sliced = arr.slice(start, sizes);

    TEST_ASSERT(sliced.nd() == 2, "Sliced array should be 2D");
    TEST_ASSERT(sliced.dimf(0) == 3, "Sliced first dimension should be 3");
    TEST_ASSERT(sliced.dimf(1) == 4, "Sliced second dimension should be 4");
    TEST_ASSERT(sliced.size() == 12, "Sliced size should be 3*4=12");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 8: Empty and reset
  {
    TEST_SECTION("Empty and reset operations");
    ftk::ndarray<double> arr;

    TEST_ASSERT(arr.empty(), "New array should be empty");

    arr.reshapef(10);
    TEST_ASSERT(!arr.empty(), "Array with data should not be empty");

    arr.reset();
    TEST_ASSERT(arr.empty(), "Reset array should be empty");
    TEST_ASSERT(arr.size() == 0, "Reset array size should be 0");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 9: Copy operations
  {
    TEST_SECTION("Copy operations");
    ftk::ndarray<double> arr1;
    arr1.reshapef(5, 5);
    arr1.fill(7.5);

    ftk::ndarray<double> arr2(arr1);
    TEST_ASSERT(arr2.size() == arr1.size(), "Copied array should have same size");
    TEST_ASSERT(arr2.nd() == arr1.nd(), "Copied array should have same dimensions");
    TEST_ASSERT(arr2[0] == 7.5, "Copied array should have same values");

    // Test assignment
    ftk::ndarray<double> arr3;
    arr3 = arr1;
    TEST_ASSERT(arr3.size() == arr1.size(), "Assigned array should have same size");
    TEST_ASSERT(arr3[0] == 7.5, "Assigned array should have same values");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 10: Different data types
  {
    TEST_SECTION("Different data types");

    ftk::ndarray<int> int_arr;
    int_arr.reshapef(10);
    int_arr.fill(42);
    TEST_ASSERT(int_arr[0] == 42, "Int array should hold integers");

    ftk::ndarray<float> float_arr;
    float_arr.reshapef(10);
    float_arr.fill(3.14f);
    TEST_ASSERT(std::abs(float_arr[0] - 3.14f) < 0.001f, "Float array should hold floats");

    ftk::ndarray<double> double_arr;
    double_arr.reshapef(10);
    double_arr.fill(2.71828);
    TEST_ASSERT(std::abs(double_arr[0] - 2.71828) < 0.00001, "Double array should hold doubles");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 11: Large arrays
  {
    TEST_SECTION("Large array handling");
    ftk::ndarray<double> arr;
    arr.reshapef(1000, 1000);

    TEST_ASSERT(arr.size() == 1000000, "Large array size should be correct");
    TEST_ASSERT(!arr.empty(), "Large array should not be empty");

    // Spot check
    arr[0] = 1.0;
    arr[999999] = 2.0;
    TEST_ASSERT(arr[0] == 1.0, "First element should be accessible");
    TEST_ASSERT(arr[999999] == 2.0, "Last element should be accessible");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 12: Element size
  {
    TEST_SECTION("Element size queries");
    ftk::ndarray<double> double_arr;
    double_arr.reshapef(10);
    TEST_ASSERT(double_arr.elem_size() == sizeof(double), "Double array element size");

    ftk::ndarray<float> float_arr;
    float_arr.reshapef(10);
    TEST_ASSERT(float_arr.elem_size() == sizeof(float), "Float array element size");

    ftk::ndarray<int> int_arr;
    int_arr.reshapef(10);
    TEST_ASSERT(int_arr.elem_size() == sizeof(int), "Int array element size");

    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
    std::cout << "=== All Core Tests Passed ===" << std::endl;
  
  #if NDARRAY_HAVE_MPI
    MPI_Finalize();
  #endif
    return 0;
  }
