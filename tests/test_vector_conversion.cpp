/**
 * Test std::vector to ndarray conversion
 *
 * Demonstrates multiple ways to convert std::vector to ndarray
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
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
  std::cout << "=== std::vector to ndarray Conversion Tests ===" << std::endl << std::endl;

  // Test 1: Constructor from vector (creates 1D array)
  {
    TEST_SECTION("Constructor from std::vector");

    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
    ftk::ndarray<double> arr(vec);

    TEST_ASSERT(arr.nd() == 1, "Should be 1D array");
    TEST_ASSERT(arr.size() == 5, "Should have 5 elements");
    TEST_ASSERT(arr[0] == 1.0, "First element should be 1.0");
    TEST_ASSERT(arr[4] == 5.0, "Last element should be 5.0");

    std::cout << "    Created 1D array with " << arr.size() << " elements" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Static factory method from_vector_data (1D)
  {
    TEST_SECTION("Static method from_vector_data (1D)");

    std::vector<float> vec = {10.0f, 20.0f, 30.0f};
    auto arr = ftk::ndarray<float>::from_vector_data(vec);

    TEST_ASSERT(arr.nd() == 1, "Should be 1D array");
    TEST_ASSERT(arr.size() == 3, "Should have 3 elements");
    TEST_ASSERT(arr[1] == 20.0f, "Second element should be 20.0");

    std::cout << "    Created 1D array: " << arr[0] << ", " << arr[1] << ", " << arr[2] << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Static factory method with shape (2D)
  {
    TEST_SECTION("Static method from_vector_data with shape (2D)");

    std::vector<double> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto arr = ftk::ndarray<double>::from_vector_data(vec, {3, 4});

    TEST_ASSERT(arr.nd() == 2, "Should be 2D array");
    TEST_ASSERT(arr.dimf(0) == 3, "First dimension should be 3");
    TEST_ASSERT(arr.dimf(1) == 4, "Second dimension should be 4");
    TEST_ASSERT(arr.size() == 12, "Should have 12 elements");
    TEST_ASSERT(arr.f(0, 0) == 1, "Element [0,0] should be 1");
    TEST_ASSERT(arr.f(2, 3) == 12, "Element [2,3] should be 12");

    std::cout << "    Created 2D array: " << arr.dimf(0) << " x " << arr.dimf(1) << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Static factory method with shape (3D)
  {
    TEST_SECTION("Static method from_vector_data with shape (3D)");

    std::vector<int> vec(24);
    for (int i = 0; i < 24; i++) {
      vec[i] = i;
    }

    auto arr = ftk::ndarray<int>::from_vector_data(vec, {2, 3, 4});

    TEST_ASSERT(arr.nd() == 3, "Should be 3D array");
    TEST_ASSERT(arr.dimf(0) == 2, "First dimension should be 2");
    TEST_ASSERT(arr.dimf(1) == 3, "Second dimension should be 3");
    TEST_ASSERT(arr.dimf(2) == 4, "Third dimension should be 4");
    TEST_ASSERT(arr[0] == 0, "First element should be 0");
    TEST_ASSERT(arr[23] == 23, "Last element should be 23");

    std::cout << "    Created 3D array: " << arr.dimf(0) << " x "
              << arr.dimf(1) << " x " << arr.dimf(2) << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: Old method copy_vector() still works
  {
    TEST_SECTION("Backward compatibility: copy_vector()");

    std::vector<double> vec = {100.0, 200.0, 300.0};
    ftk::ndarray<double> arr;
    arr.copy_vector(vec);

    TEST_ASSERT(arr.size() == 3, "Should have 3 elements");
    TEST_ASSERT(arr[0] == 100.0, "First element should be 100.0");

    std::cout << "    Old copy_vector() method still works" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Conversion back to std::vector
  {
    TEST_SECTION("Conversion back to std::vector");

    std::vector<double> original = {1.5, 2.5, 3.5};
    auto arr = ftk::ndarray<double>::from_vector_data(original);
    const auto& result = arr.std_vector();

    TEST_ASSERT(result.size() == 3, "Should have 3 elements");
    TEST_ASSERT(result[0] == 1.5, "First element should be 1.5");
    TEST_ASSERT(result[2] == 3.5, "Last element should be 3.5");

    std::cout << "    Round-trip conversion works" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: Partial fill (vector smaller than shape)
  {
    TEST_SECTION("Partial fill (vector smaller than shape)");

    std::vector<double> vec = {1.0, 2.0, 3.0};
    auto arr = ftk::ndarray<double>::from_vector_data(vec, {2, 3});  // Shape needs 6 elements

    TEST_ASSERT(arr.size() == 6, "Array should have 6 elements");
    TEST_ASSERT(arr[0] == 1.0, "First element filled");
    TEST_ASSERT(arr[2] == 3.0, "Third element filled");
    // Elements 3-5 remain uninitialized (default constructed)

    std::cout << "    Partial fill handled correctly" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 8: Different data types
  {
    TEST_SECTION("Different data types");

    std::vector<int> int_vec = {1, 2, 3};
    auto int_arr = ftk::ndarray<int>::from_vector_data(int_vec);

    std::vector<float> float_vec = {1.5f, 2.5f};
    auto float_arr = ftk::ndarray<float>::from_vector_data(float_vec);

    std::vector<uint8_t> byte_vec = {255, 128, 0};
    auto byte_arr = ftk::ndarray<uint8_t>::from_vector_data(byte_vec);

    TEST_ASSERT(int_arr[0] == 1, "Int conversion works");
    TEST_ASSERT(float_arr[0] == 1.5f, "Float conversion works");
    TEST_ASSERT(byte_arr[0] == 255, "Byte conversion works");

    std::cout << "    Multiple data types work" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All Vector Conversion Tests Passed ===" << std::endl;
  std::cout << std::endl;

  std::cout << "Summary of conversion methods:" << std::endl;
  std::cout << "  1. Constructor: ndarray<T> arr(vec)" << std::endl;
  std::cout << "  2. Static method (1D): ndarray<T>::from_vector_data(vec)" << std::endl;
  std::cout << "  3. Static method (N-D): ndarray<T>::from_vector_data(vec, shape)" << std::endl;
  std::cout << "  4. Old method: arr.copy_vector(vec)" << std::endl;
  std::cout << std::endl;

  std::cout << "Best practices:" << std::endl;
  std::cout << "  - Use constructor for simple 1D arrays" << std::endl;
  std::cout << "  - Use from_vector_data(vec, shape) for multi-dimensional arrays" << std::endl;
  std::cout << "  - Use std_vector() to convert back" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
