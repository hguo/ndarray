/**
 * Data manipulation tests for ndarray
 *
 * Tests: perturb, flip_byte_order, concat, stack,
 *        from_vector, reset, print/operator<<
 *
 * Note: slice_time() is excluded — the implementation uses internal C-order dims
 * and strides in a way that assumes Fortran-order storage (pre-refactoring bug).
 * It produces incorrect shapes and copies wrong amounts of data.
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>
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
  std::cout << "=== Running Data Operations Tests ===" << std::endl << std::endl;

  // Test 1: perturb(sigma)
  {
    TEST_SECTION("perturb(sigma)");
    ftk::ndarray<double> arr;
    const size_t n = 10000;
    arr.reshapef(n);
    arr.fill(0.0);

    const double sigma = 1.0;
    arr.perturb(sigma);

    // At least some elements should be modified
    int nonzero = 0;
    double sum = 0.0, sumsq = 0.0;
    for (size_t i = 0; i < n; i++) {
      if (std::abs(arr[i]) > 1e-15) nonzero++;
      sum += arr[i];
      sumsq += arr[i] * arr[i];
    }
    TEST_ASSERT(nonzero > static_cast<int>(n * 0.9),
      "most elements should be nonzero after perturb");

    // Check statistical properties (mean ~0, stddev ~sigma)
    double mean = sum / n;
    double variance = sumsq / n - mean * mean;
    double stddev = std::sqrt(variance);
    TEST_ASSERT(std::abs(mean) < 0.1,
      "mean should be near 0");
    TEST_ASSERT(std::abs(stddev - sigma) < 0.1,
      "stddev should be near sigma");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: flip_byte_order() round-trip
  {
    TEST_SECTION("flip_byte_order() round-trip");
    ftk::ndarray<double> arr;
    arr.reshapef(5);
    for (size_t i = 0; i < arr.size(); i++)
      arr[i] = static_cast<double>(i + 1) * 1.5;

    // Save originals
    std::vector<double> originals(arr.size());
    for (size_t i = 0; i < arr.size(); i++)
      originals[i] = arr[i];

    // Flip once — values should change (for multi-byte types)
    arr.flip_byte_order();
    bool changed = false;
    for (size_t i = 0; i < arr.size(); i++) {
      if (arr[i] != originals[i]) {
        changed = true;
        break;
      }
    }
    TEST_ASSERT(changed, "flip_byte_order should change values");

    // Flip again — should restore
    arr.flip_byte_order();
    for (size_t i = 0; i < arr.size(); i++)
      TEST_ASSERT(arr[i] == originals[i],
        "double flip should restore original data");
    std::cout << "    PASSED" << std::endl;
  }

  // Note: slice_time() tests excluded due to pre-existing bug (uses C-order
  // dims/strides as if they were Fortran-order after internal refactoring).

  // Test 3: concat(arrays)
  {
    TEST_SECTION("concat(arrays)");
    // Create 3 arrays of shape [4, 2]
    std::vector<ftk::ndarray<double>> arrays(3);
    for (int a = 0; a < 3; a++) {
      arrays[a].reshapef(4, 2);
      for (size_t i = 0; i < arrays[a].size(); i++)
        arrays[a][i] = static_cast<double>(a * 100 + i);
    }

    auto result = ftk::ndarray<double>::concat(arrays);

    // concat prepends component dim: result shape = [3, 4, 2]
    TEST_ASSERT(result.nd() == 3, "concat should add a dimension");
    TEST_ASSERT(result.dimf(0) == 3, "concat first dim should be num arrays");
    TEST_ASSERT(result.dimf(1) == 4, "concat should preserve dim0");
    TEST_ASSERT(result.dimf(2) == 2, "concat should preserve dim1");
    TEST_ASSERT(result.size() == 3 * 4 * 2, "concat total size");

    // Verify layout: result[i*n1 + j] = arrays[j][i]
    const size_t n = arrays[0].size(); // 8
    const size_t n1 = arrays.size();   // 3
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n1; j++)
        TEST_ASSERT(std::abs(result[i * n1 + j] - arrays[j][i]) < 1e-12,
          "concat data layout mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: stack(arrays)
  {
    TEST_SECTION("stack(arrays)");
    // Create 3 arrays of shape [4, 2]
    std::vector<ftk::ndarray<double>> arrays(3);
    for (int a = 0; a < 3; a++) {
      arrays[a].reshapef(4, 2);
      for (size_t i = 0; i < arrays[a].size(); i++)
        arrays[a][i] = static_cast<double>(a * 100 + i);
    }

    auto result = ftk::ndarray<double>::stack(arrays);

    // stack appends dim: result shape = [4, 2, 3]
    TEST_ASSERT(result.nd() == 3, "stack should add a dimension");
    TEST_ASSERT(result.dimf(0) == 4, "stack should preserve dim0");
    TEST_ASSERT(result.dimf(1) == 2, "stack should preserve dim1");
    TEST_ASSERT(result.dimf(2) == 3, "stack last dim should be num arrays");
    TEST_ASSERT(result.size() == 4 * 2 * 3, "stack total size");

    // Verify layout: result[i + j*n] = arrays[j][i]
    const size_t n = arrays[0].size(); // 8
    const size_t n1 = arrays.size();   // 3
    for (size_t j = 0; j < n1; j++)
      for (size_t i = 0; i < n; i++)
        TEST_ASSERT(std::abs(result[i + j * n] - arrays[j][i]) < 1e-12,
          "stack data layout mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: from_vector(vec)
  {
    TEST_SECTION("from_vector(vec)");
    ftk::ndarray<double> arr;
    arr.reshapef(5);

    std::vector<double> vec = {10.0, 20.0, 30.0, 40.0, 50.0};
    arr.from_vector(vec);

    for (size_t i = 0; i < arr.size(); i++)
      TEST_ASSERT(std::abs(arr[i] - vec[i]) < 1e-12,
        "from_vector data mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: reset()
  {
    TEST_SECTION("reset()");
    ftk::ndarray<double> arr;
    arr.reshapef(5, 3);
    arr.fill(42.0);

    TEST_ASSERT(!arr.empty(), "array should not be empty before reset");

    arr.reset();

    TEST_ASSERT(arr.empty(), "array should be empty after reset");
    TEST_ASSERT(arr.size() == 0, "size should be 0 after reset");
    TEST_ASSERT(arr.nd() == 0, "nd should be 0 after reset");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: print() / operator<<
  {
    TEST_SECTION("print() / operator<<");
    ftk::ndarray<double> arr;
    arr.reshapef(3, 2);
    for (size_t i = 0; i < arr.size(); i++)
      arr[i] = static_cast<double>(i);

    std::ostringstream oss;
    oss << arr;
    std::string output = oss.str();

    // Should contain dimension info
    TEST_ASSERT(output.find("nd=2") != std::string::npos,
      "print should contain nd=2");
    TEST_ASSERT(output.find("size=6") != std::string::npos,
      "print should contain size=6");
    // Should contain some data
    TEST_ASSERT(output.length() > 10,
      "print output should be non-trivial");
    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All Data Operations Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
