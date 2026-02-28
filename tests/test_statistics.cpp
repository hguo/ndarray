/**
 * Statistical function tests for ndarray
 *
 * Tests: min_max, maxabs, resolution
 */

#include <ndarray/ndarray.hh>
#include <iostream>
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
  std::cout << "=== Running Statistics Tests ===" << std::endl << std::endl;

  // Test 1: min_max() with known data
  {
    TEST_SECTION("min_max() known data");
    ftk::ndarray<double> arr;
    arr.reshapef(6);
    arr[0] = 3.0; arr[1] = 1.0; arr[2] = 4.0;
    arr[3] = 1.5; arr[4] = 9.0; arr[5] = 2.6;

    auto [mn, mx] = arr.min_max();
    TEST_ASSERT(std::abs(mn - 1.0) < 1e-12, "min should be 1.0");
    TEST_ASSERT(std::abs(mx - 9.0) < 1e-12, "max should be 9.0");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: min_max() with negatives
  {
    TEST_SECTION("min_max() with negatives");
    ftk::ndarray<double> arr;
    arr.reshapef(5);
    arr[0] = -3.0; arr[1] = -1.0; arr[2] = 2.0;
    arr[3] = 5.0; arr[4] = 0.0;

    auto [mn, mx] = arr.min_max();
    TEST_ASSERT(std::abs(mn - (-3.0)) < 1e-12, "min should be -3.0");
    TEST_ASSERT(std::abs(mx - 5.0) < 1e-12, "max should be 5.0");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: min_max() uniform values
  {
    TEST_SECTION("min_max() uniform");
    ftk::ndarray<double> arr;
    arr.reshapef(10);
    arr.fill(7.7);

    auto [mn, mx] = arr.min_max();
    TEST_ASSERT(std::abs(mn - 7.7) < 1e-12, "uniform min should equal value");
    TEST_ASSERT(std::abs(mx - 7.7) < 1e-12, "uniform max should equal value");
    TEST_ASSERT(std::abs(mn - mx) < 1e-12, "uniform min should equal max");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: maxabs() mixed positive/negative
  {
    TEST_SECTION("maxabs() mixed values");
    ftk::ndarray<double> arr;
    arr.reshapef(5);
    arr[0] = -7.0; arr[1] = 3.0; arr[2] = -2.0;
    arr[3] = 5.0; arr[4] = -1.0;

    double ma = arr.maxabs();
    TEST_ASSERT(std::abs(ma - 7.0) < 1e-12, "maxabs should be 7.0");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: maxabs() all zeros
  {
    TEST_SECTION("maxabs() all zeros");
    ftk::ndarray<double> arr;
    arr.reshapef(8);
    arr.fill(0.0);

    double ma = arr.maxabs();
    TEST_ASSERT(std::abs(ma) < 1e-12, "maxabs of zeros should be 0");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: resolution() known values
  {
    TEST_SECTION("resolution() known values");
    ftk::ndarray<double> arr;
    arr.reshapef(5);
    arr[0] = 10.0; arr[1] = 0.5; arr[2] = 3.0;
    arr[3] = 0.1; arr[4] = 7.0;

    double res = arr.resolution();
    TEST_ASSERT(std::abs(res - 0.1) < 1e-12,
      "resolution should be smallest abs nonzero (0.1)");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: resolution() with zeros
  {
    TEST_SECTION("resolution() with zeros");
    ftk::ndarray<double> arr;
    arr.reshapef(6);
    arr[0] = 0.0; arr[1] = 5.0; arr[2] = 0.0;
    arr[3] = 0.0; arr[4] = 0.25; arr[5] = -3.0;

    double res = arr.resolution();
    TEST_ASSERT(std::abs(res - 0.25) < 1e-12,
      "resolution should skip zeros and find 0.25");
    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All Statistics Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
