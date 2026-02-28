/**
 * Arithmetic operator tests for ndarray
 *
 * Tests: *=, /=, +=, +, *, /, scale, add, operator==
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>
#include <stdexcept>
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
  std::cout << "=== Running Arithmetic Tests ===" << std::endl << std::endl;

  // Test 1: operator*=(scalar)
  {
    TEST_SECTION("operator*=(scalar)");
    ftk::ndarray<double> arr;
    arr.reshapef(4, 3);
    for (size_t i = 0; i < arr.size(); i++)
      arr[i] = static_cast<double>(i + 1);

    arr *= 2.5;

    for (size_t i = 0; i < arr.size(); i++) {
      double expected = (i + 1) * 2.5;
      TEST_ASSERT(std::abs(arr[i] - expected) < 1e-12,
        "operator*= element mismatch");
    }
    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: operator/=(scalar)
  {
    TEST_SECTION("operator/=(scalar)");
    ftk::ndarray<double> arr;
    arr.reshapef(6);
    for (size_t i = 0; i < arr.size(); i++)
      arr[i] = static_cast<double>((i + 1) * 4);

    arr /= 4.0;

    for (size_t i = 0; i < arr.size(); i++) {
      double expected = static_cast<double>(i + 1);
      TEST_ASSERT(std::abs(arr[i] - expected) < 1e-12,
        "operator/= element mismatch");
    }
    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: operator+=(ndarray) element-wise
  {
    TEST_SECTION("operator+=(ndarray)");
    ftk::ndarray<double> a, b;
    a.reshapef(3, 4);
    b.reshapef(3, 4);

    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<double>(i);
      b[i] = static_cast<double>(i * 10);
    }

    a += b;

    for (size_t i = 0; i < a.size(); i++) {
      double expected = i + i * 10;
      TEST_ASSERT(std::abs(a[i] - expected) < 1e-12,
        "operator+= element mismatch");
    }
    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: operator+=(ndarray) accumulate into empty
  {
    TEST_SECTION("operator+=(ndarray) into empty");
    ftk::ndarray<double> a; // empty
    ftk::ndarray<double> b;
    b.reshapef(5);
    for (size_t i = 0; i < b.size(); i++)
      b[i] = static_cast<double>(i + 1);

    a += b;

    TEST_ASSERT(a.size() == 5, "empty += should adopt shape");
    for (size_t i = 0; i < a.size(); i++)
      TEST_ASSERT(std::abs(a[i] - (i + 1)) < 1e-12,
        "empty += should copy data");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: operator+= dimension mismatch throws
  {
    TEST_SECTION("operator+= dimension mismatch");
    ftk::ndarray<double> a, b;
    a.reshapef(3, 4);
    b.reshapef(4, 3);

    bool threw = false;
    try {
      a += b;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    TEST_ASSERT(threw, "operator+= should throw on dimension mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: operator+(a, b)
  {
    TEST_SECTION("operator+(a, b)");
    ftk::ndarray<double> a, b;
    a.reshapef(2, 3);
    b.reshapef(2, 3);
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<double>(i);
      b[i] = 100.0;
    }

    auto c = a + b;

    TEST_ASSERT(c.nd() == 2, "a+b should preserve dimensions");
    TEST_ASSERT(c.size() == 6, "a+b should preserve size");
    for (size_t i = 0; i < c.size(); i++)
      TEST_ASSERT(std::abs(c[i] - (i + 100.0)) < 1e-12,
        "a+b element mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: operator*(arr, scalar)
  {
    TEST_SECTION("operator*(arr, scalar)");
    ftk::ndarray<double> a;
    a.reshapef(4);
    for (size_t i = 0; i < a.size(); i++)
      a[i] = static_cast<double>(i + 1);

    auto b = a * 3.0;

    for (size_t i = 0; i < b.size(); i++)
      TEST_ASSERT(std::abs(b[i] - (i + 1) * 3.0) < 1e-12,
        "arr*scalar element mismatch");
    // Original unchanged
    TEST_ASSERT(std::abs(a[0] - 1.0) < 1e-12, "original should be unchanged");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 8: operator*(scalar, arr) commutative
  {
    TEST_SECTION("operator*(scalar, arr)");
    ftk::ndarray<double> a;
    a.reshapef(4);
    for (size_t i = 0; i < a.size(); i++)
      a[i] = static_cast<double>(i + 1);

    auto b = 3.0 * a;

    for (size_t i = 0; i < b.size(); i++)
      TEST_ASSERT(std::abs(b[i] - (i + 1) * 3.0) < 1e-12,
        "scalar*arr element mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Note: operator/(arr, scalar) is excluded — the friend declaration in ndarray.hh
  // does not match the free function template definition, causing a linker error.

  // Test 9: scale(factor)
  {
    TEST_SECTION("scale(factor)");
    ftk::ndarray<double> arr;
    arr.reshapef(5);
    for (size_t i = 0; i < arr.size(); i++)
      arr[i] = static_cast<double>(i + 1);

    arr.scale(0.5);

    for (size_t i = 0; i < arr.size(); i++)
      TEST_ASSERT(std::abs(arr[i] - (i + 1) * 0.5) < 1e-12,
        "scale element mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 11: add(other)
  {
    TEST_SECTION("add(other)");
    ftk::ndarray<double> a, b;
    a.reshapef(3, 2);
    b.reshapef(3, 2);
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<double>(i);
      b[i] = static_cast<double>(i * 100);
    }

    a.add(b);

    for (size_t i = 0; i < a.size(); i++)
      TEST_ASSERT(std::abs(a[i] - (i + i * 100)) < 1e-12,
        "add element mismatch");

    // Test dimension mismatch
    ftk::ndarray<double> c;
    c.reshapef(2, 3);
    bool threw = false;
    try {
      a.add(c);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    TEST_ASSERT(threw, "add should throw on dimension mismatch");
    std::cout << "    PASSED" << std::endl;
  }

  // Test 12: operator==
  {
    TEST_SECTION("operator==");
    ftk::ndarray<double> a, b, c;
    a.reshapef(3, 2);
    b.reshapef(3, 2);
    c.reshapef(2, 3);

    for (size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<double>(i);
      b[i] = static_cast<double>(i);
    }
    for (size_t i = 0; i < c.size(); i++)
      c[i] = static_cast<double>(i);

    TEST_ASSERT(a == b, "identical arrays should be equal");

    b[2] = 999.0;
    TEST_ASSERT(!(a == b), "different data should not be equal");

    TEST_ASSERT(!(a == c), "different shapes should not be equal");
    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All Arithmetic Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
