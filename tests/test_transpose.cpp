/**
 * Transpose functionality tests
 *
 * Tests the transpose operation for ndarray:
 * - 2D matrix transpose
 * - N-dimensional tensor permutations
 * - In-place transpose for square matrices
 * - Edge cases (empty, 0D, 1D arrays)
 * - Error handling
 * - Performance (basic validation)
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
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
  std::cout << "=== Running Transpose Tests ===" << std::endl << std::endl;

  // Test 1: Basic 2D transpose
  {
    TEST_SECTION("Basic 2D transpose");
    ftk::ndarray<double> A;
    A.reshapef(3, 4);  // 3x4 matrix

    // Fill with test data: A[i,j] = i * 10 + j
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        A.f(i, j) = i * 10.0 + j;
      }
    }

    auto At = ftk::transpose(A);

    // Check dimensions
    TEST_ASSERT(At.nd() == 2, "Transpose should be 2D");
    TEST_ASSERT(At.dimf(0) == 4, "First dimension should be 4");
    TEST_ASSERT(At.dimf(1) == 3, "Second dimension should be 3");

    // Check values: At[j,i] == A[i,j]
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        TEST_ASSERT(At.f(j, i) == A.f(i, j),
                   "Transpose value mismatch at (" + std::to_string(j) + "," + std::to_string(i) + ")");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Square matrix transpose
  {
    TEST_SECTION("Square matrix transpose");
    ftk::ndarray<int> A;
    A.reshapef(5, 5);

    // Fill with test data
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        A.f(i, j) = static_cast<int>(i * 5 + j);
      }
    }

    auto At = ftk::transpose(A);

    // Verify dimensions
    TEST_ASSERT(At.dimf(0) == 5 && At.dimf(1) == 5, "Should remain 5x5");

    // Verify values
    for (size_t i = 0; i < 5; i++) {
      for (size_t j = 0; j < 5; j++) {
        TEST_ASSERT(At.f(j, i) == A.f(i, j), "Value mismatch");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: In-place transpose for square matrix
  {
    TEST_SECTION("In-place square matrix transpose");
    ftk::ndarray<double> A;
    A.reshapef(4, 4);

    // Fill with test data
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        A.f(i, j) = i * 10.0 + j;
      }
    }

    // Store original for verification
    ftk::ndarray<double> A_orig = A;

    // Transpose in-place
    ftk::transpose_inplace(A);

    // Verify values
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        TEST_ASSERT(A.f(j, i) == A_orig.f(i, j),
                   "In-place transpose value mismatch");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: 3D tensor permutation
  {
    TEST_SECTION("3D tensor permutation");
    ftk::ndarray<float> tensor;
    tensor.reshapef(2, 3, 4);  // shape (2, 3, 4)

    // Fill with unique values
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 4; k++) {
          tensor.f(i, j, k) = i * 100.0f + j * 10.0f + k;
        }
      }
    }

    // Permute: (0,1,2) -> (2,0,1), so shape becomes (4,2,3)
    auto permuted = ftk::transpose(tensor, {2, 0, 1});

    // Check dimensions
    TEST_ASSERT(permuted.nd() == 3, "Should be 3D");
    TEST_ASSERT(permuted.dimf(0) == 4, "First dim should be 4");
    TEST_ASSERT(permuted.dimf(1) == 2, "Second dim should be 2");
    TEST_ASSERT(permuted.dimf(2) == 3, "Third dim should be 3");

    // Check values: permuted[k,i,j] == tensor[i,j,k]
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 4; k++) {
          TEST_ASSERT(std::abs(permuted.f(k, i, j) - tensor.f(i, j, k)) < 1e-6f,
                     "3D permutation value mismatch");
        }
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: 4D tensor permutation
  {
    TEST_SECTION("4D tensor permutation");
    ftk::ndarray<double> tensor4d;
    tensor4d.reshapef(2, 3, 4, 5);  // shape (2, 3, 4, 5)

    // Fill with unique values
    size_t val = 0;
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 4; k++) {
          for (size_t l = 0; l < 5; l++) {
            tensor4d.f(i, j, k, l) = static_cast<double>(val++);
          }
        }
      }
    }

    // Permute: (0,1,2,3) -> (3,1,0,2), so shape becomes (5,3,2,4)
    auto permuted = ftk::transpose(tensor4d, {3, 1, 0, 2});

    // Check dimensions
    TEST_ASSERT(permuted.dimf(0) == 5, "First dim should be 5");
    TEST_ASSERT(permuted.dimf(1) == 3, "Second dim should be 3");
    TEST_ASSERT(permuted.dimf(2) == 2, "Third dim should be 2");
    TEST_ASSERT(permuted.dimf(3) == 4, "Fourth dim should be 4");

    // Check a few values
    TEST_ASSERT(permuted.f(0, 0, 0, 0) == tensor4d.f(0, 0, 0, 0), "Value mismatch at origin");
    TEST_ASSERT(permuted.f(4, 2, 1, 3) == tensor4d.f(1, 2, 3, 4), "Value mismatch at sample point");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Identity permutation (no-op)
  {
    TEST_SECTION("Identity permutation");
    ftk::ndarray<double> A;
    A.reshapef(3, 4, 5);

    // Fill with test data
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    // Identity permutation
    auto B = ftk::transpose(A, {0, 1, 2});

    // Should be identical
    TEST_ASSERT(B.nd() == A.nd(), "Dimensions should match");
    for (size_t i = 0; i < 3; i++) {
      TEST_ASSERT(B.dimf(i) == A.dimf(i), "Dimension size should match");
    }

    for (size_t i = 0; i < A.size(); i++) {
      TEST_ASSERT(B[i] == A[i], "Identity permutation should not change values");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: Edge case - 0D array (scalar)
  {
    TEST_SECTION("Edge case: 0D array");
    ftk::ndarray<double> scalar;
    scalar.reshapef(std::vector<size_t>{});  // 0D
    if (scalar.size() > 0) {
      scalar[0] = 42.0;

      auto result = ftk::transpose(scalar, {});

      TEST_ASSERT(result.nd() == 0, "Should remain 0D");
      if (result.size() > 0) {
        TEST_ASSERT(result[0] == 42.0, "Value should be preserved");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 8: Edge case - 1D array
  {
    TEST_SECTION("Edge case: 1D array");
    ftk::ndarray<int> vec;
    vec.reshapef(10);

    for (size_t i = 0; i < 10; i++) {
      vec[i] = static_cast<int>(i);
    }

    auto result = ftk::transpose(vec, {0});

    TEST_ASSERT(result.nd() == 1, "Should remain 1D");
    TEST_ASSERT(result.dimf(0) == 10, "Size should be preserved");

    for (size_t i = 0; i < 10; i++) {
      TEST_ASSERT(result[i] == vec[i], "Values should be identical");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 9: Edge case - empty array
  {
    TEST_SECTION("Edge case: Empty array");
    ftk::ndarray<double> empty;
    empty.reshapef(0, 5, 3);  // One dimension is zero

    auto result = ftk::transpose(empty, {2, 0, 1});

    TEST_ASSERT(result.dimf(0) == 3, "First dim should be 3");
    TEST_ASSERT(result.dimf(1) == 0, "Second dim should be 0");
    TEST_ASSERT(result.dimf(2) == 5, "Third dim should be 5");
    TEST_ASSERT(result.size() == 0, "Should be empty");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 10: Error handling - invalid axes size
  {
    TEST_SECTION("Error handling: Invalid axes size");
    ftk::ndarray<double> A;
    A.reshapef(3, 4, 5);

    bool caught_error = false;
    try {
      auto B = ftk::transpose(A, {0, 1});  // Wrong size (should be 3)
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for wrong axes size");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 11: Error handling - duplicate axes
  {
    TEST_SECTION("Error handling: Duplicate axes");
    ftk::ndarray<double> A;
    A.reshapef(3, 4, 5);

    bool caught_error = false;
    try {
      auto B = ftk::transpose(A, {0, 1, 1});  // Duplicate axis
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for duplicate axes");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 12: Error handling - out of range axes
  {
    TEST_SECTION("Error handling: Out of range axes");
    ftk::ndarray<double> A;
    A.reshapef(3, 4, 5);

    bool caught_error = false;
    try {
      auto B = ftk::transpose(A, {0, 1, 5});  // Axis 5 is out of range
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for out-of-range axis");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 13: Error handling - in-place on non-square
  {
    TEST_SECTION("Error handling: In-place on non-square");
    ftk::ndarray<double> A;
    A.reshapef(3, 4);  // Non-square

    bool caught_error = false;
    try {
      ftk::transpose_inplace(A);
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for in-place on non-square");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 14: Error handling - in-place on 3D
  {
    TEST_SECTION("Error handling: In-place on 3D");
    ftk::ndarray<double> A;
    A.reshapef(5, 5, 5);  // 3D

    bool caught_error = false;
    try {
      ftk::transpose_inplace(A);
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for in-place on 3D");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 15: Error handling - 2D transpose without axes on non-2D
  {
    TEST_SECTION("Error handling: 2D transpose on 3D array");
    ftk::ndarray<double> A;
    A.reshapef(3, 4, 5);

    bool caught_error = false;
    try {
      auto B = ftk::transpose(A);  // No axes specified, but array is 3D
    } catch (const ftk::invalid_operation& e) {
      caught_error = true;
    }

    TEST_ASSERT(caught_error, "Should throw error for 2D transpose on 3D array");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 16: Larger matrix transpose (performance check)
  {
    TEST_SECTION("Larger matrix transpose (100x100)");
    ftk::ndarray<double> A;
    A.reshapef(100, 100);

    // Fill with test data
    for (size_t i = 0; i < 100; i++) {
      for (size_t j = 0; j < 100; j++) {
        A.f(i, j) = i * 100.0 + j;
      }
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto At = ftk::transpose(A);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "    Time for 100x100 transpose: " << duration.count() << " μs" << std::endl;

    // Verify a few sample values
    TEST_ASSERT(At.f(50, 25) == A.f(25, 50), "Sample value mismatch");
    TEST_ASSERT(At.f(99, 0) == A.f(0, 99), "Edge value mismatch");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 17: Double transpose should give original
  {
    TEST_SECTION("Double transpose identity");
    ftk::ndarray<int> A;
    A.reshapef(4, 6);

    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<int>(i);
    }

    auto At = ftk::transpose(A);
    auto A_restored = ftk::transpose(At);

    // Should match original
    TEST_ASSERT(A_restored.nd() == A.nd(), "Dimensions should match");
    TEST_ASSERT(A_restored.dimf(0) == A.dimf(0), "First dim should match");
    TEST_ASSERT(A_restored.dimf(1) == A.dimf(1), "Second dim should match");

    for (size_t i = 0; i < A.size(); i++) {
      TEST_ASSERT(A_restored[i] == A[i], "Double transpose should restore original");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 18: All 3D permutations (3! = 6)
  {
    TEST_SECTION("All 3D permutations");
    ftk::ndarray<double> A;
    A.reshapef(2, 3, 4);

    // Fill with unique values
    for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 4; k++) {
          A.f(i, j, k) = i * 100.0 + j * 10.0 + k;
        }
      }
    }

    // Test all 6 permutations
    std::vector<std::vector<size_t>> perms = {
      {0, 1, 2},  // identity
      {0, 2, 1},
      {1, 0, 2},
      {1, 2, 0},
      {2, 0, 1},
      {2, 1, 0}
    };

    for (const auto& perm : perms) {
      auto B = ftk::transpose(A, perm);

      // Check dimensions
      TEST_ASSERT(B.dimf(0) == A.dimf(perm[0]), "Permuted dim 0 mismatch");
      TEST_ASSERT(B.dimf(1) == A.dimf(perm[1]), "Permuted dim 1 mismatch");
      TEST_ASSERT(B.dimf(2) == A.dimf(perm[2]), "Permuted dim 2 mismatch");

      // Check a sample value
      TEST_ASSERT(B.f(0, 0, 0) == A.f(0, 0, 0), "Origin value should be preserved");
    }

    std::cout << "    PASSED (all 6 permutations)" << std::endl;
  }

  std::cout << "\n=== All Transpose Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
