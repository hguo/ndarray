/**
 * Gradient computation tests for ndarray
 *
 * Tests: gradient2D, gradient3D from grad.hh
 */

#include <ndarray/ndarray.hh>
#include <ndarray/grad.hh>
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
  std::cout << "=== Running Gradient Tests ===" << std::endl << std::endl;

  const double tol = 1e-10;

  // gradient2D uses: grad(0,i,j) = (f(i+1,j) - f(i-1,j)) * (DW-1)
  //                  grad(1,i,j) = (f(i,j+1) - f(i,j-1)) * (DH-1)
  // with boundary clamping

  // Test 1: gradient2D f(i,j) = i
  {
    TEST_SECTION("gradient2D f(i,j) = i");
    const int DW = 10, DH = 8;
    ftk::ndarray<double> scalar;
    scalar.reshapef(DW, DH);
    for (int j = 0; j < DH; j++)
      for (int i = 0; i < DW; i++)
        scalar.f(i, j) = static_cast<double>(i);

    auto grad = ftk::gradient2D(scalar);

    TEST_ASSERT(grad.nd() == 3, "grad should be 3D");
    TEST_ASSERT(grad.dimf(0) == 2, "grad dim0 should be 2 (components)");
    TEST_ASSERT(grad.dimf(1) == DW, "grad dim1 should match DW");
    TEST_ASSERT(grad.dimf(2) == DH, "grad dim2 should match DH");

    // Interior points: grad_x = 2*(DW-1), grad_y = 0
    for (int j = 0; j < DH; j++) {
      for (int i = 1; i < DW - 1; i++) {
        double gx = grad.f(0, i, j);
        double expected_gx = 2.0 * (DW - 1);
        TEST_ASSERT(std::abs(gx - expected_gx) < tol,
          "gradient2D x-component mismatch for f=i");
      }
    }
    // grad_y should be 0 everywhere
    for (int j = 0; j < DH; j++)
      for (int i = 0; i < DW; i++)
        TEST_ASSERT(std::abs(grad.f(1, i, j)) < tol,
          "gradient2D y-component should be 0 for f=i");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: gradient2D f(i,j) = j
  {
    TEST_SECTION("gradient2D f(i,j) = j");
    const int DW = 10, DH = 8;
    ftk::ndarray<double> scalar;
    scalar.reshapef(DW, DH);
    for (int j = 0; j < DH; j++)
      for (int i = 0; i < DW; i++)
        scalar.f(i, j) = static_cast<double>(j);

    auto grad = ftk::gradient2D(scalar);

    // grad_x should be 0 everywhere
    for (int j = 0; j < DH; j++)
      for (int i = 0; i < DW; i++)
        TEST_ASSERT(std::abs(grad.f(0, i, j)) < tol,
          "gradient2D x-component should be 0 for f=j");

    // Interior points: grad_y = 2*(DH-1)
    for (int j = 1; j < DH - 1; j++)
      for (int i = 0; i < DW; i++) {
        double gy = grad.f(1, i, j);
        double expected_gy = 2.0 * (DH - 1);
        TEST_ASSERT(std::abs(gy - expected_gy) < tol,
          "gradient2D y-component mismatch for f=j");
      }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: gradient2D f(i,j) = i + j
  {
    TEST_SECTION("gradient2D f(i,j) = i + j");
    const int DW = 10, DH = 8;
    ftk::ndarray<double> scalar;
    scalar.reshapef(DW, DH);
    for (int j = 0; j < DH; j++)
      for (int i = 0; i < DW; i++)
        scalar.f(i, j) = static_cast<double>(i + j);

    auto grad = ftk::gradient2D(scalar);

    // Interior: grad_x = 2*(DW-1), grad_y = 2*(DH-1)
    for (int j = 1; j < DH - 1; j++) {
      for (int i = 1; i < DW - 1; i++) {
        double gx = grad.f(0, i, j);
        double gy = grad.f(1, i, j);
        TEST_ASSERT(std::abs(gx - 2.0 * (DW - 1)) < tol,
          "gradient2D x mismatch for f=i+j");
        TEST_ASSERT(std::abs(gy - 2.0 * (DH - 1)) < tol,
          "gradient2D y mismatch for f=i+j");
      }
    }
    std::cout << "    PASSED" << std::endl;
  }

  // gradient3D uses: grad(c,i,j,k) = 0.5 * (scalar(i+-1,j,k) - scalar(i-+1,j,k))
  // Only computes at interior points (i=1..DW-2, j=1..DH-2, k=1..DD-2)

  // Test 4: gradient3D f(i,j,k) = i
  {
    TEST_SECTION("gradient3D f(i,j,k) = i");
    const int DW = 8, DH = 6, DD = 5;
    ftk::ndarray<double> scalar;
    scalar.reshapef(DW, DH, DD);
    for (int k = 0; k < DD; k++)
      for (int j = 0; j < DH; j++)
        for (int i = 0; i < DW; i++)
          scalar.f(i, j, k) = static_cast<double>(i);

    auto grad = ftk::gradient3D(scalar);

    TEST_ASSERT(grad.nd() == 4, "grad3D should be 4D");
    TEST_ASSERT(grad.dimf(0) == 3, "grad3D dim0 should be 3 (components)");
    TEST_ASSERT(grad.dimf(1) == DW, "grad3D dim1 should match DW");
    TEST_ASSERT(grad.dimf(2) == DH, "grad3D dim2 should match DH");
    TEST_ASSERT(grad.dimf(3) == DD, "grad3D dim3 should match DD");

    // Interior: grad_x = 0.5 * ((i+1)-(i-1)) = 1.0
    //           grad_y = 0, grad_z = 0
    for (int k = 1; k < DD - 1; k++)
      for (int j = 1; j < DH - 1; j++)
        for (int i = 1; i < DW - 1; i++) {
          TEST_ASSERT(std::abs(grad.f(0, i, j, k) - 1.0) < tol,
            "gradient3D x should be 1.0 for f=i");
          TEST_ASSERT(std::abs(grad.f(1, i, j, k)) < tol,
            "gradient3D y should be 0 for f=i");
          TEST_ASSERT(std::abs(grad.f(2, i, j, k)) < tol,
            "gradient3D z should be 0 for f=i");
        }
    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: gradient3D f(i,j,k) = i + j + k
  {
    TEST_SECTION("gradient3D f(i,j,k) = i + j + k");
    const int DW = 8, DH = 6, DD = 5;
    ftk::ndarray<double> scalar;
    scalar.reshapef(DW, DH, DD);
    for (int k = 0; k < DD; k++)
      for (int j = 0; j < DH; j++)
        for (int i = 0; i < DW; i++)
          scalar.f(i, j, k) = static_cast<double>(i + j + k);

    auto grad = ftk::gradient3D(scalar);

    // Interior: all grad components = 0.5 * 2 = 1.0
    for (int k = 1; k < DD - 1; k++)
      for (int j = 1; j < DH - 1; j++)
        for (int i = 1; i < DW - 1; i++) {
          TEST_ASSERT(std::abs(grad.f(0, i, j, k) - 1.0) < tol,
            "gradient3D x should be 1.0 for f=i+j+k");
          TEST_ASSERT(std::abs(grad.f(1, i, j, k) - 1.0) < tol,
            "gradient3D y should be 1.0 for f=i+j+k");
          TEST_ASSERT(std::abs(grad.f(2, i, j, k) - 1.0) < tol,
            "gradient3D z should be 1.0 for f=i+j+k");
        }
    std::cout << "    PASSED" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "=== All Gradient Tests Passed ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
