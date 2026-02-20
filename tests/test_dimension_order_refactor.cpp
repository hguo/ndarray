#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>

using namespace ftk;

// Test to verify current behavior before refactoring
void test_current_behavior() {
  std::cout << "=== Testing CURRENT behavior ===" << std::endl;

  // Create 2×3 array
  ndarray<int> arr;
  arr.reshapef({2, 3});

  // Fill with known pattern: value = i0 + i1*10
  for (size_t i0 = 0; i0 < 2; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      arr.f(i0, i1) = i0 + i1 * 10;
    }
  }

  std::cout << "Array (2×3) filled with pattern: value = i0 + i1*10" << std::endl;
  std::cout << "Fortran-order access f(i0,i1):" << std::endl;
  for (size_t i0 = 0; i0 < 2; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      std::cout << "  f(" << i0 << "," << i1 << ") = " << arr.f(i0, i1) << std::endl;
    }
  }

  std::cout << "Linear memory order [0..5]:" << std::endl;
  for (size_t i = 0; i < 6; i++) {
    std::cout << "  [" << i << "] = " << arr[i] << std::endl;
  }

  std::cout << "C-order access c(i0,i1):" << std::endl;
  for (size_t i0 = 0; i0 < 2; i0++) {
    for (size_t i1 = 0; i1 < 3; i1++) {
      std::cout << "  c(" << i0 << "," << i1 << ") = " << arr.c(i0, i1) << std::endl;
    }
  }

  // Check current dims vector
  std::cout << "\nCurrent dims vector: [";
  for (size_t i = 0; i < arr.nd(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << arr.shapef()[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "dimf(0) = " << arr.dimf(0) << ", dimf(1) = " << arr.dimf(1) << std::endl;
  std::cout << "dimc(0) = " << arr.dimc(0) << ", dimc(1) = " << arr.dimc(1) << std::endl;
}

// Test expected behavior after refactoring
void test_expected_behavior() {
  std::cout << "\n=== Expected behavior AFTER refactoring ===" << std::endl;
  std::cout << "After refactoring, dims will be stored in C-order (last varies fastest)" << std::endl;
  std::cout << "But f() and c() APIs remain unchanged!" << std::endl;
  std::cout << "reshapef({2,3}) will internally store dims as [3,2] (C-order)" << std::endl;
  std::cout << "shapef() will return [2,3] (Fortran-order, user-facing)" << std::endl;
  std::cout << "shapec() will return [3,2] (C-order)" << std::endl;
  std::cout << "Memory layout: UNCHANGED (same as before)" << std::endl;
  std::cout << "f(i0,i1) indexing: UNCHANGED (same behavior)" << std::endl;
  std::cout << "c(i0,i1) indexing: UNCHANGED (same behavior)" << std::endl;
}

int main() {
  test_current_behavior();
  test_expected_behavior();

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "This test captures current behavior." << std::endl;
  std::cout << "After refactoring, all APIs must produce identical results!" << std::endl;

  return 0;
}
