#include <ndarray/ndarray.hh>
#include <iostream>
#include <iomanip>

using namespace ftk;

void print_array_memory(const ndarray<int>& arr) {
  std::cout << "Memory: [";
  for (size_t i = 0; i < arr.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << arr[i];
  }
  std::cout << "]" << std::endl;
}

void test_2d_column_major() {
  std::cout << "=== Test 2D Column-Major Access (f) ===" << std::endl;

  ndarray<int> arr;
  arr.reshapef(3, 4);  // dims: 3 x 4

  std::cout << "Dimensions: (" << arr.dimf(0) << ", " << arr.dimf(1) << ")" << std::endl;
  std::cout << "Computed strides for column-major: [1, " << arr.dimf(0) << "]" << std::endl;

  // Initialize sequentially in memory
  for (size_t i = 0; i < arr.size(); i++)
    arr[i] = i;

  print_array_memory(arr);

  std::cout << "\nAccess with f(i0, i1) - first index varies fastest:" << std::endl;
  for (size_t i1 = 0; i1 < arr.dimf(1); i1++) {
    for (size_t i0 = 0; i0 < arr.dimf(0); i0++) {
      size_t mem_idx = i0 + i1 * arr.dimf(0);  // column-major formula
      std::cout << "f(" << i0 << "," << i1 << ")=[" << mem_idx << "]=" << arr.f(i0, i1);
      if (i0 < arr.dimf(0) - 1) std::cout << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nConclusion: f() is column-major (Fortran-style)" << std::endl;
  std::cout << std::endl;
}

void test_2d_row_major() {
  std::cout << "=== Test 2D Row-Major Access (c) ===" << std::endl;

  ndarray<int> arr;
  arr.reshapef(3, 4);  // dims: 3 x 4

  std::cout << "Dimensions: (" << arr.dimf(0) << ", " << arr.dimf(1) << ")" << std::endl;
  std::cout << "Note: c(i0, i1) = memory[i1 + i0*" << arr.dimf(0) << "]" << std::endl;

  // Initialize sequentially
  for (size_t i = 0; i < arr.size(); i++)
    arr[i] = i;

  print_array_memory(arr);

  std::cout << "\nAccess with c(i0, i1) - last index varies fastest:" << std::endl;
  // For c() to make sense, we treat it as transposed: (4, 3)
  std::cout << "(Treating as " << arr.dimf(1) << "x" << arr.dimf(0) << " when using c())" << std::endl;
  for (size_t i0 = 0; i0 < arr.dimf(1); i0++) {
    for (size_t i1 = 0; i1 < arr.dimf(0); i1++) {
      size_t mem_idx = i1 + i0 * arr.dimf(0);  // row-major formula with transposed dims
      std::cout << "c(" << i0 << "," << i1 << ")=[" << mem_idx << "]=" << arr.c(i0, i1);
      if (i1 < arr.dimf(0) - 1) std::cout << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nConclusion: c() is row-major (C-style)" << std::endl;
  std::cout << std::endl;
}

void test_3d_ordering() {
  std::cout << "=== Test 3D Ordering ===" << std::endl;

  ndarray<int> arr;
  arr.reshapef(2, 3, 4);  // dims: 2 x 3 x 4

  std::cout << "Dimensions: (" << arr.dimf(0) << ", " << arr.dimf(1) << ", " << arr.dimf(2) << ")" << std::endl;
  size_t s0 = 1, s1 = arr.dimf(0), s2 = arr.dimf(0) * arr.dimf(1);
  std::cout << "Strides for column-major: [" << s0 << ", " << s1 << ", " << s2 << "]" << std::endl;

  // Initialize sequentially
  for (size_t i = 0; i < arr.size(); i++)
    arr[i] = i;

  std::cout << "\nf() access - first index varies fastest:" << std::endl;
  for (size_t i2 = 0; i2 < 2; i2++) {
    std::cout << "i2=" << i2 << ":" << std::endl;
    for (size_t i1 = 0; i1 < 3; i1++) {
      std::cout << "  i1=" << i1 << ": ";
      for (size_t i0 = 0; i0 < 2; i0++) {
        size_t mem_idx = i0 + i1 * s1 + i2 * s2;
        std::cout << "f(" << i0 << "," << i1 << "," << i2 << ")=[" << mem_idx << "]=" << arr.f(i0, i1, i2) << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nc() access - last index varies fastest:" << std::endl;
  for (size_t i0 = 0; i0 < 2; i0++) {
    std::cout << "i0=" << i0 << ":" << std::endl;
    for (size_t i1 = 0; i1 < 2; i1++) {
      std::cout << "  i1=" << i1 << ": ";
      for (size_t i2 = 0; i2 < 3; i2++) {
        size_t mem_idx = i2 + i1 * s1 + i0 * s2;
        std::cout << "c(" << i0 << "," << i1 << "," << i2 << ")=[" << mem_idx << "]=" << arr.c(i0, i1, i2) << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
}

void test_fortran_vs_c() {
  std::cout << "=== Fortran vs C Convention ===" << std::endl;

  ndarray<double> arr;
  arr.reshapef(3, 2);  // 3 x 2 array

  // Initialize using f() column-by-column
  double val = 1.0;
  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < 3; i++) {
      arr.f(i, j) = val;
      val += 1.0;
    }
  }

  std::cout << "Initialized with f() column-by-column:" << std::endl;
  std::cout << "Col 0: f(0,0)=" << arr.f(0,0) << ", f(1,0)=" << arr.f(1,0) << ", f(2,0)=" << arr.f(2,0) << std::endl;
  std::cout << "Col 1: f(0,1)=" << arr.f(0,1) << ", f(1,1)=" << arr.f(1,1) << ", f(2,1)=" << arr.f(2,1) << std::endl;

  std::cout << "\nSame data via c() (transposed view as 2x3):" << std::endl;
  std::cout << "Row 0: c(0,0)=" << arr.c(0,0) << ", c(0,1)=" << arr.c(0,1) << ", c(0,2)=" << arr.c(0,2) << std::endl;
  std::cout << "Row 1: c(1,0)=" << arr.c(1,0) << ", c(1,1)=" << arr.c(1,1) << ", c(1,2)=" << arr.c(1,2) << std::endl;

  std::cout << "\nKey: f(i,j) on (3,2) = c(j,i) on transposed (2,3) view" << std::endl;
  std::cout << std::endl;
}

int main() {
  test_2d_column_major();
  test_2d_row_major();
  test_3d_ordering();
  test_fortran_vs_c();

  std::cout << "=== Summary ===" << std::endl;
  std::cout << "f(): Column-major (Fortran) - first index varies fastest" << std::endl;
  std::cout << "c(): Row-major (C) - last index varies fastest" << std::endl;
  std::cout << "Both access the same memory with different indexing" << std::endl;
  std::cout << "\nDemonstration completed successfully!" << std::endl;

  return 0;
}
