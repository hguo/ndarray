/**
 * Storage Backend Comprehensive Tests
 *
 * Tests all storage backend functionality:
 * - Basic operations (resize, fill, indexing) for each backend
 * - Cross-backend conversions (native↔xtensor↔Eigen)
 * - I/O with different storage backends
 * - Groups with different storage backends
 * - Streams with different storage backends
 */

#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_group.hh>
#include <iostream>
#include <cassert>
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
  std::cout << "  " << name << std::endl

// Helper to compare arrays element-wise
template <typename T, typename SP1, typename SP2>
bool arrays_equal(const ftk::ndarray<T, SP1>& a, const ftk::ndarray<T, SP2>& b, T tolerance = T(1e-6)) {
  if (a.size() != b.size()) return false;
  if (a.shapef() != b.shapef()) return false;

  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

int test_native_storage_basic() {
  std::cout << "\n=== Test 1: Native Storage Basic Operations ===" << std::endl;

  TEST_SECTION("Construction and reshaping");
  ftk::ndarray<float, ftk::native_storage> arr;
  arr.reshapef(10, 20);
  TEST_ASSERT(arr.nd() == 2, "Should have 2 dimensions");
  TEST_ASSERT(arr.size() == 200, "Size should be 200");
  TEST_ASSERT(arr.shapef(0) == 10, "First dim should be 10");
  TEST_ASSERT(arr.shapef(1) == 20, "Second dim should be 20");

  TEST_SECTION("Fill operation");
  arr.fill(3.14f);
  TEST_ASSERT(arr[0] == 3.14f, "First element should be 3.14");
  TEST_ASSERT(arr[199] == 3.14f, "Last element should be 3.14");

  TEST_SECTION("Element access and modification");
  arr[50] = 2.71f;
  TEST_ASSERT(arr[50] == 2.71f, "Modified element should be 2.71");

  TEST_SECTION("Array indexing");
  float val = arr.f(5, 10);  // Column-major indexing
  arr.f(5, 10) = 1.41f;
  TEST_ASSERT(arr.f(5, 10) == 1.41f, "Column-major access should work");

  std::cout << "  ✓ All native storage basic tests passed" << std::endl;
  return 0;
}

#if NDARRAY_HAVE_EIGEN
int test_eigen_storage_basic() {
  std::cout << "\n=== Test 2: Eigen Storage Basic Operations ===" << std::endl;

  TEST_SECTION("Construction and reshaping");
  ftk::ndarray<double, ftk::eigen_storage> arr;
  arr.reshapef(15, 25);
  TEST_ASSERT(arr.nd() == 2, "Should have 2 dimensions");
  TEST_ASSERT(arr.size() == 375, "Size should be 375");
  TEST_ASSERT(arr.shapef(0) == 15, "First dim should be 15");
  TEST_ASSERT(arr.shapef(1) == 25, "Second dim should be 25");

  TEST_SECTION("Fill operation");
  arr.fill(2.718);
  TEST_ASSERT(std::abs(arr[0] - 2.718) < 1e-6, "First element should be 2.718");
  TEST_ASSERT(std::abs(arr[374] - 2.718) < 1e-6, "Last element should be 2.718");

  TEST_SECTION("Element access and modification");
  arr[100] = 1.414;
  TEST_ASSERT(std::abs(arr[100] - 1.414) < 1e-6, "Modified element should be 1.414");

  TEST_SECTION("3D array");
  arr.reshapef(5, 6, 7);
  TEST_ASSERT(arr.nd() == 3, "Should have 3 dimensions");
  TEST_ASSERT(arr.size() == 210, "Size should be 210");

  std::cout << "  ✓ All Eigen storage basic tests passed" << std::endl;
  return 0;
}
#endif

#if NDARRAY_HAVE_XTENSOR
int test_xtensor_storage_basic() {
  std::cout << "\n=== Test 3: xtensor Storage Basic Operations ===" << std::endl;

  TEST_SECTION("Construction and reshaping");
  ftk::ndarray<float, ftk::xtensor_storage> arr;
  arr.reshapef(12, 18);
  TEST_ASSERT(arr.nd() == 2, "Should have 2 dimensions");
  TEST_ASSERT(arr.size() == 216, "Size should be 216");

  TEST_SECTION("Fill operation");
  arr.fill(1.732f);
  TEST_ASSERT(std::abs(arr[0] - 1.732f) < 1e-6f, "First element should be 1.732");

  TEST_SECTION("Element access");
  arr[50] = 0.707f;
  TEST_ASSERT(std::abs(arr[50] - 0.707f) < 1e-6f, "Modified element should be 0.707");

  std::cout << "  ✓ All xtensor storage basic tests passed" << std::endl;
  return 0;
}
#endif

int test_cross_backend_conversion() {
  std::cout << "\n=== Test 4: Cross-Backend Conversions ===" << std::endl;

  TEST_SECTION("Native to Native (copy)");
  ftk::ndarray<float, ftk::native_storage> native_arr1;
  native_arr1.reshapef(5, 10);
  for (size_t i = 0; i < native_arr1.size(); i++) {
    native_arr1[i] = static_cast<float>(i) * 0.5f;
  }

  ftk::ndarray<float, ftk::native_storage> native_arr2 = native_arr1;
  TEST_ASSERT(arrays_equal(native_arr1, native_arr2), "Native copy should be identical");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Native to Eigen conversion");
  ftk::ndarray<float, ftk::eigen_storage> eigen_arr = native_arr1;
  TEST_ASSERT(eigen_arr.size() == native_arr1.size(), "Sizes should match after conversion");
  TEST_ASSERT(eigen_arr.shapef() == native_arr1.shapef(), "Shapes should match");
  TEST_ASSERT(arrays_equal(native_arr1, eigen_arr), "Data should be identical after conversion");

  TEST_SECTION("Eigen to Native conversion");
  ftk::ndarray<float, ftk::native_storage> native_arr3 = eigen_arr;
  TEST_ASSERT(arrays_equal(native_arr1, native_arr3), "Round-trip conversion should preserve data");

  TEST_SECTION("Eigen modification and reconversion");
  eigen_arr[0] = 999.0f;
  ftk::ndarray<float, ftk::native_storage> native_arr4 = eigen_arr;
  TEST_ASSERT(native_arr4[0] == 999.0f, "Modified value should survive conversion");
  TEST_ASSERT(native_arr4[1] == native_arr1[1], "Unmodified values should remain unchanged");
#endif

#if NDARRAY_HAVE_XTENSOR
  TEST_SECTION("Native to xtensor conversion");
  ftk::ndarray<float, ftk::xtensor_storage> xtensor_arr = native_arr1;
  TEST_ASSERT(xtensor_arr.size() == native_arr1.size(), "Sizes should match");
  TEST_ASSERT(arrays_equal(native_arr1, xtensor_arr), "Data should match");

  TEST_SECTION("xtensor to Native conversion");
  ftk::ndarray<float, ftk::native_storage> native_arr5 = xtensor_arr;
  TEST_ASSERT(arrays_equal(native_arr1, native_arr5), "Round-trip should work");
#endif

#if NDARRAY_HAVE_EIGEN && NDARRAY_HAVE_XTENSOR
  TEST_SECTION("Eigen to xtensor conversion");
  ftk::ndarray<float, ftk::xtensor_storage> xtensor_arr2 = eigen_arr;
  TEST_ASSERT(arrays_equal(eigen_arr, xtensor_arr2), "Eigen->xtensor conversion should work");
#endif

  std::cout << "  ✓ All cross-backend conversion tests passed" << std::endl;
  return 0;
}

int test_io_with_backends() {
  std::cout << "\n=== Test 5: I/O with Different Backends ===" << std::endl;

  TEST_SECTION("Write with native, read with native");
  ftk::ndarray<float, ftk::native_storage> native_write;
  native_write.reshapef(8, 12);
  for (size_t i = 0; i < native_write.size(); i++) {
    native_write[i] = static_cast<float>(i) * 1.5f;
  }
  native_write.to_binary_file("test_backend_native.bin");

  ftk::ndarray<float, ftk::native_storage> native_read;
  native_read.reshapef(8, 12);
  native_read.read_binary_file("test_backend_native.bin");
  TEST_ASSERT(arrays_equal(native_write, native_read), "Native I/O round-trip should work");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Write with native, read with Eigen");
  ftk::ndarray<float, ftk::eigen_storage> eigen_read;
  eigen_read.reshapef(8, 12);
  eigen_read.read_binary_file("test_backend_native.bin");
  TEST_ASSERT(arrays_equal(native_write, eigen_read), "Cross-backend I/O (native->Eigen) should work");

  TEST_SECTION("Write with Eigen, read with native");
  ftk::ndarray<float, ftk::eigen_storage> eigen_write;
  eigen_write.reshapef(6, 10);
  for (size_t i = 0; i < eigen_write.size(); i++) {
    eigen_write[i] = static_cast<float>(i) * 2.5f;
  }
  eigen_write.to_binary_file("test_backend_eigen.bin");

  ftk::ndarray<float, ftk::native_storage> native_read2;
  native_read2.reshapef(6, 10);
  native_read2.read_binary_file("test_backend_eigen.bin");
  TEST_ASSERT(arrays_equal(eigen_write, native_read2), "Cross-backend I/O (Eigen->native) should work");

  TEST_SECTION("Write with Eigen, read with Eigen");
  ftk::ndarray<float, ftk::eigen_storage> eigen_read2;
  eigen_read2.reshapef(6, 10);
  eigen_read2.read_binary_file("test_backend_eigen.bin");
  TEST_ASSERT(arrays_equal(eigen_write, eigen_read2), "Eigen I/O round-trip should work");
#endif

#if NDARRAY_HAVE_XTENSOR
  TEST_SECTION("Write with xtensor, read with native");
  ftk::ndarray<float, ftk::xtensor_storage> xtensor_write;
  xtensor_write.reshapef(7, 9);
  for (size_t i = 0; i < xtensor_write.size(); i++) {
    xtensor_write[i] = static_cast<float>(i) * 3.5f;
  }
  xtensor_write.to_binary_file("test_backend_xtensor.bin");

  ftk::ndarray<float, ftk::native_storage> native_read3;
  native_read3.reshapef(7, 9);
  native_read3.read_binary_file("test_backend_xtensor.bin");
  TEST_ASSERT(arrays_equal(xtensor_write, native_read3), "Cross-backend I/O (xtensor->native) should work");
#endif

  std::cout << "  ✓ All I/O backend tests passed" << std::endl;
  return 0;
}

int test_groups_with_backends() {
  std::cout << "\n=== Test 6: Groups with Different Backends ===" << std::endl;

  TEST_SECTION("Native storage group");
  ftk::ndarray_group<ftk::native_storage> native_group;

  auto arr1 = std::make_shared<ftk::ndarray<float, ftk::native_storage>>();
  arr1->reshapef(5, 6);
  arr1->fill(1.0f);
  native_group.set("array1", *arr1);

  auto arr2 = std::make_shared<ftk::ndarray<double, ftk::native_storage>>();
  arr2->reshapef(3, 4);
  arr2->fill(2.0);
  native_group.set("array2", *arr2);

  TEST_ASSERT(native_group.size() == 2, "Group should have 2 arrays");

  auto retrieved1 = native_group.get_ptr<float>("array1");
  TEST_ASSERT(retrieved1 != nullptr, "Should retrieve array1");
  TEST_ASSERT(retrieved1->size() == 30, "Retrieved array should have size 30");
  TEST_ASSERT((*retrieved1)[0] == 1.0f, "Retrieved array should have correct values");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Eigen storage group");
  ftk::ndarray_group<ftk::eigen_storage> eigen_group;

  auto eigen_arr1 = std::make_shared<ftk::ndarray<float, ftk::eigen_storage>>();
  eigen_arr1->reshapef(4, 5);
  eigen_arr1->fill(3.14f);
  eigen_group.set("eigen_array1", *eigen_arr1);

  TEST_ASSERT(eigen_group.size() == 1, "Eigen group should have 1 array");

  auto retrieved_eigen = eigen_group.get_ptr<float>("eigen_array1");
  TEST_ASSERT(retrieved_eigen != nullptr, "Should retrieve eigen array");
  TEST_ASSERT(std::abs((*retrieved_eigen)[0] - 3.14f) < 1e-6f, "Retrieved array should have correct value");
#endif

  std::cout << "  ✓ All group backend tests passed" << std::endl;
  return 0;
}

int test_type_conversions() {
  std::cout << "\n=== Test 7: Type Conversions Across Backends ===" << std::endl;

  TEST_SECTION("Float to Double conversion (native)");
  ftk::ndarray<float, ftk::native_storage> float_arr;
  float_arr.reshapef(10);
  for (size_t i = 0; i < 10; i++) {
    float_arr[i] = static_cast<float>(i) * 0.1f;
  }

  ftk::ndarray<double, ftk::native_storage> double_arr;
  double_arr.from_array(float_arr);
  TEST_ASSERT(double_arr.size() == 10, "Size should be preserved");
  TEST_ASSERT(std::abs(double_arr[5] - 0.5) < 1e-6, "Type conversion should work");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Float to Double conversion (Eigen)");
  ftk::ndarray<float, ftk::eigen_storage> float_eigen;
  float_eigen.reshapef(10);
  for (size_t i = 0; i < 10; i++) {
    float_eigen[i] = static_cast<float>(i) * 0.2f;
  }

  ftk::ndarray<double, ftk::eigen_storage> double_eigen;
  double_eigen.from_array(float_eigen);
  TEST_ASSERT(double_eigen.size() == 10, "Size should be preserved");
  TEST_ASSERT(std::abs(double_eigen[5] - 1.0) < 1e-6, "Type conversion should work with Eigen");

  TEST_SECTION("Cross-backend and cross-type conversion");
  ftk::ndarray<double, ftk::native_storage> double_native;
  double_native.from_array(float_eigen);
  TEST_ASSERT(double_native.size() == 10, "Size should be preserved");
  TEST_ASSERT(std::abs(double_native[5] - 1.0) < 1e-6, "Cross-backend + type conversion should work");
#endif

  std::cout << "  ✓ All type conversion tests passed" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     Storage Backend Comprehensive Test Suite              ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

  int result = 0;

  // Test 1: Native storage basic operations
  result |= test_native_storage_basic();

#if NDARRAY_HAVE_EIGEN
  // Test 2: Eigen storage basic operations
  result |= test_eigen_storage_basic();
#else
  std::cout << "\n⊘ Skipping Eigen storage tests (NDARRAY_HAVE_EIGEN not defined)" << std::endl;
#endif

#if NDARRAY_HAVE_XTENSOR
  // Test 3: xtensor storage basic operations
  result |= test_xtensor_storage_basic();
#else
  std::cout << "\n⊘ Skipping xtensor storage tests (NDARRAY_HAVE_XTENSOR not defined)" << std::endl;
#endif

  // Test 4: Cross-backend conversions
  result |= test_cross_backend_conversion();

  // Test 5: I/O with different backends
  result |= test_io_with_backends();

  // Test 6: Groups with different backends
  result |= test_groups_with_backends();

  // Test 7: Type conversions
  result |= test_type_conversions();

  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  if (result == 0) {
    std::cout << "║  ✓✓✓ ALL STORAGE BACKEND TESTS PASSED ✓✓✓                ║" << std::endl;
  } else {
    std::cout << "║  ✗✗✗ SOME TESTS FAILED ✗✗✗                               ║" << std::endl;
  }
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";

  // Cleanup test files
  std::remove("test_backend_native.bin");
  std::remove("test_backend_eigen.bin");
  std::remove("test_backend_xtensor.bin");

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return result;
}
