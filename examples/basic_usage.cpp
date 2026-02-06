#include <ndarray/ndarray.hh>
#include <iostream>

/**
 * Basic usage example for ndarray
 *
 * This example demonstrates:
 * - Creating arrays with different dimensions
 * - Reshaping arrays
 * - Filling arrays with values
 * - Accessing array elements
 * - Basic array properties
 */

int main() {
  std::cout << "=== ndarray Basic Usage Example ===" << std::endl << std::endl;

  // 1. Creating a 1D array
  std::cout << "1. Creating a 1D array:" << std::endl;
  ftk::ndarray<double> arr1d;
  arr1d.reshapef(10);
  arr1d.fill(3.14);
  std::cout << "   - Dimensions: " << arr1d.nd() << std::endl;
  std::cout << "   - Size: " << arr1d.size() << std::endl;
  std::cout << "   - First element: " << arr1d[0] << std::endl << std::endl;

  // 2. Creating a 2D array
  std::cout << "2. Creating a 2D array (5x10):" << std::endl;
  ftk::ndarray<float> arr2d;
  arr2d.reshapef(5, 10);
  std::cout << "   - Dimensions: " << arr2d.nd() << std::endl;
  std::cout << "   - Size: " << arr2d.size() << std::endl;
  std::cout << "   - Shape: " << arr2d.dimf(0) << " x " << arr2d.dimf(1) << std::endl << std::endl;

  // 3. Creating a 3D array
  std::cout << "3. Creating a 3D array (4x5x6):" << std::endl;
  ftk::ndarray<int> arr3d;
  arr3d.reshapef(4, 5, 6);
  std::cout << "   - Dimensions: " << arr3d.nd() << std::endl;
  std::cout << "   - Size: " << arr3d.size() << std::endl;
  std::cout << "   - Shape: " << arr3d.dimf(0) << " x " << arr3d.dimf(1)
            << " x " << arr3d.dimf(2) << std::endl << std::endl;

  // 4. Filling array with different methods
  std::cout << "4. Filling arrays:" << std::endl;
  ftk::ndarray<double> arr;
  arr.reshapef(20);

  // Fill with constant
  arr.fill(1.5);
  std::cout << "   - After fill(1.5): arr[0] = " << arr[0] << std::endl;

  // Manual fill
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = static_cast<double>(i) * 0.1;
  }
  std::cout << "   - After manual fill: arr[10] = " << arr[10] << std::endl << std::endl;

  // 5. Reshaping arrays
  std::cout << "5. Reshaping arrays:" << std::endl;
  ftk::ndarray<double> arr_reshape;
  arr_reshape.reshapef(100);
  std::cout << "   - Initial shape: 1D with size " << arr_reshape.size() << std::endl;

  arr_reshape.reshapef(10, 10);
  std::cout << "   - After reshape to 2D: " << arr_reshape.dimf(0)
            << " x " << arr_reshape.dimf(1) << std::endl;

  arr_reshape.reshapef(5, 4, 5);
  std::cout << "   - After reshape to 3D: " << arr_reshape.dimf(0)
            << " x " << arr_reshape.dimf(1) << " x " << arr_reshape.dimf(2) << std::endl << std::endl;

  // 6. Using vector constructor
  std::cout << "6. Creating array with vector dimensions:" << std::endl;
  std::vector<size_t> dims = {3, 4, 5};
  ftk::ndarray<double> arr_vec(dims);
  std::cout << "   - Created array with dimensions: ";
  for (size_t i = 0; i < arr_vec.nd(); i++) {
    std::cout << arr_vec.dimf(i);
    if (i < arr_vec.nd() - 1) std::cout << " x ";
  }
  std::cout << std::endl << std::endl;

  // 7. Type conversions
  std::cout << "7. Working with different types:" << std::endl;
  ftk::ndarray<double> arr_double;
  arr_double.reshapef(5);
  arr_double.fill(2.5);

  ftk::ndarray<int> arr_int;
  arr_int.reshapef(5);
  arr_int.fill(10);

  std::cout << "   - Double array element: " << arr_double[0] << std::endl;
  std::cout << "   - Int array element: " << arr_int[0] << std::endl << std::endl;

  // 8. Array slicing example
  std::cout << "8. Array slicing:" << std::endl;
  ftk::ndarray<double> large_arr;
  large_arr.reshapef(10, 10);
  for (size_t i = 0; i < large_arr.size(); i++) {
    large_arr[i] = static_cast<double>(i);
  }

  std::vector<size_t> start = {2, 2};
  std::vector<size_t> sizes = {3, 3};
  auto sliced = large_arr.slice(start, sizes);
  std::cout << "   - Original array: " << large_arr.dimf(0) << " x " << large_arr.dimf(1) << std::endl;
  std::cout << "   - Sliced array: " << sliced.dimf(0) << " x " << sliced.dimf(1) << std::endl;
  std::cout << "   - Slice starts at [2,2] with size [3,3]" << std::endl << std::endl;

  std::cout << "=== Example completed successfully ===" << std::endl;
  return 0;
}
