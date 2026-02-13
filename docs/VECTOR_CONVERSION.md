# Converting between std::vector and ndarray

## Problem

Students report it's not easy to convert `std::vector` to `ndarray`. This document provides simple solutions.

## Solution 1: Constructor (1D array)

The simplest way to create a 1D array from a vector:

```cpp
std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
ftk::ndarray<double> arr(vec);  // Creates 1D array with 5 elements

// Access elements
std::cout << arr[0] << std::endl;  // 1.0
std::cout << arr.size() << std::endl;  // 5
```

## Solution 2: Static Factory Method (1D array)

Alternative way using static method:

```cpp
std::vector<float> vec = {10.0f, 20.0f, 30.0f};
auto arr = ftk::ndarray<float>::from_vector_data(vec);

std::cout << arr.size() << std::endl;  // 3
```

## Solution 3: Create Multi-Dimensional Array

Convert vector data to N-D array with specified shape:

```cpp
std::vector<double> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// Create 3x4 2D array
auto arr2d = ftk::ndarray<double>::from_vector_data(vec, {3, 4});
std::cout << arr2d.dimf(0) << " x " << arr2d.dimf(1) << std::endl;  // 3 x 4

// Create 2x3x2 3D array (uses first 12 elements)
auto arr3d = ftk::ndarray<double>::from_vector_data(vec, {2, 3, 2});
```

## Solution 4: Old Method (Still Works)

The original method still works for backward compatibility:

```cpp
std::vector<double> vec = {100.0, 200.0, 300.0};

ftk::ndarray<double> arr;
arr.copy_vector(vec);  // Creates 1D array

std::cout << arr.size() << std::endl;  // 3
```

## Converting Back to std::vector

Get the underlying vector from ndarray:

```cpp
ftk::ndarray<double> arr({3});  // Create array
arr[0] = 1.5;
arr[1] = 2.5;
arr[2] = 3.5;

const std::vector<double>& vec = arr.std_vector();  // Get reference
std::cout << vec[0] << std::endl;  // 1.5
```

**Warning:** This returns a const reference to the internal storage. Don't use it after the array is destroyed.

## Complete Example: MOPS Use Case

Common workflow in MOPS project:

```cpp
#include <ndarray/ndarray.hh>
#include <vector>

int main() {
  // Step 1: Collect data in std::vector (from simulation, file, etc.)
  std::vector<double> velocities;
  for (int i = 0; i < 1000; i++) {
    velocities.push_back(compute_velocity(i));
  }

  // Step 2: Convert to ndarray for processing
  ftk::ndarray<double> vel_array(velocities);

  // Step 3: Reshape if needed (e.g., to 2D grid)
  vel_array.reshapef(10, 100);  // 10 x 100 grid

  // Step 4: Process with ndarray operations
  for (size_t i = 0; i < vel_array.size(); i++) {
    vel_array[i] *= 2.0;  // Scale velocities
  }

  // Step 5: Convert back if needed
  const auto& result = vel_array.std_vector();

  return 0;
}
```

## Comparison Table

| Method | Syntax | Use Case |
|--------|--------|----------|
| Constructor | `ndarray<T> arr(vec)` | Simple 1D conversion |
| Static method (1D) | `ndarray<T>::from_vector_data(vec)` | Factory-style creation |
| Static method (N-D) | `ndarray<T>::from_vector_data(vec, shape)` | Multi-dimensional arrays |
| Old method | `arr.copy_vector(vec)` | Legacy code |
| To vector | `arr.std_vector()` | Get data back |

## Performance Notes

All these methods **copy** the data from `std::vector` to `ndarray`:

```cpp
std::vector<double> vec = {1, 2, 3};
ft::ndarray<double> arr(vec);  // Copies data

// vec and arr are independent
vec[0] = 999;
std::cout << arr[0] << std::endl;  // Still 1 (not 999)
```

If you need zero-copy access to existing ndarray data, use:

```cpp
ndarray_group g = stream.read(t);
const auto& arr = g->get_ref<double>("temperature");  // Zero-copy reference
// Do NOT copy with get_arr() unless you need an independent copy
```

## FAQ

**Q: Can I avoid copying the data?**

A: For construction from `std::vector`, copying is necessary because ndarray needs to own the data. However, when reading from files via `ndarray_group`, use `get_ref()` for zero-copy access.

**Q: What if vector size doesn't match shape?**

A: When using `from_vector_data(vec, shape)`:
- If vec is smaller: remaining array elements are uninitialized
- If vec is larger: only the first N elements are copied

**Q: Can I use this with different types?**

A: Yes, works with any type:
```cpp
std::vector<int> int_vec = {1, 2, 3};
ftk::ndarray<int> int_arr(int_vec);

std::vector<float> float_vec = {1.5f, 2.5f};
ftk::ndarray<float> float_arr(float_vec);

std::vector<uint8_t> byte_vec = {255, 128, 0};
ftk::ndarray<uint8_t> byte_arr(byte_vec);
```

## Best Practices

1. **For simple 1D arrays**: Use constructor
   ```cpp
   ftk::ndarray<double> arr(vec);
   ```

2. **For multi-dimensional arrays**: Use static method with shape
   ```cpp
   auto arr = ftk::ndarray<double>::from_vector_data(vec, {rows, cols});
   ```

3. **For reading from files**: Use zero-copy `get_ref()`
   ```cpp
   const auto& arr = group->get_ref<double>("varname");
   ```

4. **For modifying and returning**: Copy is unavoidable
   ```cpp
   auto result = process_data(input_vec);  // Returns ndarray
   const auto& result_vec = result.std_vector();
   ```

## Migration from Old Code

If students are using manual allocation:

```cpp
// Old way (tedious)
ftk::ndarray<double> arr;
arr.reshapef(vec.size());
for (size_t i = 0; i < vec.size(); i++) {
  arr[i] = vec[i];
}

// New way (simple)
ftk::ndarray<double> arr(vec);
```

Or if using the old `from_vector()` method:

```cpp
// Old way
ftk::ndarray<double> arr;
arr.reshapef(vec.size());
arr.from_vector(vec);  // Requires pre-allocated array

// New way
ftk::ndarray<double> arr(vec);  // All in one step
```
