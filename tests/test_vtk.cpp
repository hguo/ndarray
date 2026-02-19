/**
 * VTK functionality tests for ndarray
 *
 * Tests VTK I/O operations:
 * - Writing/reading VTK image data (.vti files)
 * - Converting to/from vtkImageData
 * - Converting to/from vtkDataArray
 * - Multi-component arrays
 * - Different data types
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

#if NDARRAY_HAVE_VTK
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
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

#if NDARRAY_HAVE_VTK

int test_vtk_image_data_file_io() {
  TEST_SECTION("VTK ImageData file I/O");

  // Create test data (2D scalar field)
  ftk::ndarray<double> original;
  original.reshapef(10, 20);
  for (size_t i = 0; i < original.size(); i++) {
    original[i] = static_cast<double>(i) * 0.5;
  }

  // Write to VTI file
  try {
    original.to_vtk_image_data_file("test_output.vti", "temperature");
    std::cout << "    - Wrote VTI file" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "    - Write error: " << e.what() << std::endl;
    return 1;
  }

  // Read back
  ftk::ndarray<double> loaded;
  try {
    loaded.read_vtk_image_data_file("test_output.vti", "temperature");
    std::cout << "    - Read VTI file" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "    - Read error: " << e.what() << std::endl;
    return 1;
  }

  // Verify dimensions
  TEST_ASSERT(loaded.nd() == original.nd(), "Dimension count mismatch");
  TEST_ASSERT(loaded.dimf(0) == original.dimf(0), "Dimension 0 mismatch");
  TEST_ASSERT(loaded.dimf(1) == original.dimf(1), "Dimension 1 mismatch");
  TEST_ASSERT(loaded.size() == original.size(), "Total size mismatch");

  // Verify data
  for (size_t i = 0; i < original.size(); i++) {
    TEST_ASSERT(std::abs(loaded[i] - original[i]) < 1e-10, "Data value mismatch");
  }

  std::cout << "    - Data integrity verified" << std::endl;
  return 0;
}

int test_vtk_image_data_3d() {
  TEST_SECTION("VTK ImageData 3D");

  // Create 3D test data
  ftk::ndarray<float> original;
  original.reshapef(5, 6, 7);
  for (size_t k = 0; k < 7; k++) {
    for (size_t j = 0; j < 6; j++) {
      for (size_t i = 0; i < 5; i++) {
        original.f(i, j, k) = static_cast<float>(i + j * 10 + k * 100);
      }
    }
  }

  // Write and read
  original.to_vtk_image_data_file("test_3d.vti", "scalar");
  ftk::ndarray<float> loaded;
  loaded.read_vtk_image_data_file("test_3d.vti", "scalar");

  // Verify
  TEST_ASSERT(loaded.nd() == 3, "3D dimension count mismatch");
  TEST_ASSERT(loaded.dimf(0) == 5, "Dimension 0 mismatch");
  TEST_ASSERT(loaded.dimf(1) == 6, "Dimension 1 mismatch");
  TEST_ASSERT(loaded.dimf(2) == 7, "Dimension 2 mismatch");

  for (size_t k = 0; k < 7; k++) {
    for (size_t j = 0; j < 6; j++) {
      for (size_t i = 0; i < 5; i++) {
        float expected = static_cast<float>(i + j * 10 + k * 100);
        TEST_ASSERT(std::abs(loaded.f(i, j, k) - expected) < 1e-5, "3D data mismatch");
      }
    }
  }

  std::cout << "    - 3D data verified" << std::endl;
  return 0;
}

int test_vtk_data_array_conversion() {
  TEST_SECTION("VTK DataArray conversion");

  // Create ndarray
  ftk::ndarray<double> original;
  original.reshapef(100);
  for (size_t i = 0; i < 100; i++) {
    original[i] = static_cast<double>(i) * 0.1;
  }

  // Convert to vtkDataArray
  vtkSmartPointer<vtkDataArray> vtkArr = original.to_vtk_data_array("test_array");
  TEST_ASSERT(vtkArr != nullptr, "VTK array conversion failed");
  TEST_ASSERT(vtkArr->GetNumberOfTuples() == 100, "VTK array size mismatch");
  TEST_ASSERT(vtkArr->GetNumberOfComponents() == 1, "VTK array components mismatch");

  std::cout << "    - Converted to vtkDataArray" << std::endl;

  // Convert back to ndarray
  auto loaded = ftk::ndarray_base::new_from_vtk_data_array(vtkArr);
  TEST_ASSERT(loaded != nullptr, "Conversion from VTK array failed");
  TEST_ASSERT(loaded->size() == 100, "Loaded array size mismatch");

  // Verify data
  auto loaded_typed = std::dynamic_pointer_cast<ftk::ndarray<double>>(loaded);
  TEST_ASSERT(loaded_typed != nullptr, "Type cast failed");

  for (size_t i = 0; i < 100; i++) {
    TEST_ASSERT(std::abs((*loaded_typed)[i] - original[i]) < 1e-10, "Data mismatch after round-trip");
  }

  std::cout << "    - Round-trip conversion verified" << std::endl;
  return 0;
}

int test_vtk_multicomponent() {
  TEST_SECTION("VTK multi-component arrays");

  // Create 3-component vector field (e.g., velocity)
  ftk::ndarray<float> velocity;
  velocity.reshapef(3, 10, 20);  // 3 components, 10x20 grid

  for (size_t j = 0; j < 20; j++) {
    for (size_t i = 0; i < 10; i++) {
      velocity.f(0, i, j) = static_cast<float>(i);      // vx
      velocity.f(1, i, j) = static_cast<float>(j);      // vy
      velocity.f(2, i, j) = static_cast<float>(i + j);  // vz
    }
  }

  // Write to file
  velocity.to_vtk_image_data_file("test_velocity.vti", "velocity");
  std::cout << "    - Wrote multi-component VTI file" << std::endl;

  // Read back
  ftk::ndarray<float> loaded;
  loaded.read_vtk_image_data_file("test_velocity.vti", "velocity");

  // Verify dimensions
  TEST_ASSERT(loaded.nd() == 3, "Multi-component dimension count mismatch");
  TEST_ASSERT(loaded.dimf(0) == 3, "Component count mismatch");
  TEST_ASSERT(loaded.dimf(1) == 10, "X dimension mismatch");
  TEST_ASSERT(loaded.dimf(2) == 20, "Y dimension mismatch");

  // Verify data
  for (size_t j = 0; j < 20; j++) {
    for (size_t i = 0; i < 10; i++) {
      TEST_ASSERT(std::abs(loaded.f(0, i, j) - static_cast<float>(i)) < 1e-5, "vx mismatch");
      TEST_ASSERT(std::abs(loaded.f(1, i, j) - static_cast<float>(j)) < 1e-5, "vy mismatch");
      TEST_ASSERT(std::abs(loaded.f(2, i, j) - static_cast<float>(i + j)) < 1e-5, "vz mismatch");
    }
  }

  std::cout << "    - Multi-component data verified" << std::endl;
  return 0;
}

int test_vtk_image_data_direct() {
  TEST_SECTION("Direct vtkImageData conversion");

  // Create vtkImageData directly
  vtkSmartPointer<vtkImageData> vtkImg = vtkSmartPointer<vtkImageData>::New();
  vtkImg->SetDimensions(8, 9, 1);
  vtkImg->AllocateScalars(VTK_DOUBLE, 1);

  // Fill with data
  for (int j = 0; j < 9; j++) {
    for (int i = 0; i < 8; i++) {
      double* pixel = static_cast<double*>(vtkImg->GetScalarPointer(i, j, 0));
      *pixel = i * 10.0 + j;
    }
  }

  // Add array name
  vtkImg->GetPointData()->GetScalars()->SetName("test_scalar");

  std::cout << "    - Created vtkImageData" << std::endl;

  // Convert to ndarray
  auto loaded = ftk::ndarray_base::new_from_vtk_image_data(vtkImg, "test_scalar");
  TEST_ASSERT(loaded != nullptr, "Conversion from vtkImageData failed");

  auto loaded_typed = std::dynamic_pointer_cast<ftk::ndarray<double>>(loaded);
  TEST_ASSERT(loaded_typed != nullptr, "Type cast failed");
  TEST_ASSERT(loaded_typed->dimf(0) == 8, "X dimension mismatch");
  TEST_ASSERT(loaded_typed->dimf(1) == 9, "Y dimension mismatch");

  // Verify data
  for (int j = 0; j < 9; j++) {
    for (int i = 0; i < 8; i++) {
      double expected = i * 10.0 + j;
      TEST_ASSERT(std::abs(loaded_typed->f(i, j) - expected) < 1e-10, "Direct conversion data mismatch");
    }
  }

  std::cout << "    - Direct conversion verified" << std::endl;
  return 0;
}

int test_vtk_different_types() {
  TEST_SECTION("VTK with different data types");

  // Test int array
  {
    ftk::ndarray<int> int_array;
    int_array.reshapef(5, 6);
    for (size_t i = 0; i < int_array.size(); i++) {
      int_array[i] = static_cast<int>(i);
    }

    int_array.to_vtk_image_data_file("test_int.vti", "int_data");
    ftk::ndarray<int> loaded_int;
    loaded_int.read_vtk_image_data_file("test_int.vti", "int_data");

    TEST_ASSERT(loaded_int.size() == int_array.size(), "Int array size mismatch");
    for (size_t i = 0; i < int_array.size(); i++) {
      TEST_ASSERT(loaded_int[i] == int_array[i], "Int data mismatch");
    }
    std::cout << "    - Int type verified" << std::endl;
  }

  // Test float array
  {
    ftk::ndarray<float> float_array;
    float_array.reshapef(7, 8);
    for (size_t i = 0; i < float_array.size(); i++) {
      float_array[i] = static_cast<float>(i) * 0.25f;
    }

    float_array.to_vtk_image_data_file("test_float.vti", "float_data");
    ftk::ndarray<float> loaded_float;
    loaded_float.read_vtk_image_data_file("test_float.vti", "float_data");

    TEST_ASSERT(loaded_float.size() == float_array.size(), "Float array size mismatch");
    for (size_t i = 0; i < float_array.size(); i++) {
      TEST_ASSERT(std::abs(loaded_float[i] - float_array[i]) < 1e-6f, "Float data mismatch");
    }
    std::cout << "    - Float type verified" << std::endl;
  }

  return 0;
}

int test_vtk_to_vtk_image_data() {
  TEST_SECTION("to_vtk_image_data method");

  ftk::ndarray<double> arr;
  arr.reshapef(4, 5);
  for (size_t j = 0; j < 5; j++) {
    for (size_t i = 0; i < 4; i++) {
      arr.f(i, j) = i + j * 0.1;
    }
  }

  // Convert to vtkImageData
  vtkSmartPointer<vtkImageData> vtkImg = arr.to_vtk_image_data("test_field");
  TEST_ASSERT(vtkImg != nullptr, "to_vtk_image_data returned null");

  // Check dimensions
  int* dims = vtkImg->GetDimensions();
  TEST_ASSERT(dims[0] == 4, "VTK X dimension mismatch");
  TEST_ASSERT(dims[1] == 5, "VTK Y dimension mismatch");

  // Check data
  vtkDataArray* scalars = vtkImg->GetPointData()->GetScalars();
  TEST_ASSERT(scalars != nullptr, "Scalars not found");
  TEST_ASSERT(scalars->GetNumberOfTuples() == 20, "Number of tuples mismatch");

  for (size_t j = 0; j < 5; j++) {
    for (size_t i = 0; i < 4; i++) {
      double value = scalars->GetTuple1(j * 4 + i);  // VTK uses row-major indexing
      double expected = i + j * 0.1;
      TEST_ASSERT(std::abs(value - expected) < 1e-10, "VTK image data value mismatch");
    }
  }

  std::cout << "    - to_vtk_image_data verified" << std::endl;
  return 0;
}

#endif // NDARRAY_HAVE_VTK

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "=== Running ndarray VTK Tests ===" << std::endl << std::endl;

#if NDARRAY_HAVE_VTK
  int result = 0;

  result |= test_vtk_image_data_file_io();
  result |= test_vtk_image_data_3d();
  result |= test_vtk_data_array_conversion();
  result |= test_vtk_multicomponent();
  result |= test_vtk_image_data_direct();
  result |= test_vtk_different_types();
  result |= test_vtk_to_vtk_image_data();

  if (result == 0) {
    std::cout << std::endl << "=== All VTK tests passed ===" << std::endl;
  } else {
    std::cout << std::endl << "=== Some VTK tests failed ===" << std::endl;
  }

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return result;
#else
  std::cout << "VTK support not enabled. Skipping tests." << std::endl;
  std::cout << "Build with -DNDARRAY_USE_VTK=ON to enable VTK tests." << std::endl;
  return 0;
#endif
}
