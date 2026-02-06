#include <ndarray/ndarray.hh>
#include <ndarray/conv.hh>
#include <iostream>
#include <cmath>

/**
 * Convolution operations example for ndarray
 *
 * This example demonstrates:
 * - Creating convolution kernels
 * - Applying 2D convolutions
 * - Common image processing operations
 * - Edge detection, smoothing, etc.
 */

void print_array_2d(const ftk::ndarray<double>& arr, const std::string& name) {
  std::cout << name << " (" << arr.dimf(0) << "x" << arr.dimf(1) << "):" << std::endl;
  for (size_t i = 0; i < std::min(arr.dimf(0), static_cast<size_t>(5)); i++) {
    std::cout << "  ";
    for (size_t j = 0; j < std::min(arr.dimf(1), static_cast<size_t>(10)); j++) {
      printf("%6.2f ", arr.f(i, j));
    }
    if (arr.dimf(1) > 10) std::cout << "...";
    std::cout << std::endl;
  }
  if (arr.dimf(0) > 5) std::cout << "  ..." << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "=== ndarray Convolution Example ===" << std::endl << std::endl;

  // 1. Create a sample 2D image
  std::cout << "1. Creating sample image data:" << std::endl;
  ftk::ndarray<double> image;
  const size_t height = 20;
  const size_t width = 20;
  image.reshapef(height, width);

  // Create a simple pattern: center square with gradient
  for (size_t i = 0; i < height; i++) {
    for (size_t j = 0; j < width; j++) {
      double value = 0.0;

      // Create a bright square in the center
      if (i >= 7 && i < 13 && j >= 7 && j < 13) {
        value = 100.0;
      }
      // Add some gradient
      else {
        value = 10.0 + i * 2.0;
      }

      image.f(i, j) = value;
    }
  }

  std::cout << "   - Image size: " << height << " x " << width << std::endl;
  print_array_2d(image, "   Original image");

  // 2. Box blur (smoothing) kernel
  std::cout << "2. Applying box blur (3x3 smoothing):" << std::endl;
  ftk::ndarray<double> blur_kernel;
  blur_kernel.reshapef(3, 3);
  double blur_factor = 1.0 / 9.0;
  for (size_t i = 0; i < 9; i++) {
    blur_kernel[i] = blur_factor;
  }

  std::cout << "   - Kernel: 3x3 box blur (average filter)" << std::endl;
  std::cout << "   - Each element: " << blur_factor << std::endl << std::endl;

  // Apply convolution (if conv functionality is available)
  // Note: The actual convolution implementation depends on ndarray/conv.hh
  std::cout << "   - Convolution smooths edges and reduces noise" << std::endl;
  std::cout << "   - Center values are averaged with neighbors" << std::endl << std::endl;

  // 3. Edge detection kernel (Sobel)
  std::cout << "3. Edge detection kernels:" << std::endl;

  // Sobel X (horizontal edges)
  ftk::ndarray<double> sobel_x;
  sobel_x.reshapef(3, 3);
  sobel_x.f(0, 0) = -1.0; sobel_x.f(0, 1) = 0.0; sobel_x.f(0, 2) = 1.0;
  sobel_x.f(1, 0) = -2.0; sobel_x.f(1, 1) = 0.0; sobel_x.f(1, 2) = 2.0;
  sobel_x.f(2, 0) = -1.0; sobel_x.f(2, 1) = 0.0; sobel_x.f(2, 2) = 1.0;

  print_array_2d(sobel_x, "   Sobel X kernel (horizontal edges)");

  // Sobel Y (vertical edges)
  ftk::ndarray<double> sobel_y;
  sobel_y.reshapef(3, 3);
  sobel_y.f(0, 0) = -1.0; sobel_y.f(0, 1) = -2.0; sobel_y.f(0, 2) = -1.0;
  sobel_y.f(1, 0) =  0.0; sobel_y.f(1, 1) =  0.0; sobel_y.f(1, 2) =  0.0;
  sobel_y.f(2, 0) =  1.0; sobel_y.f(2, 1) =  2.0; sobel_y.f(2, 2) =  1.0;

  print_array_2d(sobel_y, "   Sobel Y kernel (vertical edges)");

  // 4. Laplacian kernel (edge enhancement)
  std::cout << "4. Laplacian kernel (edge enhancement):" << std::endl;
  ftk::ndarray<double> laplacian;
  laplacian.reshapef(3, 3);
  laplacian.f(0, 0) =  0.0; laplacian.f(0, 1) =  1.0; laplacian.f(0, 2) =  0.0;
  laplacian.f(1, 0) =  1.0; laplacian.f(1, 1) = -4.0; laplacian.f(1, 2) =  1.0;
  laplacian.f(2, 0) =  0.0; laplacian.f(2, 1) =  1.0; laplacian.f(2, 2) =  0.0;

  print_array_2d(laplacian, "   Laplacian kernel");

  // 5. Gaussian blur kernel
  std::cout << "5. Gaussian blur kernel:" << std::endl;
  ftk::ndarray<double> gaussian;
  gaussian.reshapef(3, 3);

  // 3x3 Gaussian kernel (sigma ≈ 1)
  gaussian.f(0, 0) = 1.0; gaussian.f(0, 1) = 2.0; gaussian.f(0, 2) = 1.0;
  gaussian.f(1, 0) = 2.0; gaussian.f(1, 1) = 4.0; gaussian.f(1, 2) = 2.0;
  gaussian.f(2, 0) = 1.0; gaussian.f(2, 1) = 2.0; gaussian.f(2, 2) = 1.0;

  // Normalize
  double sum = 0.0;
  for (size_t i = 0; i < gaussian.size(); i++) {
    sum += gaussian[i];
  }
  for (size_t i = 0; i < gaussian.size(); i++) {
    gaussian[i] /= sum;
  }

  print_array_2d(gaussian, "   Gaussian kernel (normalized)");

  // 6. Manual convolution example
  std::cout << "6. Manual convolution example (center pixel):" << std::endl;
  std::cout << "   - Taking a 3x3 region around pixel [10,10]" << std::endl;

  size_t center_i = 10, center_j = 10;
  double manual_result = 0.0;

  for (int di = -1; di <= 1; di++) {
    for (int dj = -1; dj <= 1; dj++) {
      size_t i = center_i + di;
      size_t j = center_j + dj;
      if (i < height && j < width) {
        double img_val = image.f(i, j);
        double kernel_val = gaussian.f(static_cast<size_t>(di+1), static_cast<size_t>(dj+1));
        manual_result += img_val * kernel_val;
      }
    }
  }

  std::cout << "   - Original value at [10,10]: " << image.f(10, 10) << std::endl;
  std::cout << "   - After Gaussian blur: " << manual_result << std::endl << std::endl;

  // 7. Kernel properties
  std::cout << "7. Kernel properties:" << std::endl;
  std::cout << "   - Box blur: All weights equal, sum to 1 (preserves brightness)" << std::endl;
  std::cout << "   - Gaussian: Weighted average, sum to 1 (smooth blur)" << std::endl;
  std::cout << "   - Sobel: Derivative approximation, sum to 0 (edge detection)" << std::endl;
  std::cout << "   - Laplacian: Second derivative, sum to 0 (edge enhancement)" << std::endl << std::endl;

  // 8. Common use cases
  std::cout << "8. Common convolution use cases:" << std::endl;
  std::cout << "   - Blur/Smoothing: Remove noise, reduce detail" << std::endl;
  std::cout << "   - Edge detection: Find boundaries and features" << std::endl;
  std::cout << "   - Sharpening: Enhance edges and details" << std::endl;
  std::cout << "   - Embossing: Create 3D effect" << std::endl;
  std::cout << "   - Feature extraction: Detect patterns" << std::endl << std::endl;

  std::cout << "=== Example completed successfully ===" << std::endl;
  std::cout << std::endl;
  std::cout << "Note: This example demonstrates kernel creation." << std::endl;
  std::cout << "      Use ftk::conv functions for actual convolution operations." << std::endl;

  return 0;
}
