/**
 * PNG I/O functionality tests
 *
 * Tests PNG image reading and writing:
 * - Grayscale images
 * - RGB images
 * - RGBA images
 * - Data integrity
 * - Type conversions
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
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
  std::cout << "=== Running PNG Tests ===" << std::endl << std::endl;

#if !NDARRAY_HAVE_PNG
  std::cout << "PNG support not enabled (libpng required)" << std::endl;
  std::cout << "Compile with -DNDARRAY_USE_PNG=ON" << std::endl;
  return 0; // Skip tests, not a failure
#else

  std::cout << "PNG support: ENABLED" << std::endl << std::endl;

  const std::string test_dir = "test_png_files";

  // Create test directory
  system(("mkdir -p " + test_dir).c_str());

  // Test 1: Write and read grayscale image
  {
    TEST_SECTION("Grayscale PNG - write and read");

    const int width = 100;
    const int height = 80;

    // Create grayscale image with gradient
    ftk::ndarray<unsigned char> gray_out;
    gray_out.reshapef(height, width);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Create a gradient pattern
        unsigned char val = static_cast<unsigned char>((x + y) % 256);
        gray_out.at(y, x) = val;
      }
    }

    // Write PNG
    std::string filename = test_dir + "/test_gray.png";
    gray_out.to_png(filename);
    std::cout << "    Wrote: " << filename << std::endl;

    // Read PNG back
    ftk::ndarray<unsigned char> gray_in;
    gray_in.read_png(filename);

    // Verify dimensions
    TEST_ASSERT(gray_in.nd() == 2, "Grayscale should be 2D");
    TEST_ASSERT(gray_in.dimf(0) == height, "Height mismatch");
    TEST_ASSERT(gray_in.dimf(1) == width, "Width mismatch");

    // Verify data
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        TEST_ASSERT(gray_in.at(y, x) == gray_out.at(y, x), "Pixel value mismatch");
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Write and read RGB image
  {
    TEST_SECTION("RGB PNG - write and read");

    const int width = 64;
    const int height = 48;
    const int channels = 3;

    // Create RGB image with color pattern
    ftk::ndarray<unsigned char> rgb_out;
    rgb_out.reshapef(channels, height, width);
    rgb_out.set_multicomponents();

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // R channel: horizontal gradient
        rgb_out.at(0, y, x) = static_cast<unsigned char>((x * 255) / width);
        // G channel: vertical gradient
        rgb_out.at(1, y, x) = static_cast<unsigned char>((y * 255) / height);
        // B channel: checkerboard
        rgb_out.at(2, y, x) = ((x / 8 + y / 8) % 2) ? 255 : 0;
      }
    }

    // Write PNG
    std::string filename = test_dir + "/test_rgb.png";
    rgb_out.to_png(filename);
    std::cout << "    Wrote: " << filename << std::endl;

    // Read PNG back
    ftk::ndarray<unsigned char> rgb_in;
    rgb_in.read_png(filename);

    // Verify dimensions
    TEST_ASSERT(rgb_in.nd() == 3, "RGB should be 3D");
    TEST_ASSERT(rgb_in.multicomponents(), "RGB should be multicomponent");
    TEST_ASSERT(rgb_in.dimf(0) == 3, "Should have 3 channels");
    TEST_ASSERT(rgb_in.dimf(1) == height, "Height mismatch");
    TEST_ASSERT(rgb_in.dimf(2) == width, "Width mismatch");

    // Verify data
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 3; c++) {
          TEST_ASSERT(rgb_in.at(c, y, x) == rgb_out.at(c, y, x),
                      "RGB pixel value mismatch");
        }
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Write and read RGBA image
  {
    TEST_SECTION("RGBA PNG - write and read");

    const int width = 50;
    const int height = 50;
    const int channels = 4;

    // Create RGBA image with transparency
    ftk::ndarray<unsigned char> rgba_out;
    rgba_out.reshapef(channels, height, width);
    rgba_out.set_multicomponents();

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // RGB: solid red
        rgba_out.at(0, y, x) = 255;
        rgba_out.at(1, y, x) = 0;
        rgba_out.at(2, y, x) = 0;

        // Alpha: radial gradient from center
        int dx = x - width/2;
        int dy = y - height/2;
        double dist = std::sqrt(dx*dx + dy*dy);
        double alpha = std::min(1.0, dist / (width/2));
        rgba_out.at(3, y, x) = static_cast<unsigned char>(alpha * 255);
      }
    }

    // Write PNG
    std::string filename = test_dir + "/test_rgba.png";
    rgba_out.to_png(filename);
    std::cout << "    Wrote: " << filename << std::endl;

    // Read PNG back
    ftk::ndarray<unsigned char> rgba_in;
    rgba_in.read_png(filename);

    // Verify dimensions
    TEST_ASSERT(rgba_in.nd() == 3, "RGBA should be 3D");
    TEST_ASSERT(rgba_in.multicomponents(), "RGBA should be multicomponent");
    TEST_ASSERT(rgba_in.dimf(0) == 4, "Should have 4 channels");
    TEST_ASSERT(rgba_in.dimf(1) == height, "Height mismatch");
    TEST_ASSERT(rgba_in.dimf(2) == width, "Width mismatch");

    // Verify data
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          TEST_ASSERT(rgba_in.at(c, y, x) == rgba_out.at(c, y, x),
                      "RGBA pixel value mismatch");
        }
      }
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Type conversion - float to PNG
  {
    TEST_SECTION("Type conversion - float to unsigned char");

    const int width = 32;
    const int height = 32;

    // Create float array with values [0.0, 1.0]
    ftk::ndarray<float> float_arr;
    float_arr.reshapef(height, width);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float_arr.at(y, x) = static_cast<float>(x * y) / (width * height);
      }
    }

    // Convert to [0, 255] range
    ftk::ndarray<unsigned char> char_arr;
    char_arr.reshapef(height, width);
    for (size_t i = 0; i < float_arr.size(); i++) {
      char_arr[i] = static_cast<unsigned char>(float_arr[i] * 255.0f);
    }

    // Write and read
    std::string filename = test_dir + "/test_float_convert.png";
    char_arr.to_png(filename);

    ftk::ndarray<unsigned char> read_arr;
    read_arr.read_png(filename);

    // Verify data integrity
    for (size_t i = 0; i < char_arr.size(); i++) {
      TEST_ASSERT(read_arr[i] == char_arr[i], "Converted data mismatch");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: Type conversion - double to PNG (RGB)
  {
    TEST_SECTION("Type conversion - double RGB to PNG");

    const int width = 40;
    const int height = 30;
    const int channels = 3;

    // Create double RGB array
    ftk::ndarray<double> double_rgb;
    double_rgb.reshapef(channels, height, width);
    double_rgb.set_multicomponents();

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        double_rgb.at(0, y, x) = 255.0 * x / width;
        double_rgb.at(1, y, x) = 255.0 * y / height;
        double_rgb.at(2, y, x) = 128.0;
      }
    }

    // Convert to unsigned char
    ftk::ndarray<unsigned char> char_rgb;
    char_rgb.reshapef(channels, height, width);
    char_rgb.set_multicomponents();
    for (size_t i = 0; i < double_rgb.size(); i++) {
      char_rgb[i] = static_cast<unsigned char>(double_rgb[i]);
    }

    // Write and read
    std::string filename = test_dir + "/test_double_rgb.png";
    char_rgb.to_png(filename);

    ftk::ndarray<unsigned char> read_rgb;
    read_rgb.read_png(filename);

    // Verify
    TEST_ASSERT(read_rgb.dimf(0) == 3, "Channel count mismatch");
    TEST_ASSERT(read_rgb.dimf(1) == height, "Height mismatch");
    TEST_ASSERT(read_rgb.dimf(2) == width, "Width mismatch");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Large image
  {
    TEST_SECTION("Large image (1024x768)");

    const int width = 1024;
    const int height = 768;
    const int channels = 3;

    ftk::ndarray<unsigned char> large_img;
    large_img.reshapef(channels, height, width);
    large_img.set_multicomponents();

    // Create a simple pattern (to avoid spending too much time)
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        large_img.at(0, y, x) = (x * 255) / width;
        large_img.at(1, y, x) = (y * 255) / height;
        large_img.at(2, y, x) = ((x + y) * 255) / (width + height);
      }
    }

    std::string filename = test_dir + "/test_large.png";

    auto start = std::chrono::high_resolution_clock::now();
    large_img.to_png(filename);
    auto end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "    Write time: " << write_time << " ms" << std::endl;

    ftk::ndarray<unsigned char> read_img;
    start = std::chrono::high_resolution_clock::now();
    read_img.read_png(filename);
    end = std::chrono::high_resolution_clock::now();
    auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "    Read time: " << read_time << " ms" << std::endl;

    TEST_ASSERT(read_img.dimf(0) == channels, "Channel mismatch");
    TEST_ASSERT(read_img.dimf(1) == height, "Height mismatch");
    TEST_ASSERT(read_img.dimf(2) == width, "Width mismatch");

    // Spot check a few pixels
    TEST_ASSERT(read_img.at(0, 0, 0) == large_img.at(0, 0, 0), "Corner pixel mismatch");
    TEST_ASSERT(read_img.at(0, height/2, width/2) == large_img.at(0, height/2, width/2),
                "Center pixel mismatch");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 7: Error handling
  {
    TEST_SECTION("Error handling");

    // Test reading non-existent file
    {
      ftk::ndarray<unsigned char> arr;
      bool caught = false;
      try {
        arr.read_png(test_dir + "/nonexistent.png");
      } catch (const ftk::nd::file_error& e) {
        caught = true;
      }
      TEST_ASSERT(caught, "Should throw file_error for non-existent file");
    }

    // Test writing invalid format (wrong number of channels)
    {
      ftk::ndarray<unsigned char> arr;
      arr.reshapef(2, 10, 10);  // 2 channels - invalid for PNG
      arr.set_multicomponents();

      bool caught = false;
      try {
        arr.to_png(test_dir + "/invalid.png");
      } catch (const std::exception& e) {
        caught = true;
      }
      TEST_ASSERT(caught, "Should throw exception for invalid channel count");
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Cleanup
  std::cout << std::endl;
  std::cout << "Test files written to: " << test_dir << "/" << std::endl;
  std::cout << "To clean up: rm -rf " << test_dir << std::endl;

  std::cout << std::endl;
  std::cout << "=== All PNG tests passed ===" << std::endl;

#endif // NDARRAY_HAVE_PNG

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
