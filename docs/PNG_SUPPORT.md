# PNG Support Guide

## Overview

ndarray provides PNG image I/O for grayscale, RGB, and RGBA images. This is useful for:
- Visualizing 2D scientific data
- Processing image data in scientific workflows
- Converting between scientific formats and images

## Building with PNG Support

```bash
cmake .. \
  -DNDARRAY_USE_PNG=ON \
  -DPNG_ROOT=/path/to/libpng  # Optional if not in standard location
make
```

## Supported Formats

| Image Type | ndarray Shape | multicomponents |
|-----------|---------------|-----------------|
| Grayscale | `(height, width)` | No |
| RGB | `(3, height, width)` | Yes |
| RGBA | `(4, height, width)` | Yes |

## API Reference

### Reading PNG Files

```cpp
ftk::ndarray<unsigned char> img;
img.read_png("input.png");

// Query format
if (img.nd() == 2) {
  std::cout << "Grayscale: " << img.dimf(0) << "x" << img.dimf(1) << std::endl;
} else if (img.nd() == 3 && img.multicomponents()) {
  int channels = img.dimf(0);
  int height = img.dimf(1);
  int width = img.dimf(2);
  std::cout << channels << " channels, " << height << "x" << width << std::endl;
}
```

### Writing PNG Files

```cpp
// Grayscale
ftk::ndarray<unsigned char> gray;
gray.reshapef(480, 640);  // height, width
// ... fill with data ...
gray.to_png("output_gray.png");

// RGB
ftk::ndarray<unsigned char> rgb;
rgb.reshapef(3, 480, 640);  // channels, height, width
rgb.set_multicomponents();
// ... fill with data ...
rgb.to_png("output_rgb.png");

// RGBA
ftk::ndarray<unsigned char> rgba;
rgba.reshapef(4, 480, 640);
rgba.set_multicomponents();
// ... fill with data ...
rgba.to_png("output_rgba.png");
```

## Usage Examples

### Example 1: Visualize Scientific Data

Convert a 2D scalar field to grayscale image:

```cpp
#include <ndarray/ndarray.hh>

int main() {
  // Read scientific data
  ftk::ndarray<float> temperature;
  temperature.read_netcdf("simulation.nc", "temperature");  // 2D array

  // Find min/max for normalization
  float min_val = *std::min_element(temperature.p.begin(), temperature.p.end());
  float max_val = *std::max_element(temperature.p.begin(), temperature.p.end());

  // Normalize to [0, 255] and convert to unsigned char
  ftk::ndarray<unsigned char> img;
  img.reshapef(temperature.dimf(0), temperature.dimf(1));

  for (size_t i = 0; i < temperature.size(); i++) {
    float normalized = (temperature[i] - min_val) / (max_val - min_val);
    img[i] = static_cast<unsigned char>(normalized * 255.0f);
  }

  // Save as PNG
  img.to_png("temperature_visualization.png");

  return 0;
}
```

### Example 2: False Color Visualization

Map scalar values to colors:

```cpp
// Simple heat map: blue (cold) -> red (hot)
ftk::ndarray<unsigned char> create_heatmap(const ftk::ndarray<float>& data) {
  // Find range
  float min_val = *std::min_element(data.p.begin(), data.p.end());
  float max_val = *std::max_element(data.p.begin(), data.p.end());

  // Create RGB image
  ftk::ndarray<unsigned char> rgb;
  rgb.reshapef(3, data.dimf(0), data.dimf(1));
  rgb.set_multicomponents();

  for (size_t y = 0; y < data.dimf(0); y++) {
    for (size_t x = 0; x < data.dimf(1); x++) {
      // Normalize to [0, 1]
      float t = (data.at(y, x) - min_val) / (max_val - min_val);

      // Blue -> Cyan -> Green -> Yellow -> Red
      if (t < 0.25f) {
        // Blue -> Cyan
        float s = t / 0.25f;
        rgb.at(0, y, x) = 0;
        rgb.at(1, y, x) = static_cast<unsigned char>(s * 255);
        rgb.at(2, y, x) = 255;
      } else if (t < 0.5f) {
        // Cyan -> Green
        float s = (t - 0.25f) / 0.25f;
        rgb.at(0, y, x) = 0;
        rgb.at(1, y, x) = 255;
        rgb.at(2, y, x) = static_cast<unsigned char>((1 - s) * 255);
      } else if (t < 0.75f) {
        // Green -> Yellow
        float s = (t - 0.5f) / 0.25f;
        rgb.at(0, y, x) = static_cast<unsigned char>(s * 255);
        rgb.at(1, y, x) = 255;
        rgb.at(2, y, x) = 0;
      } else {
        // Yellow -> Red
        float s = (t - 0.75f) / 0.25f;
        rgb.at(0, y, x) = 255;
        rgb.at(1, y, x) = static_cast<unsigned char>((1 - s) * 255);
        rgb.at(2, y, x) = 0;
      }
    }
  }

  return rgb;
}

int main() {
  ftk::ndarray<float> data;
  data.read_netcdf("data.nc", "field");

  auto heatmap = create_heatmap(data);
  heatmap.to_png("heatmap.png");

  return 0;
}
```

### Example 3: Process Image Data

Read image, apply filter, write back:

```cpp
#include <ndarray/ndarray.hh>

ftk::ndarray<unsigned char> gaussian_blur(const ftk::ndarray<unsigned char>& img) {
  // Simple 3x3 Gaussian kernel
  float kernel[3][3] = {
    {1/16.0f, 2/16.0f, 1/16.0f},
    {2/16.0f, 4/16.0f, 2/16.0f},
    {1/16.0f, 2/16.0f, 1/16.0f}
  };

  ftk::ndarray<unsigned char> result;

  if (img.nd() == 2) {
    // Grayscale
    result.reshapef(img.dimf(0), img.dimf(1));

    for (size_t y = 1; y < img.dimf(0) - 1; y++) {
      for (size_t x = 1; x < img.dimf(1) - 1; x++) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            sum += img.at(y + ky, x + kx) * kernel[ky + 1][kx + 1];
          }
        }
        result.at(y, x) = static_cast<unsigned char>(sum);
      }
    }
  } else {
    // RGB - apply to each channel
    result.reshapef(img.dimf(0), img.dimf(1), img.dimf(2));
    result.set_multicomponents();

    for (int c = 0; c < img.dimf(0); c++) {
      for (size_t y = 1; y < img.dimf(1) - 1; y++) {
        for (size_t x = 1; x < img.dimf(2) - 1; x++) {
          float sum = 0.0f;
          for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
              sum += img.at(c, y + ky, x + kx) * kernel[ky + 1][kx + 1];
            }
          }
          result.at(c, y, x) = static_cast<unsigned char>(sum);
        }
      }
    }
  }

  return result;
}

int main() {
  ftk::ndarray<unsigned char> img;
  img.read_png("input.png");

  auto blurred = gaussian_blur(img);
  blurred.to_png("output_blurred.png");

  return 0;
}
```

### Example 4: Alpha Channel for Transparency

Create transparent overlay:

```cpp
ftk::ndarray<unsigned char> create_transparent_overlay(int width, int height) {
  ftk::ndarray<unsigned char> rgba;
  rgba.reshapef(4, height, width);
  rgba.set_multicomponents();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Red color
      rgba.at(0, y, x) = 255;
      rgba.at(1, y, x) = 0;
      rgba.at(2, y, x) = 0;

      // Alpha: radial gradient
      int dx = x - width/2;
      int dy = y - height/2;
      double dist = std::sqrt(dx*dx + dy*dy);
      double alpha = std::min(1.0, dist / (width/2));
      rgba.at(3, y, x) = static_cast<unsigned char>((1 - alpha) * 255);
    }
  }

  return rgba;
}
```

## Type Conversions

PNG files use 8-bit unsigned integers. For other data types:

### Float/Double to PNG

```cpp
// Option 1: Manual normalization
ftk::ndarray<float> float_data;  // Values in [0, 1]
// ... initialize ...

ftk::ndarray<unsigned char> img;
img.reshapef(float_data.dimf(0), float_data.dimf(1));
for (size_t i = 0; i < float_data.size(); i++) {
  img[i] = static_cast<unsigned char>(float_data[i] * 255.0f);
}
img.to_png("output.png");

// Option 2: Auto-scale with min/max
float min_val = *std::min_element(float_data.p.begin(), float_data.p.end());
float max_val = *std::max_element(float_data.p.begin(), float_data.p.end());
for (size_t i = 0; i < float_data.size(); i++) {
  float normalized = (float_data[i] - min_val) / (max_val - min_val);
  img[i] = static_cast<unsigned char>(normalized * 255.0f);
}
```

### PNG to Float/Double

```cpp
ftk::ndarray<unsigned char> img;
img.read_png("input.png");

// Convert to float [0, 1]
ftk::ndarray<float> float_data;
float_data.reshapef(img.dimf(0), img.dimf(1));
for (size_t i = 0; i < img.size(); i++) {
  float_data[i] = img[i] / 255.0f;
}
```

## Integration with Scientific Workflows

### NetCDF -> PNG

```cpp
ftk::ndarray<float> sci_data;
sci_data.read_netcdf("simulation.nc", "temperature");

// Normalize and convert
ftk::ndarray<unsigned char> img;
img.reshapef(sci_data.dimf(0), sci_data.dimf(1));
// ... normalization code ...
img.to_png("visualization.png");
```

### PNG -> HDF5

```cpp
ftk::ndarray<unsigned char> img;
img.read_png("input.png");

// Store as HDF5 (preserves exact values)
img.write_h5("output.h5", "image_data");
```

### Time-Series Visualization

```cpp
#include <ndarray/ndarray_group_stream.hh>

ftk::stream s;
s.parse_yaml("config.yaml");

for (int t = 0; t < s.total_timesteps(); t++) {
  auto g = s.read(t);
  const auto& field = g->get_ref<float>("temperature");

  // Convert to grayscale
  ftk::ndarray<unsigned char> img;
  // ... normalization ...

  // Save frame
  char filename[256];
  snprintf(filename, sizeof(filename), "frame_%04d.png", t);
  img.to_png(filename);
}

// Create video: ffmpeg -i frame_%04d.png -c:v libx264 video.mp4
```

## Error Handling

```cpp
try {
  ftk::ndarray<unsigned char> img;
  img.read_png("input.png");

} catch (const ftk::file_error& e) {
  std::cerr << "Cannot read PNG: " << e.what() << std::endl;
  return 1;
}

try {
  ftk::ndarray<unsigned char> invalid;
  invalid.reshapef(2, 100, 100);  // 2 channels - invalid!
  invalid.set_multicomponents();
  invalid.to_png("output.png");

} catch (const ftk::file_error& e) {
  std::cerr << "Cannot write PNG: " << e.what() << std::endl;
  // PNG requires 1, 3, or 4 channels
}
```

## Performance Notes

- PNG compression is relatively slow (compared to raw binary)
- Use PNG for visualization and final output, not intermediate data
- For large images, consider downsampling first

```cpp
// Downsample 2x before saving
ftk::ndarray<unsigned char> downsample_2x(const ftk::ndarray<unsigned char>& img) {
  size_t new_h = img.dimf(0) / 2;
  size_t new_w = img.dimf(1) / 2;

  ftk::ndarray<unsigned char> result;
  result.reshapef(new_h, new_w);

  for (size_t y = 0; y < new_h; y++) {
    for (size_t x = 0; x < new_w; x++) {
      // Simple average of 2x2 block
      int sum = img.at(y*2, x*2) + img.at(y*2, x*2+1) +
                img.at(y*2+1, x*2) + img.at(y*2+1, x*2+1);
      result.at(y, x) = sum / 4;
    }
  }

  return result;
}
```

## Limitations

- Only 8-bit per channel (no 16-bit PNG support)
- No support for indexed color (palette) images
- No support for grayscale+alpha (2 channel) in write (auto-converted to RGBA)
- No animation (APNG) support

## Testing

Run PNG tests:

```bash
cd build
make test_png
./bin/test_png
```

Creates test images in `test_png_files/` directory.

## See Also

- [ARRAY_ACCESS.md](ARRAY_ACCESS.md) - Accessing array elements
- [BACKENDS.md](BACKENDS.md) - Eigen/xtensor integration
- [GPU_SUPPORT.md](GPU_SUPPORT.md) - GPU visualization workflow
