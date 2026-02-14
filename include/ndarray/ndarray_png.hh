#ifndef _NDARRAY_PNG_HH
#define _NDARRAY_PNG_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_PNG

#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ftk {
namespace nd {

/**
 * @brief Read PNG image file
 *
 * Reads PNG images as ndarray with shape:
 * - Grayscale: (height, width)
 * - Grayscale + Alpha: (2, height, width) - multicomponent
 * - RGB: (3, height, width) - multicomponent
 * - RGBA: (4, height, width) - multicomponent
 *
 * @param filename Path to PNG file
 * @param data Output data buffer (unsigned char)
 * @param width Output image width
 * @param height Output image height
 * @param channels Output number of channels (1=gray, 2=gray+alpha, 3=RGB, 4=RGBA)
 * @throws file_error if file cannot be opened or read
 */
inline void read_png_file(const std::string& filename,
                          std::vector<unsigned char>& data,
                          int& width, int& height, int& channels)
{
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw file_error(ERR_FILE_CANNOT_OPEN, "Cannot open PNG file: " + filename);
  }

  // Check PNG signature
  unsigned char sig[8];
  if (fread(sig, 1, 8, fp) != 8) {
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "Cannot read PNG signature");
  }

  if (png_sig_cmp(sig, 0, 8)) {
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "Not a valid PNG file: " + filename);
  }

  // Create PNG structures
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "png_create_read_struct failed");
  }

  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "png_create_info_struct failed");
  }

  // Error handling
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "Error reading PNG file");
  }

  // Initialize PNG reading
  png_init_io(png, fp);
  png_set_sig_bytes(png, 8);

  // Read PNG info
  png_read_info(png, info);

  width = png_get_image_width(png, info);
  height = png_get_image_height(png, info);
  png_byte color_type = png_get_color_type(png, info);
  png_byte bit_depth = png_get_bit_depth(png, info);

  // Convert to 8-bit if needed
  if (bit_depth == 16) {
    png_set_strip_16(png);
  }

  // Expand paletted images to RGB
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png);
  }

  // Expand grayscale images with bit depth < 8 to 8 bits
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(png);
  }

  // Add alpha channel if tRNS chunk is present
  if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }

  // Update info after transformations
  png_read_update_info(png, info);
  color_type = png_get_color_type(png, info);

  // Determine number of channels
  switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
      channels = 1;
      break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
      channels = 2;
      break;
    case PNG_COLOR_TYPE_RGB:
      channels = 3;
      break;
    case PNG_COLOR_TYPE_RGBA:
      channels = 4;
      break;
    default:
      png_destroy_read_struct(&png, &info, nullptr);
      fclose(fp);
      throw file_error(ERR_FILE_FORMAT, "Unsupported PNG color type");
  }

  // Allocate memory for image data
  size_t row_bytes = png_get_rowbytes(png, info);
  data.resize(height * row_bytes);

  // Create row pointers
  std::vector<png_bytep> row_pointers(height);
  for (int y = 0; y < height; y++) {
    row_pointers[y] = data.data() + y * row_bytes;
  }

  // Read image data
  png_read_image(png, row_pointers.data());

  // Clean up
  png_destroy_read_struct(&png, &info, nullptr);
  fclose(fp);
}

/**
 * @brief Write PNG image file
 *
 * Writes ndarray to PNG file. Input should have shape:
 * - Grayscale: (height, width) or (1, height, width)
 * - RGB: (3, height, width) - multicomponent
 * - RGBA: (4, height, width) - multicomponent
 *
 * @param filename Output PNG file path
 * @param data Image data (unsigned char)
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (1, 3, or 4)
 * @throws file_error if file cannot be written
 */
inline void write_png_file(const std::string& filename,
                           const unsigned char* data,
                           int width, int height, int channels)
{
  if (channels != 1 && channels != 3 && channels != 4) {
    throw file_error(ERR_FILE_FORMAT,
                    "PNG write supports 1 (gray), 3 (RGB), or 4 (RGBA) channels only");
  }

  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    throw file_error(ERR_FILE_CANNOT_WRITE, "Cannot open file for writing: " + filename);
  }

  // Create PNG structures
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "png_create_write_struct failed");
  }

  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "png_create_info_struct failed");
  }

  // Error handling
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    throw file_error(ERR_FILE_FORMAT, "Error writing PNG file");
  }

  // Initialize PNG writing
  png_init_io(png, fp);

  // Set color type
  int color_type;
  switch (channels) {
    case 1:
      color_type = PNG_COLOR_TYPE_GRAY;
      break;
    case 3:
      color_type = PNG_COLOR_TYPE_RGB;
      break;
    case 4:
      color_type = PNG_COLOR_TYPE_RGBA;
      break;
    default:
      png_destroy_write_struct(&png, &info);
      fclose(fp);
      throw file_error(ERR_FILE_FORMAT, "Invalid channel count");
  }

  // Write PNG header
  png_set_IHDR(png, info, width, height,
               8, // bit depth
               color_type,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  // Write image data
  size_t row_bytes = channels * width;
  std::vector<png_const_bytep> row_pointers(height);
  for (int y = 0; y < height; y++) {
    row_pointers[y] = data + y * row_bytes;
  }

  png_write_image(png, const_cast<png_bytepp>(row_pointers.data()));
  png_write_end(png, nullptr);

  // Clean up
  png_destroy_write_struct(&png, &info);
  fclose(fp);
}

} // namespace nd
} // namespace ftk

#endif // NDARRAY_HAVE_PNG

#endif // _NDARRAY_PNG_HH
