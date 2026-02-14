#ifndef _NDARRAY_XTENSOR_HH
#define _NDARRAY_XTENSOR_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_XTENSOR

#include <ndarray/ndarray.hh>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

namespace ftk {

/**
 * @brief Convert ndarray to xtensor::xarray
 *
 * Converts an ndarray to an xtensor::xarray. Works with any dimensionality.
 * Creates a copy of the data.
 *
 * @tparam T Element type
 * @param arr Input ndarray
 * @return xt::xarray with the data
 *
 * @code
 * ftk::ndarray<double> arr;
 * arr.reshapef(10, 20, 30);
 * auto xarr = ftk::ndarray_to_xtensor(arr);
 * @endcode
 */
template <typename T>
xt::xarray<T> ndarray_to_xtensor(const ndarray<T>& arr)
{
  // Get shape in Fortran order
  std::vector<size_t> shape;
  for (size_t i = 0; i < arr.nd(); i++) {
    shape.push_back(arr.dimf(i));
  }

  // Create xtensor array with the shape
  xt::xarray<T> result(shape);

  // Copy data element by element
  if (arr.nd() == 1) {
    for (size_t i = 0; i < shape[0]; i++) {
      result(i) = arr.at(i);
    }
  } else if (arr.nd() == 2) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        result(i, j) = arr.at(i, j);
      }
    }
  } else if (arr.nd() == 3) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          result(i, j, k) = arr.at(i, j, k);
        }
      }
    }
  } else if (arr.nd() == 4) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          for (size_t l = 0; l < shape[3]; l++) {
            result(i, j, k, l) = arr.at(i, j, k, l);
          }
        }
      }
    }
  } else {
    throw std::runtime_error("ndarray_to_xtensor: only supports 1D-4D arrays");
  }

  return result;
}

/**
 * @brief Convert xtensor::xarray to ndarray
 *
 * Converts an xtensor::xarray to an ndarray. Works with any dimensionality.
 * Creates a copy of the data.
 *
 * @tparam T Element type
 * @param xarr Input xtensor array
 * @return ndarray with the data (Fortran order)
 *
 * @code
 * xt::xarray<double> xarr = xt::random::randn<double>({10, 20, 30});
 * auto arr = ftk::xtensor_to_ndarray(xarr);
 * @endcode
 */
template <typename T>
ndarray<T> xtensor_to_ndarray(const xt::xarray<T>& xarr)
{
  ndarray<T> arr;

  // Get shape and reshape ndarray
  std::vector<size_t> shape(xarr.shape().begin(), xarr.shape().end());

  if (shape.size() == 1) {
    arr.reshapef(shape[0]);
  } else if (shape.size() == 2) {
    arr.reshapef(shape[0], shape[1]);
  } else if (shape.size() == 3) {
    arr.reshapef(shape[0], shape[1], shape[2]);
  } else if (shape.size() == 4) {
    arr.reshapef(shape[0], shape[1], shape[2], shape[3]);
  } else {
    throw std::runtime_error("xtensor_to_ndarray: only supports 1D-4D arrays");
  }

  // Copy data element by element
  if (shape.size() == 1) {
    for (size_t i = 0; i < shape[0]; i++) {
      arr.at(i) = xarr(i);
    }
  } else if (shape.size() == 2) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        arr.at(i, j) = xarr(i, j);
      }
    }
  } else if (shape.size() == 3) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          arr.at(i, j, k) = xarr(i, j, k);
        }
      }
    }
  } else if (shape.size() == 4) {
    for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
        for (size_t k = 0; k < shape[2]; k++) {
          for (size_t l = 0; l < shape[3]; l++) {
            arr.at(i, j, k, l) = xarr(i, j, k, l);
          }
        }
      }
    }
  }

  return arr;
}

/**
 * @brief Create a zero-copy view of ndarray data as xtensor
 *
 * Creates an xtensor view that references the ndarray's data without copying.
 * The view is only valid as long as the ndarray exists and is not reshaped.
 *
 * @tparam T Element type
 * @param arr Input ndarray
 * @return xt::xarray view referencing the ndarray data
 *
 * @warning The returned view is invalidated if arr is destroyed or reshaped
 *
 * @code
 * ftk::ndarray<double> arr;
 * arr.reshapef(100, 200);
 * auto view = ftk::ndarray_to_xtensor_view(arr);
 * // view references arr's data - no copy
 * @endcode
 */
template <typename T>
auto ndarray_to_xtensor_view(ndarray<T>& arr)
{
  std::vector<size_t> shape;
  for (size_t i = 0; i < arr.nd(); i++) {
    shape.push_back(arr.dimf(i));
  }

  // Create view using xt::adapt
  // Note: xtensor uses row-major (C order) by default, but we specify column-major
  return xt::adapt(arr.data(), arr.size(), xt::no_ownership(), shape, xt::layout_type::column_major);
}

/**
 * @brief Create a const zero-copy view of ndarray data as xtensor
 *
 * Creates a const xtensor view that references the ndarray's data without copying.
 *
 * @tparam T Element type
 * @param arr Input ndarray (const)
 * @return Const xt::xarray view referencing the ndarray data
 *
 * @warning The returned view is invalidated if arr is destroyed or reshaped
 */
template <typename T>
auto ndarray_to_xtensor_view(const ndarray<T>& arr)
{
  std::vector<size_t> shape;
  for (size_t i = 0; i < arr.nd(); i++) {
    shape.push_back(arr.dimf(i));
  }

  return xt::adapt(arr.data(), arr.size(), xt::no_ownership(), shape, xt::layout_type::column_major);
}

} // namespace ftk

#endif // NDARRAY_HAVE_XTENSOR

#endif // _NDARRAY_XTENSOR_HH
