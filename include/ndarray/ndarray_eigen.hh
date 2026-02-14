#ifndef _NDARRAY_EIGEN_HH
#define _NDARRAY_EIGEN_HH

#include <ndarray/config.hh>

#if NDARRAY_HAVE_EIGEN

#include <ndarray/ndarray.hh>
#include <Eigen/Dense>

namespace ftk {

/**
 * @brief Convert ndarray to Eigen matrix
 *
 * Converts an ndarray to an Eigen matrix. The ndarray must be 2D (rows x cols).
 * Uses Eigen::Map to avoid copying when possible.
 *
 * @tparam T Element type (must match between ndarray and Eigen matrix)
 * @param arr Input ndarray (must be 2D)
 * @return Eigen::Matrix with the data
 *
 * @code
 * ftk::ndarray<double> arr;
 * arr.reshapef(100, 50);  // 100 rows, 50 cols
 * auto mat = ftk::ndarray_to_eigen(arr);
 * @endcode
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ndarray_to_eigen(const ndarray<T>& arr)
{
  if (arr.nd() != 2) {
    throw std::runtime_error("ndarray_to_eigen: array must be 2D");
  }

  // ndarray uses Fortran order by default: dims[0] = rows, dims[1] = cols
  const size_t rows = arr.dimf(0);
  const size_t cols = arr.dimf(1);

  // Create Eigen matrix and copy data
  // Eigen uses row-major by default, ndarray is column-major (Fortran)
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result(i, j) = arr.at(i, j);
    }
  }

  return result;
}

/**
 * @brief Convert ndarray to Eigen vector
 *
 * Converts a 1D ndarray to an Eigen vector.
 *
 * @tparam T Element type
 * @param arr Input ndarray (must be 1D)
 * @return Eigen::VectorX with the data
 *
 * @code
 * ftk::ndarray<double> arr;
 * arr.reshapef(100);
 * auto vec = ftk::ndarray_to_eigen_vector(arr);
 * @endcode
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
ndarray_to_eigen_vector(const ndarray<T>& arr)
{
  if (arr.nd() != 1) {
    throw std::runtime_error("ndarray_to_eigen_vector: array must be 1D");
  }

  const size_t n = arr.dimf(0);
  Eigen::Matrix<T, Eigen::Dynamic, 1> result(n);

  for (size_t i = 0; i < n; i++) {
    result(i) = arr[i];
  }

  return result;
}

/**
 * @brief Convert Eigen matrix to ndarray
 *
 * Converts an Eigen matrix to a 2D ndarray.
 *
 * @tparam T Element type
 * @param mat Input Eigen matrix
 * @return ndarray with the data (Fortran order)
 *
 * @code
 * Eigen::MatrixXd mat(100, 50);
 * mat.setRandom();
 * auto arr = ftk::eigen_to_ndarray(mat);
 * @endcode
 */
template <typename T>
ndarray<T> eigen_to_ndarray(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
  ndarray<T> arr;
  arr.reshapef(mat.rows(), mat.cols());

  for (size_t i = 0; i < static_cast<size_t>(mat.rows()); i++) {
    for (size_t j = 0; j < static_cast<size_t>(mat.cols()); j++) {
      arr.at(i, j) = mat(i, j);
    }
  }

  return arr;
}

/**
 * @brief Convert Eigen vector to ndarray
 *
 * Converts an Eigen vector to a 1D ndarray.
 *
 * @tparam T Element type
 * @param vec Input Eigen vector
 * @return 1D ndarray with the data
 *
 * @code
 * Eigen::VectorXd vec(100);
 * vec.setRandom();
 * auto arr = ftk::eigen_vector_to_ndarray(vec);
 * @endcode
 */
template <typename T>
ndarray<T> eigen_vector_to_ndarray(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec)
{
  ndarray<T> arr;
  arr.reshapef(vec.size());

  for (size_t i = 0; i < static_cast<size_t>(vec.size()); i++) {
    arr[i] = vec(i);
  }

  return arr;
}

/**
 * @brief Convert Eigen row vector to ndarray
 *
 * Converts an Eigen row vector to a 1D ndarray.
 *
 * @tparam T Element type
 * @param vec Input Eigen row vector
 * @return 1D ndarray with the data
 */
template <typename T>
ndarray<T> eigen_row_vector_to_ndarray(const Eigen::Matrix<T, 1, Eigen::Dynamic>& vec)
{
  ndarray<T> arr;
  arr.reshapef(vec.size());

  for (size_t i = 0; i < static_cast<size_t>(vec.size()); i++) {
    arr[i] = vec(i);
  }

  return arr;
}

} // namespace ftk

#endif // NDARRAY_HAVE_EIGEN

#endif // _NDARRAY_EIGEN_HH
