#include <ndarray/ndarray_cuda.hh>
#include <device_launch_parameters.h>

namespace ftk {

// Basic kernels
template <typename T>
__global__ void kernel_fill(T* data, size_t n, T val) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = val;
}

template <typename T>
__global__ void kernel_scale(T* data, size_t n, T factor) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= factor;
}

template <typename T>
__global__ void kernel_add(T* dst, const T* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// Pack/Unpack kernels for ghost exchange
template <typename T>
__global__ void kernel_pack_1d(T* buffer, const T* data, int n, bool is_high, int ghost_width, int core_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ghost_width) {
    int src_idx = is_high ? (core_size - ghost_width + i) : i;
    buffer[i] = data[src_idx];
  }
}

template <typename T>
__global__ void kernel_unpack_1d(T* data, const T* buffer, int n, bool is_high, int ghost_width, int ghost_low, int ghost_high, int core_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ghost_width) {
    int dst_idx = is_high ? (core_size + ghost_low + i) : i;
    data[dst_idx] = buffer[i];
  }
}

// 2D Pack (Fortran order: dim 0 fastest)
template <typename T>
__global__ void kernel_pack_2d(T* buffer, const T* data, int n0, int n1, int dim, bool is_high, int ghost_width, int c0, int c1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (dim == 0) { // Pack along dim 0
    if (i < ghost_width && j < c1) {
      int src_i = is_high ? (c0 - ghost_width + i) : i;
      // Account for ghost_low offset if the array has ghosts in other dims
      // (Simplified: assume core starts at (off0, off1))
      // For now, assume data is the full extent array
      buffer[i + j * ghost_width] = data[src_i + j * n0]; 
    }
  } else { // Pack along dim 1
    if (i < c0 && j < ghost_width) {
      int src_j = is_high ? (c1 - ghost_width + j) : j;
      buffer[i + j * c0] = data[i + src_j * n0];
    }
  }
}

template <typename T>
__global__ void kernel_unpack_2d(T* data, const T* buffer, int n0, int n1, int dim, bool is_high, int ghost_width, int ghost_low, int ghost_high, int c0, int c1) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (dim == 0) {
    if (i < ghost_width && j < c1) {
      int dst_i = is_high ? (c0 + ghost_low + i) : i;
      data[dst_i + j * n0] = buffer[i + j * ghost_width];
    }
  } else {
    if (i < c0 && j < ghost_width) {
      int dst_j = is_high ? (c1 + ghost_low + j) : j;
      data[i + dst_j * n0] = buffer[i + j * c0];
    }
  }
}

template <typename T>
__global__ void kernel_pack_3d(T* buffer, const T* data, int n0, int n1, int n2, int dim, bool is_high, int ghost_width, int c0, int c1, int c2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (dim == 0) {
    if (i < ghost_width && j < c1 && k < c2) {
      int src_i = is_high ? (c0 - ghost_width + i) : i;
      buffer[i + j*ghost_width + k*ghost_width*c1] = data[src_i + j*n0 + k*n0*n1];
    }
  } else if (dim == 1) {
    if (i < c0 && j < ghost_width && k < c2) {
      int src_j = is_high ? (c1 - ghost_width + j) : j;
      buffer[i + j*c0 + k*c0*ghost_width] = data[i + src_j*n0 + k*n0*n1];
    }
  } else {
    if (i < c0 && j < c1 && k < ghost_width) {
      int src_k = is_high ? (c2 - ghost_width + k) : k;
      buffer[i + j*c0 + k*c0*c1] = data[i + j*n0 + src_k*n0*n1];
    }
  }
}

template <typename T>
__global__ void kernel_unpack_3d(T* data, const T* buffer, int n0, int n1, int n2, int dim, bool is_high, int ghost_width, int ghost_low, int ghost_high, int c0, int c1, int c2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (dim == 0) {
    if (i < ghost_width && j < c1 && k < c2) {
      int dst_i = is_high ? (c0 + ghost_low + i) : i;
      data[dst_i + j*n0 + k*n0*n1] = buffer[i + j*ghost_width + k*ghost_width*c1];
    }
  } else if (dim == 1) {
    if (i < c0 && j < ghost_width && k < c2) {
      int dst_j = is_high ? (c1 + ghost_low + j) : j;
      data[i + dst_j*n0 + k*n0*n1] = buffer[i + j*c0 + k*c0*ghost_width];
    }
  } else {
    if (i < c0 && j < c1 && k < ghost_width) {
      int dst_k = is_high ? (c2 + ghost_low + k) : k;
      data[i + j*n0 + dst_k*n0*n1] = buffer[i + j*c0 + k*c0*c1];
    }
  }
}

// Launchers
template <typename T>
void launch_fill(T* data, size_t n, T val) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kernel_fill<<<blocks, threads>>>(data, n, val);
  CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations for launchers (add more as needed)
template void launch_fill<float>(float*, size_t, float);
template void launch_fill<double>(double*, size_t, double);
template void launch_fill<int>(int*, size_t, int);

template <typename T>
void launch_scale(T* data, size_t n, T factor) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kernel_scale<<<blocks, threads>>>(data, n, factor);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_add(T* dst, const T* src, size_t n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  kernel_add<<<blocks, threads>>>(dst, src, n);
  CUDA_CHECK(cudaGetLastError());
}

template void launch_scale<float>(float*, size_t, float);
template void launch_scale<double>(double*, size_t, double);
template void launch_add<float>(float*, const float*, size_t);
template void launch_add<double>(double*, const double*, size_t);

// Pack/Unpack Launchers
template <typename T>
void launch_pack_boundary_1d(T* buffer, const T* data, int n, bool is_high, int ghost_width, int core_size) {
  int threads = 256;
  int blocks = (ghost_width + threads - 1) / threads;
  kernel_pack_1d<<<blocks, threads>>>(buffer, data, n, is_high, ghost_width, core_size);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_unpack_ghost_1d(T* data, const T* buffer, int n, bool is_high, int ghost_width, int ghost_low, int ghost_high, int core_size) {
  int threads = 256;
  int blocks = (ghost_width + threads - 1) / threads;
  kernel_unpack_1d<<<blocks, threads>>>(data, buffer, n, is_high, ghost_width, ghost_low, ghost_high, core_size);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_pack_boundary_2d(T* buffer, const T* data, int n0, int n1, int dim, bool is_high, int ghost_width, int c0, int c1) {
  dim3 threads(16, 16);
  dim3 blocks((dim == 0 ? ghost_width + 15 : c0 + 15) / 16, (dim == 0 ? c1 + 15 : ghost_width + 15) / 16);
  kernel_pack_2d<<<blocks, threads>>>(buffer, data, n0, n1, dim, is_high, ghost_width, c0, c1);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_unpack_ghost_2d(T* data, const T* buffer, int n0, int n1, int dim, bool is_high, int ghost_width, int ghost_low, int ghost_high, int c0, int c1) {
  dim3 threads(16, 16);
  dim3 blocks((dim == 0 ? ghost_width + 15 : c0 + 15) / 16, (dim == 0 ? c1 + 15 : ghost_width + 15) / 16);
  kernel_unpack_2d<<<blocks, threads>>>(data, buffer, n0, n1, dim, is_high, ghost_width, ghost_low, ghost_high, c0, c1);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_pack_boundary_3d(T* buffer, const T* data, int n0, int n1, int n2, int dim, bool is_high, int ghost_width, int c0, int c1, int c2) {
  dim3 threads(8, 8, 4);
  dim3 blocks((dim == 0 ? ghost_width + 7 : c0 + 7) / 8, 
              (dim == 1 ? ghost_width + 7 : c1 + 7) / 8, 
              (dim == 2 ? ghost_width + 3 : c2 + 3) / 4);
  kernel_pack_3d<<<blocks, threads>>>(buffer, data, n0, n1, n2, dim, is_high, ghost_width, c0, c1, c2);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_unpack_ghost_3d(T* data, const T* buffer, int n0, int n1, int n2, int dim, bool is_high, int ghost_width, int ghost_low, int ghost_high, int c0, int c1, int c2) {
  dim3 threads(8, 8, 4);
  dim3 blocks((dim == 0 ? ghost_width + 7 : c0 + 7) / 8, 
              (dim == 1 ? ghost_width + 7 : c1 + 7) / 8, 
              (dim == 2 ? ghost_width + 3 : c2 + 3) / 4);
  kernel_unpack_3d<<<blocks, threads>>>(data, buffer, n0, n1, n2, dim, is_high, ghost_width, ghost_low, ghost_high, c0, c1, c2);
  CUDA_CHECK(cudaGetLastError());
}

// Instantiations for float/double (commonly used in scientific data)
template void launch_pack_boundary_1d<float>(float*, const float*, int, bool, int, int);
template void launch_pack_boundary_1d<double>(double*, const double*, int, bool, int, int);
template void launch_unpack_ghost_1d<float>(float*, const float*, int, bool, int, int, int, int);
template void launch_unpack_ghost_1d<double>(double*, const double*, int, bool, int, int, int, int);

template void launch_pack_boundary_2d<float>(float*, const float*, int, int, int, bool, int, int, int);
template void launch_pack_boundary_2d<double>(double*, const double*, int, int, int, bool, int, int, int);
template void launch_unpack_ghost_2d<float>(float*, const float*, int, int, int, bool, int, int, int, int, int);
template void launch_unpack_ghost_2d<double>(double*, const double*, int, int, int, bool, int, int, int, int, int);

template void launch_pack_boundary_3d<float>(float*, const float*, int, int, int, int, bool, int, int, int, int);
template void launch_pack_boundary_3d<double>(double*, const double*, int, int, int, int, bool, int, int, int, int);
template void launch_unpack_ghost_3d<float>(float*, const float*, int, int, int, int, bool, int, int, int, int, int, int);
template void launch_unpack_ghost_3d<double>(double*, const double*, int, int, int, int, bool, int, int, int, int, int, int);

} // namespace ftk
