#ifndef __NDARRAY_DEVICE_HH
#define __NDARRAY_DEVICE_HH

#include <ndarray/config.hh>
#include <memory>

#if NDARRAY_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if NDARRAY_HAVE_SYCL
#include <CL/sycl.hpp>
#endif

namespace ftk {

enum {
  NDARRAY_DEVICE_HOST,
  NDARRAY_DEVICE_CUDA,
  NDARRAY_DEVICE_HIP,
  NDARRAY_DEVICE_SYCL
};

/**
 * @brief RAII wrapper for device memory pointers
 *
 * Automatically frees device memory when the object is destroyed,
 * preventing memory leaks. Supports CUDA, HIP, and SYCL backends.
 */
class device_ptr {
public:
  device_ptr() : ptr_(nullptr), device_type_(NDARRAY_DEVICE_HOST), device_id_(0) {}

  ~device_ptr() {
    free();
  }

  // Disable copy, enable move
  device_ptr(const device_ptr&) = delete;
  device_ptr& operator=(const device_ptr&) = delete;

  device_ptr(device_ptr&& other) noexcept
    : ptr_(other.ptr_), device_type_(other.device_type_), device_id_(other.device_id_)
#if NDARRAY_HAVE_SYCL
    , sycl_queue_(other.sycl_queue_)
#endif
  {
    other.ptr_ = nullptr;
    other.device_type_ = NDARRAY_DEVICE_HOST;
  }

  device_ptr& operator=(device_ptr&& other) noexcept {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      device_type_ = other.device_type_;
      device_id_ = other.device_id_;
#if NDARRAY_HAVE_SYCL
      sycl_queue_ = other.sycl_queue_;
#endif
      other.ptr_ = nullptr;
      other.device_type_ = NDARRAY_DEVICE_HOST;
    }
    return *this;
  }

  // Allocate device memory
  void allocate(size_t bytes, int device_type, int device_id = 0) {
    free(); // Free any existing allocation

    device_type_ = device_type;
    device_id_ = device_id;

    if (device_type == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
      cudaSetDevice(device_id);
      cudaMalloc(&ptr_, bytes);
#endif
    } else if (device_type == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
      // For SYCL, allocation requires a queue - defer to allocate_sycl()
      throw std::runtime_error("Use allocate_sycl() for SYCL device allocation");
#endif
    }
  }

#if NDARRAY_HAVE_SYCL
  // SYCL-specific allocation that requires a queue
  void allocate_sycl(size_t bytes, sycl::queue& q, int device_id = 0) {
    free();

    device_type_ = NDARRAY_DEVICE_SYCL;
    device_id_ = device_id;
    sycl_queue_ = &q;

    ptr_ = sycl::malloc_device(bytes, q);
  }
#endif

  // Free device memory
  void free() {
    if (ptr_ != nullptr) {
      if (device_type_ == NDARRAY_DEVICE_CUDA) {
#if NDARRAY_HAVE_CUDA
        cudaSetDevice(device_id_);
        cudaFree(ptr_);
#endif
      } else if (device_type_ == NDARRAY_DEVICE_SYCL) {
#if NDARRAY_HAVE_SYCL
        if (sycl_queue_ != nullptr) {
          sycl::free(ptr_, *sycl_queue_);
        }
#endif
      }
      ptr_ = nullptr;
      device_type_ = NDARRAY_DEVICE_HOST;
    }
  }

  // Access raw pointer
  void* get() { return ptr_; }
  const void* get() const { return ptr_; }

  // Check if allocated
  bool is_allocated() const { return ptr_ != nullptr; }

  // Get device info
  int device_type() const { return device_type_; }
  int device_id() const { return device_id_; }

  // Reset to null without freeing (for cases where ownership is transferred)
  void release() {
    ptr_ = nullptr;
    device_type_ = NDARRAY_DEVICE_HOST;
  }

private:
  void* ptr_;
  int device_type_;
  int device_id_;
#if NDARRAY_HAVE_SYCL
  sycl::queue* sycl_queue_ = nullptr;
#endif
};

}

#endif
