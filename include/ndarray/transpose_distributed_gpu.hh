#ifndef _NDARRAY_TRANSPOSE_DISTRIBUTED_GPU_HH
#define _NDARRAY_TRANSPOSE_DISTRIBUTED_GPU_HH

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_CUDA

#include <ndarray/ndarray.hh>
#include <ndarray/transpose_distributed.hh>
#include <ndarray/transpose_cuda.hh>
#include <ndarray/error.hh>
#include <mpi.h>
#include <cuda_runtime.h>

namespace ftk {
namespace detail {

/**
 * @brief Check if MPI implementation supports GPU-aware communication
 *
 * GPU-aware MPI allows passing device pointers directly to MPI functions.
 * This is much more efficient than staging through CPU memory.
 *
 * Supported by:
 * - OpenMPI with CUDA support
 * - MVAPICH2-GDR
 * - Cray MPI
 * - IBM Spectrum MPI
 */
inline bool is_cuda_aware_mpi() {
  // Check for MPIX_CUDA_AWARE_SUPPORT (if available)
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  return true;
#elif defined(OPEN_MPI) && defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
  return true;
#else
  // Runtime check: Try to get attribute (may not be portable)
  // Conservative: return false and use CPU staging
  return false;
#endif
}

/**
 * @brief Pack data from GPU array for MPI send (GPU version)
 *
 * Extracts data from specified region and transposes it on GPU before
 * packing for communication.
 *
 * @param input Input array on GPU
 * @param global_region_original Region to extract (original coordinates)
 * @param axes Transpose permutation
 * @param d_buffer Device buffer for packed data
 * @param h_buffer Host buffer (for CPU staging if needed)
 * @param use_gpu_direct If true, pack to device buffer; if false, to host
 */
template <typename T, typename StoragePolicy>
void pack_transposed_region_gpu(const ndarray<T, StoragePolicy>& input,
                                 const lattice& global_region_original,
                                 const std::vector<size_t>& axes,
                                 T* d_buffer,
                                 T* h_buffer,
                                 bool use_gpu_direct) {
  const size_t nd = input.nd();
  const size_t n_elems = global_region_original.n();

  if (n_elems == 0) return;

  // Create temporary array for the region on GPU
  std::vector<size_t> region_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    region_dims[i] = global_region_original.size(i);
  }

  ndarray<T, StoragePolicy> region;
  region.reshapef(region_dims);
  region.to_device(NDARRAY_DEVICE_CUDA, input.get_device_id());

  // Extract region from input (on GPU)
  const T* d_input = static_cast<const T*>(input.get_devptr());
  T* d_region = static_cast<T*>(region.get_devptr());

  // Copy region elements (could optimize with custom CUDA kernel)
  std::vector<size_t> idx(nd);
  const lattice& local_core = input.local_core();

  // For now, stage through CPU for extraction
  // TODO: Optimize with GPU kernel for extraction
  std::vector<T> temp_extract(n_elems);
  size_t extract_idx = 0;

  for (size_t linear = 0; linear < n_elems; linear++) {
    size_t tmp = linear;
    for (size_t d = 0; d < nd; d++) {
      idx[d] = global_region_original.start(d) + (tmp % global_region_original.size(d));
      tmp /= global_region_original.size(d);
    }

    if (input.is_local(idx)) {
      auto local_idx = input.global_to_local(idx);
      // Would need to access GPU memory here - for now stage through CPU
    }
  }

  // Alternative: Copy entire region to host, pack, and optionally copy back
  // This is the CPU staging path
  if (!use_gpu_direct) {
    // Pack on CPU
    pack_transposed_region(input, global_region_original, axes,
                          std::vector<T>(h_buffer, h_buffer + n_elems));
  } else {
    // Pack to GPU buffer (more complex, needs GPU kernel)
    // For now, pack on CPU and copy to GPU
    std::vector<T> temp_buffer;
    pack_transposed_region(input, global_region_original, axes, temp_buffer);
    CUDA_CHECK(cudaMemcpy(d_buffer, temp_buffer.data(),
                         n_elems * sizeof(T), cudaMemcpyHostToDevice));
  }
}

/**
 * @brief Transpose distributed array with GPU acceleration
 *
 * Handles transpose of arrays that are both:
 * - Distributed across MPI ranks
 * - Stored on GPU devices
 *
 * Two modes:
 * 1. GPU-aware MPI: Direct GPU-to-GPU communication
 * 2. CPU staging: GPU → CPU → MPI → CPU → GPU
 *
 * @param input Distributed array on GPU
 * @param axes Transpose permutation
 * @return Transposed distributed array on GPU
 */
template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> transpose_distributed_gpu(const ndarray<T, StoragePolicy>& input,
                                                     const std::vector<size_t>& axes) {
  const size_t nd = input.nd();

  // Validate
  validate_distributed_transpose(input, axes);

  // Verify input is on GPU
  if (!input.is_on_device() || input.get_device_type() != NDARRAY_DEVICE_CUDA) {
    throw device_error(ERR_NOT_BUILT_WITH_CUDA,
                      "transpose_distributed_gpu: input must be on CUDA device");
  }

  // Get distribution info
  const lattice& old_global = input.global_lattice();
  const lattice& old_local_core = input.local_core();
  const lattice_partitioner& old_part = input.partitioner();
  const std::vector<size_t>& old_decomp = input.decomp_pattern();
  const std::vector<size_t>& old_ghost = input.ghost_widths();

  MPI_Comm comm = input.comm();
  int rank = input.rank();
  int nprocs = input.nprocs();
  int device_id = input.get_device_id();

  // Compute new distribution
  std::vector<size_t> new_global_dims(nd);
  std::vector<size_t> new_decomp(nd);
  std::vector<size_t> new_ghost(nd);

  for (size_t i = 0; i < nd; i++) {
    new_global_dims[i] = old_global.size(axes[i]);
    new_decomp[i] = old_decomp[axes[i]];
    new_ghost[i] = old_ghost[axes[i]];
  }

  // Create output array with new distribution
  ndarray<T, StoragePolicy> output;
  output.decompose(comm, new_global_dims, nprocs, new_decomp, new_ghost);

  const lattice& new_local_core = output.local_core();
  const lattice_partitioner& new_part = output.partitioner();

  // Move output to same GPU
  output.to_device(NDARRAY_DEVICE_CUDA, device_id);

  // Determine if we can use GPU-aware MPI
  const bool use_cuda_aware = is_cuda_aware_mpi();

  if (!use_cuda_aware) {
    // CPU staging path: This is the safe, compatible approach
    // 1. Copy input to CPU
    // 2. Use regular distributed transpose
    // 3. Copy result back to GPU

    ndarray<T, StoragePolicy> input_cpu = input;
    input_cpu.to_host();

    auto output_cpu = transpose_distributed(input_cpu, axes);

    // Copy back to GPU
    const size_t n_elems = output_cpu.nelem();
    T* d_output = static_cast<T*>(output.get_devptr());

    CUDA_CHECK(cudaMemcpy(d_output, output_cpu.data(),
                         n_elems * sizeof(T), cudaMemcpyHostToDevice));

    // Copy metadata
    output.set_multicomponents(input.multicomponents());
    output.set_has_time(input.has_time());

    return output;
  }

  // GPU-aware MPI path: Direct GPU-to-GPU communication
  // This is more efficient but requires GPU-aware MPI

  std::vector<std::vector<T>> send_buffers_cpu(nprocs);  // Fallback
  std::vector<std::vector<T>> recv_buffers_cpu(nprocs);
  std::vector<T*> send_buffers_gpu(nprocs, nullptr);
  std::vector<T*> recv_buffers_gpu(nprocs, nullptr);
  std::vector<size_t> send_counts(nprocs, 0);
  std::vector<size_t> recv_counts(nprocs, 0);
  std::vector<lattice> recv_regions(nprocs);

  // Phase 1: Compute send/recv sizes
  for (int target_rank = 0; target_rank < nprocs; target_rank++) {
    const lattice& target_new_core = new_part.get_core(target_rank);
    lattice target_needs_old = inverse_permute_lattice(target_new_core, axes);
    lattice send_region = compute_intersection(old_local_core, target_needs_old);

    if (send_region.n() > 0) {
      send_counts[target_rank] = send_region.n();
    }
  }

  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    const lattice& source_old_core = old_part.get_core(source_rank);
    lattice i_need_old = inverse_permute_lattice(new_local_core, axes);
    lattice recv_region_old = compute_intersection(source_old_core, i_need_old);

    if (recv_region_old.n() > 0) {
      recv_regions[source_rank] = permute_lattice(recv_region_old, axes);
      recv_counts[source_rank] = recv_region_old.n();
    }
  }

  // Phase 2: Allocate GPU buffers
  for (int r = 0; r < nprocs; r++) {
    if (send_counts[r] > 0) {
      CUDA_CHECK(cudaMalloc(&send_buffers_gpu[r], send_counts[r] * sizeof(T)));
    }
    if (recv_counts[r] > 0) {
      CUDA_CHECK(cudaMalloc(&recv_buffers_gpu[r], recv_counts[r] * sizeof(T)));
    }
  }

  // Phase 3: Pack data on GPU
  for (int target_rank = 0; target_rank < nprocs; target_rank++) {
    if (send_counts[target_rank] > 0) {
      const lattice& target_new_core = new_part.get_core(target_rank);
      lattice target_needs_old = inverse_permute_lattice(target_new_core, axes);
      lattice send_region = compute_intersection(old_local_core, target_needs_old);

      // Pack data (currently stages through CPU, could optimize with GPU kernel)
      std::vector<T> temp_buffer;
      pack_transposed_region(input, send_region, axes, temp_buffer);
      CUDA_CHECK(cudaMemcpy(send_buffers_gpu[target_rank], temp_buffer.data(),
                           temp_buffer.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  // Phase 4: GPU-aware MPI communication
  MPI_Datatype mpi_type = get_mpi_type<T>();
  std::vector<MPI_Request> requests;
  requests.reserve(2 * nprocs);

  // Post receives (pass GPU pointers directly!)
  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    if (recv_counts[source_rank] > 0 && source_rank != rank) {
      MPI_Request req;
      MPI_Irecv(recv_buffers_gpu[source_rank],  // GPU pointer!
                recv_counts[source_rank],
                mpi_type,
                source_rank,
                0,
                comm,
                &req);
      requests.push_back(req);
    }
  }

  // Post sends (pass GPU pointers directly!)
  for (int target_rank = 0; target_rank < nprocs; target_rank++) {
    if (send_counts[target_rank] > 0 && target_rank != rank) {
      MPI_Request req;
      MPI_Isend(send_buffers_gpu[target_rank],  // GPU pointer!
                send_counts[target_rank],
                mpi_type,
                target_rank,
                0,
                comm,
                &req);
      requests.push_back(req);
    }
  }

  // Handle self-copy on GPU
  if (send_counts[rank] > 0) {
    CUDA_CHECK(cudaMemcpy(recv_buffers_gpu[rank],
                         send_buffers_gpu[rank],
                         send_counts[rank] * sizeof(T),
                         cudaMemcpyDeviceToDevice));
  }

  // Wait for communication
  if (requests.size() > 0) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  // Phase 5: Unpack received data on GPU
  T* d_output = static_cast<T*>(output.get_devptr());

  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    if (recv_counts[source_rank] > 0) {
      // Copy from recv buffer to correct position in output
      // This needs a GPU kernel for proper unpacking
      // For now, stage through CPU
      std::vector<T> temp_recv(recv_counts[source_rank]);
      CUDA_CHECK(cudaMemcpy(temp_recv.data(), recv_buffers_gpu[source_rank],
                           recv_counts[source_rank] * sizeof(T),
                           cudaMemcpyDeviceToHost));

      // Unpack on CPU then copy to GPU
      // TODO: Optimize with GPU unpacking kernel
      unpack_transposed_region(output, recv_regions[source_rank], temp_recv);
    }
  }

  // Phase 6: Cleanup GPU buffers
  for (int r = 0; r < nprocs; r++) {
    if (send_buffers_gpu[r]) CUDA_CHECK(cudaFree(send_buffers_gpu[r]));
    if (recv_buffers_gpu[r]) CUDA_CHECK(cudaFree(recv_buffers_gpu[r]));
  }

  // Copy metadata
  output.set_multicomponents(input.multicomponents());
  output.set_has_time(input.has_time());

  return output;
}

} // namespace detail
} // namespace ftk

#endif // NDARRAY_HAVE_MPI && NDARRAY_HAVE_CUDA
#endif // _NDARRAY_TRANSPOSE_DISTRIBUTED_GPU_HH
