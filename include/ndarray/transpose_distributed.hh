#ifndef _NDARRAY_TRANSPOSE_DISTRIBUTED_HH
#define _NDARRAY_TRANSPOSE_DISTRIBUTED_HH

#if NDARRAY_HAVE_MPI

#include <ndarray/ndarray.hh>
#include <ndarray/lattice.hh>
#include <ndarray/lattice_partitioner.hh>
#include <ndarray/error.hh>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace ftk {
namespace detail {

/**
 * @brief Validate that transpose permutation is compatible with distributed array constraints
 *
 * CRITICAL CONSTRAINT: Only spatial dimensions can be decomposed across ranks.
 * Component dimensions (first n_component_dims) and time dimension (last if has_time)
 * must NOT be decomposed.
 *
 * Valid transpose must:
 * 1. Keep component dimensions in first n_component_dims positions
 * 2. Keep time dimension at last position (if has_time)
 * 3. Only permute spatial dimensions (middle dimensions)
 *
 * @throws invalid_operation if permutation violates constraints
 */
template <typename T, typename StoragePolicy>
void validate_distributed_transpose(const ndarray<T, StoragePolicy>& input,
                                     const std::vector<size_t>& axes) {
  const size_t nd = input.nd();
  const size_t n_comp = input.multicomponents();
  const bool has_time = input.has_time();

  // Component dimensions must stay at the beginning
  if (n_comp > 0) {
    for (size_t i = 0; i < n_comp; i++) {
      if (axes[i] >= n_comp) {
        throw invalid_operation(
          "transpose: Cannot move component dimension " + std::to_string(axes[i]) +
          " to position " + std::to_string(i) + " (outside component region). " +
          "Distributed arrays require component dimensions at beginning (first " +
          std::to_string(n_comp) + " positions).");
      }
    }

    // Also check that no spatial/time dimension maps into component region
    for (size_t i = n_comp; i < nd; i++) {
      if (axes[i] < n_comp) {
        throw invalid_operation(
          "transpose: Cannot move spatial/time dimension " + std::to_string(axes[i]) +
          " to component position " + std::to_string(i) + ". " +
          "Distributed arrays require component dimensions (first " +
          std::to_string(n_comp) + " dims) to stay at beginning.");
      }
    }
  }

  // Time dimension must stay at the end
  if (has_time) {
    if (axes[nd - 1] != nd - 1) {
      throw invalid_operation(
        "transpose: Cannot move time dimension (position " + std::to_string(nd - 1) +
        ") to position " + std::to_string(axes[nd - 1]) + ". " +
        "Distributed arrays require time dimension to stay at end.");
    }
  }
}

/**
 * @brief Compute intersection of two lattice regions
 */
inline lattice compute_intersection(const lattice& a, const lattice& b) {
  if (a.nd() == 0 || b.nd() == 0 || a.nd() != b.nd()) {
    // Return empty lattice with same dimensions as a (or 0 if a is empty)
    if (a.nd() > 0) {
      std::vector<size_t> zero_sizes(a.nd(), 0);
      std::vector<size_t> starts(a.nd(), 0);
      return lattice(starts, zero_sizes);
    }
    return lattice();  // Both empty
  }

  const size_t nd = a.nd();
  std::vector<size_t> starts(nd);
  std::vector<size_t> sizes(nd);

  for (size_t i = 0; i < nd; i++) {
    const size_t start_a = a.start(i);
    const size_t end_a = a.start(i) + a.size(i);
    const size_t start_b = b.start(i);
    const size_t end_b = b.start(i) + b.size(i);

    const size_t start_intersect = std::max(start_a, start_b);
    const size_t end_intersect = std::min(end_a, end_b);

    if (start_intersect >= end_intersect) {
      // No overlap - return zero-sized lattice with proper dimensions
      std::vector<size_t> zero_sizes(nd, 0);
      std::vector<size_t> zero_starts(nd, 0);
      return lattice(zero_starts, zero_sizes);
    }

    starts[i] = start_intersect;
    sizes[i] = end_intersect - start_intersect;
  }

  return lattice(starts, sizes);
}

/**
 * @brief Apply transpose permutation to lattice
 */
inline lattice permute_lattice(const lattice& orig, const std::vector<size_t>& axes) {
  const size_t nd = orig.nd();
  std::vector<size_t> new_starts(nd);
  std::vector<size_t> new_sizes(nd);

  for (size_t i = 0; i < nd; i++) {
    new_starts[i] = orig.start(axes[i]);
    new_sizes[i] = orig.size(axes[i]);
  }

  return lattice(new_starts, new_sizes);
}

/**
 * @brief Apply inverse permutation to lattice
 */
inline lattice inverse_permute_lattice(const lattice& transposed, const std::vector<size_t>& axes) {
  const size_t nd = transposed.nd();
  std::vector<size_t> orig_starts(nd);
  std::vector<size_t> orig_sizes(nd);

  // If axes[i] = j, then inv_axes[j] = i
  for (size_t i = 0; i < nd; i++) {
    orig_starts[axes[i]] = transposed.start(i);
    orig_sizes[axes[i]] = transposed.size(i);
  }

  return lattice(orig_starts, orig_sizes);
}

/**
 * @brief Get MPI datatype for template type
 */
template <typename T>
MPI_Datatype get_mpi_type() {
  if (std::is_same<T, double>::value) return MPI_DOUBLE;
  if (std::is_same<T, float>::value) return MPI_FLOAT;
  if (std::is_same<T, int>::value) return MPI_INT;
  if (std::is_same<T, long>::value) return MPI_LONG;
  if (std::is_same<T, unsigned int>::value) return MPI_UNSIGNED;
  if (std::is_same<T, unsigned long>::value) return MPI_UNSIGNED_LONG;
  if (std::is_same<T, char>::value) return MPI_CHAR;
  if (std::is_same<T, unsigned char>::value) return MPI_UNSIGNED_CHAR;

  // Default: treat as byte array
  return MPI_BYTE;
}

/**
 * @brief Pack data from region with transpose applied
 *
 * Iterates in transposed coordinate order to ensure packed data
 * matches the order expected during unpacking.
 */
template <typename T, typename StoragePolicy>
void pack_transposed_region(const ndarray<T, StoragePolicy>& input,
                            const lattice& global_region_original,
                            const std::vector<size_t>& axes,
                            std::vector<T>& buffer) {
  const size_t nd = input.nd();
  const size_t n_elems = global_region_original.n();

  if (global_region_original.nd() != nd) {
    throw std::runtime_error("pack_transposed_region: lattice nd mismatch");
  }

  buffer.clear();
  buffer.reserve(n_elems);

  // Pre-allocate vectors to avoid repeated allocations in the loop
  std::vector<size_t> transposed_idx(nd);
  std::vector<size_t> original_idx(nd);
  std::vector<size_t> local_idx(nd);

  // Compute transposed region dimensions for iteration
  std::vector<size_t> transposed_sizes(nd);
  for (size_t d = 0; d < nd; d++) {
    transposed_sizes[d] = global_region_original.size(axes[d]);
  }

  // Iterate in transposed coordinate order (to match unpacking order)
  for (size_t linear = 0; linear < n_elems; linear++) {
    // Compute multi-dimensional index in transposed space
    size_t tmp = linear;
    for (size_t d = 0; d < nd; d++) {
      transposed_idx[d] = tmp % transposed_sizes[d];
      tmp /= transposed_sizes[d];
    }

    // Map transposed index back to original coordinates
    for (size_t d = 0; d < nd; d++) {
      original_idx[axes[d]] = transposed_idx[d] + global_region_original.start(axes[d]);
    }

    // Access element (convert to local if needed)
    if (input.is_local(original_idx)) {
      local_idx = input.global_to_local(original_idx);

      // Use dimension-specific accessor for better performance
      T value;
      if (nd == 1) {
        value = input.f(local_idx[0]);
      } else if (nd == 2) {
        value = input.f(local_idx[0], local_idx[1]);
      } else if (nd == 3) {
        value = input.f(local_idx[0], local_idx[1], local_idx[2]);
      } else if (nd == 4) {
        value = input.f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]);
      } else {
        // Fall back to array accessor for higher dimensions
        value = input.f(local_idx.data());
      }

      buffer.push_back(value);
    } else {
      throw std::runtime_error("pack_transposed_region: index not in local core");
    }
  }
}

/**
 * @brief Unpack data to region (data is in transposed layout)
 */
template <typename T, typename StoragePolicy>
void unpack_transposed_region(ndarray<T, StoragePolicy>& output,
                              const lattice& global_region_transposed,
                              const std::vector<T>& buffer) {
  const size_t nd = output.nd();
  const size_t n_elems = global_region_transposed.n();

  if (buffer.size() != n_elems) {
    throw std::runtime_error("unpack_transposed_region: buffer size mismatch");
  }

  // Pre-allocate vectors to avoid repeated allocations in the loop
  std::vector<size_t> idx(nd);
  std::vector<size_t> local_idx(nd);

  // Iterate over region in transposed coordinates
  for (size_t linear = 0; linear < n_elems; linear++) {
    // Compute multi-dimensional index within region (transposed coordinates)
    size_t tmp = linear;
    for (size_t d = 0; d < nd; d++) {
      idx[d] = global_region_transposed.start(d) + (tmp % global_region_transposed.size(d));
      tmp /= global_region_transposed.size(d);
    }

    // Write element (convert to local)
    if (output.is_local(idx)) {
      local_idx = output.global_to_local(idx);

      // Use dimension-specific accessor for better performance
      if (nd == 1) {
        output.f(local_idx[0]) = buffer[linear];
      } else if (nd == 2) {
        output.f(local_idx[0], local_idx[1]) = buffer[linear];
      } else if (nd == 3) {
        output.f(local_idx[0], local_idx[1], local_idx[2]) = buffer[linear];
      } else if (nd == 4) {
        output.f(local_idx[0], local_idx[1], local_idx[2], local_idx[3]) = buffer[linear];
      } else {
        // Fall back to array accessor for higher dimensions
        output.f(local_idx.data()) = buffer[linear];
      }
    } else {
      throw std::runtime_error("unpack_transposed_region: index not in local core");
    }
  }
}

/**
 * @brief Main distributed transpose implementation
 */
template <typename T, typename StoragePolicy>
ndarray<T, StoragePolicy> transpose_distributed(const ndarray<T, StoragePolicy>& input,
                                                 const std::vector<size_t>& axes) {
  const size_t nd = input.nd();

  // Validate
  validate_distributed_transpose(input, axes);

  // Get old distribution info
  const lattice& old_global = input.global_lattice();
  const lattice& old_local_core = input.local_core();
  const lattice_partitioner& old_part = input.partitioner();
  const std::vector<size_t>& old_decomp = input.decomp_pattern();
  const std::vector<size_t>& old_ghost = input.ghost_widths();

  MPI_Comm comm = input.comm();
  int rank = input.rank();
  int nprocs = input.nprocs();

  // Compute new global dimensions
  std::vector<size_t> new_global_dims(nd);
  for (size_t i = 0; i < nd; i++) {
    new_global_dims[i] = old_global.size(axes[i]);
  }

  // Permute decomp pattern and ghosts
  std::vector<size_t> new_decomp(nd);
  std::vector<size_t> new_ghost(nd);
  for (size_t i = 0; i < nd; i++) {
    new_decomp[i] = old_decomp[axes[i]];
    new_ghost[i] = old_ghost[axes[i]];
  }

  // Create output array with new distribution
  ndarray<T, StoragePolicy> output;
  output.decompose(comm, new_global_dims, nprocs, new_decomp, new_ghost);

  const lattice& new_local_core = output.local_core();
  const lattice_partitioner& new_part = output.partitioner();

  // Prepare send/recv buffers
  std::vector<std::vector<T>> send_buffers(nprocs);
  std::vector<std::vector<T>> recv_buffers(nprocs);
  std::vector<lattice> recv_regions(nprocs);

  // Phase 1: Compute what to send to each rank
  for (int target_rank = 0; target_rank < nprocs; target_rank++) {
    // What does target_rank need in new (transposed) coordinates?
    const lattice& target_new_core = new_part.get_core(target_rank);

    // Convert to old (original) coordinates
    lattice target_needs_old = inverse_permute_lattice(target_new_core, axes);

    // What part of my old core overlaps with what target needs?
    lattice send_region = compute_intersection(old_local_core, target_needs_old);

    if (send_region.n() > 0) {
      // Pack data from this region
      pack_transposed_region(input, send_region, axes, send_buffers[target_rank]);
    }
  }

  // Phase 2: Compute what to receive from each rank
  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    // What does source_rank have in old coordinates?
    const lattice& source_old_core = old_part.get_core(source_rank);

    // Convert my new core to old coordinates
    lattice i_need_old = inverse_permute_lattice(new_local_core, axes);

    // What part of source's old core do I need?
    lattice recv_region_old = compute_intersection(source_old_core, i_need_old);

    if (recv_region_old.n() > 0) {
      // Convert to new (transposed) coordinates for unpacking
      recv_regions[source_rank] = permute_lattice(recv_region_old, axes);
      recv_buffers[source_rank].resize(recv_region_old.n());
    }
  }

  // Phase 3: MPI communication
  MPI_Datatype mpi_type = get_mpi_type<T>();
  size_t type_size = (mpi_type == MPI_BYTE) ? sizeof(T) : 1;

  std::vector<MPI_Request> requests;
  requests.reserve(2 * nprocs);

  // Post receives
  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    if (recv_buffers[source_rank].size() > 0 && source_rank != rank) {
      MPI_Request req;
      MPI_Irecv(recv_buffers[source_rank].data(),
                recv_buffers[source_rank].size() * type_size,
                mpi_type,
                source_rank,
                0,  // tag
                comm,
                &req);
      requests.push_back(req);
    }
  }

  // Post sends
  for (int target_rank = 0; target_rank < nprocs; target_rank++) {
    if (send_buffers[target_rank].size() > 0 && target_rank != rank) {
      MPI_Request req;
      MPI_Isend(send_buffers[target_rank].data(),
                send_buffers[target_rank].size() * type_size,
                mpi_type,
                target_rank,
                0,  // tag
                comm,
                &req);
      requests.push_back(req);
    }
  }

  // Handle self-copy (rank sending to itself)
  if (send_buffers[rank].size() > 0) {
    recv_buffers[rank] = send_buffers[rank];
  }

  // Wait for all communication
  if (requests.size() > 0) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  // Phase 4: Unpack received data
  for (int source_rank = 0; source_rank < nprocs; source_rank++) {
    if (recv_buffers[source_rank].size() > 0) {
      unpack_transposed_region(output, recv_regions[source_rank], recv_buffers[source_rank]);
    }
  }

  // Copy metadata
  output.set_multicomponents(input.multicomponents());
  output.set_has_time(input.has_time());

  return output;
}

} // namespace detail
} // namespace ftk

#endif // NDARRAY_HAVE_MPI
#endif // _NDARRAY_TRANSPOSE_DISTRIBUTED_HH
