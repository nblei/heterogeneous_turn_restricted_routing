#include "digraph.hh"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <omp.h>
#include <vector>

// Private methods
void digraph::update_path(unsigned i, unsigned j) {
  // Update paths from i based on paths from j
  auto start_loop1 = std::chrono::high_resolution_clock::now();

  // Instead of checking each bit individually, use SIMD to copy all set bits
  // from row j to row i We'll extract row j, then apply it to row i using SIMD
  // operations with the efficient version
  const auto j_row_data = _path.get_row_data(j);
  _path.apply_row_mask_efficient(i, j_row_data);

  auto end_loop1 = std::chrono::high_resolution_clock::now();
  _perf_stats.update_path_loop1_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_loop1 -
                                                            start_loop1);
  _perf_stats.update_path_loop1_calls++;

  // For each node that had a path to i, update its paths
  auto start_loop2 = std::chrono::high_resolution_clock::now();

  // Cache the i-th row for repeated access
  const auto i_row_data = _path.get_row_data(i);

// Collect all nodes that need updating
#ifdef ENABLE_PROFILING
  auto start_collect = std::chrono::high_resolution_clock::now();
#endif

  std::vector<unsigned> nodes_to_update;
  // Use column-based collection which is more efficient for this operation
  // This is a transpose operation - finding all rows that have column i set
  _path.collect_column_indices_fast(i, nodes_to_update);

#ifdef ENABLE_PROFILING
  auto end_collect = std::chrono::high_resolution_clock::now();
  auto collect_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_collect - start_collect)
                          .count();
  std::cout << "Column collection time: " << collect_time / 1000.0 << " ms for "
            << nodes_to_update.size() << " nodes" << std::endl;
#endif

// THREAD SAFETY NOTE:
// This parallel implementation depends on the specific implementation of
// bit_matrix where set_bit() is thread-safe when operating on different rows.
// If the bit_matrix implementation changes, this may need locking.
//
// Specifically, this assumes:
// 1. Each row's storage is separate in memory (no overlap)
// 2. set_bit() only modifies memory associated with the given row
// 3. No internal shared state is modified when setting bits
//
// If these assumptions break, you would need to either:
// - Use critical sections around _path.set_bit(k, l) calls
// - Create thread-local bit_matrix objects for each thread and merge at the end

// PROFILING: Print thread activity information
#ifdef ENABLE_PROFILING
  std::cout << "Starting parallel section with " << nodes_to_update.size()
            << " nodes to update. Available threads: " << omp_get_max_threads()
            << std::endl;
  std::atomic<int> actual_threads_used{0};
  int max_threads_seen = 0;
#endif

#ifdef ENABLE_PROFILING
#pragma omp parallel for schedule(dynamic, 8) reduction(max : max_threads_seen)
#else
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (size_t idx = 0; idx < nodes_to_update.size(); idx++) {
#ifdef ENABLE_PROFILING
    // Track thread utilization
    int thread_num = omp_get_thread_num();
    actual_threads_used++;
    max_threads_seen = std::max(max_threads_seen, omp_get_num_threads());
#endif

    unsigned k = nodes_to_update[idx];

    // We'll use a faster SIMD approach to check if any bits need to be updated
    // We're checking if (i_row_data & ~k_row_data) has any bits set
    const auto k_row_data = _path.get_row_data(k);

    // Can use vector instructions to compute this quickly
    bool needs_update = false;
    for (unsigned w = 0; w < i_row_data.size() && !needs_update; w++) {
      __m256i i_block = i_row_data[w].val;
      __m256i k_block = k_row_data[w].val;

      // Create ~k_row_data
      __m256i k_complement = _mm256_xor_si256(k_block, _mm256_set1_epi32(-1));

      // Calculate i_row_data & ~k_row_data
      __m256i new_bits = _mm256_and_si256(i_block, k_complement);

      // Check if result has any bits set (is non-zero)
      __m256i zero = _mm256_setzero_si256();
      if (!_mm256_testz_si256(
              new_bits, new_bits)) { // Returns 1 if all zero, 0 if any bit set
        needs_update = true;
      }
    }

    // Only perform updates if we found at least one new bit to set
    if (needs_update) {
      // Use SIMD-accelerated row operations with the efficient version
      // This only writes to memory blocks that actually have new bits to set
      _path.apply_row_mask_efficient(k, i_row_data);
    }
  }

#ifdef ENABLE_PROFILING
  // Report thread utilization
  std::cout << "Completed parallel section. "
            << "Actual threads used: " << actual_threads_used.load()
            << ", Max concurrent threads: " << max_threads_seen
            << ", Total updates: " << nodes_to_update.size() << std::endl;
#endif

  auto end_loop2 = std::chrono::high_resolution_clock::now();
  _perf_stats.update_path_loop2_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_loop2 -
                                                            start_loop2);
  _perf_stats.update_path_loop2_calls++;
}

size_t digraph::chip_idx(size_t chip_x, size_t chip_y) const {
  return total_nodes * (chip_x + _num_chips_x * chip_y);
}

size_t digraph::node_idx(size_t chip_x, size_t chip_y, internal_node n) const {
  return chip_idx(chip_x, chip_y) + n;
}

// Private constructors
digraph::digraph(unsigned n)
    : _n(n), _adj(n, n), _path(n, n), _viable_edges(n, n) {}

// Initialize viable edges (all possible turns)
void digraph::initialize_viable_edges() {
  for (int chip_x = 0; chip_x < _num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < _num_chips_y; chip_y++) {
      // Add viable edges between all cardinal in/out ports
      // North in -> {East,South,West} out
      _viable_edges.set_bit(node_idx(chip_x, chip_y, north_in),
                            node_idx(chip_x, chip_y, east_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, north_in),
                            node_idx(chip_x, chip_y, south_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, north_in),
                            node_idx(chip_x, chip_y, west_out));

      // East in -> {North,South,West} out
      _viable_edges.set_bit(node_idx(chip_x, chip_y, east_in),
                            node_idx(chip_x, chip_y, north_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, east_in),
                            node_idx(chip_x, chip_y, south_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, east_in),
                            node_idx(chip_x, chip_y, west_out));

      // South in -> {North,East,West} out
      _viable_edges.set_bit(node_idx(chip_x, chip_y, south_in),
                            node_idx(chip_x, chip_y, north_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, south_in),
                            node_idx(chip_x, chip_y, east_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, south_in),
                            node_idx(chip_x, chip_y, west_out));

      // West in -> {North,East,South} out
      _viable_edges.set_bit(node_idx(chip_x, chip_y, west_in),
                            node_idx(chip_x, chip_y, north_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, west_in),
                            node_idx(chip_x, chip_y, east_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, west_in),
                            node_idx(chip_x, chip_y, south_out));

      // Add viable edges from cardinal in ports to cpu_in
      _viable_edges.set_bit(node_idx(chip_x, chip_y, north_in),
                            node_idx(chip_x, chip_y, cpu_in));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, east_in),
                            node_idx(chip_x, chip_y, cpu_in));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, south_in),
                            node_idx(chip_x, chip_y, cpu_in));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, west_in),
                            node_idx(chip_x, chip_y, cpu_in));

      // Add viable edges from cpu_out to cardinal out ports
      _viable_edges.set_bit(node_idx(chip_x, chip_y, cpu_out),
                            node_idx(chip_x, chip_y, north_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, cpu_out),
                            node_idx(chip_x, chip_y, east_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, cpu_out),
                            node_idx(chip_x, chip_y, south_out));
      _viable_edges.set_bit(node_idx(chip_x, chip_y, cpu_out),
                            node_idx(chip_x, chip_y, west_out));

      // Add required edges from all cardinal in ports to cpu_in
      // These are essential for CPU connectivity
      _adj.set_bit(node_idx(chip_x, chip_y, north_in),
                   node_idx(chip_x, chip_y, cpu_in));
      _adj.set_bit(node_idx(chip_x, chip_y, east_in),
                   node_idx(chip_x, chip_y, cpu_in));
      _adj.set_bit(node_idx(chip_x, chip_y, south_in),
                   node_idx(chip_x, chip_y, cpu_in));
      _adj.set_bit(node_idx(chip_x, chip_y, west_in),
                   node_idx(chip_x, chip_y, cpu_in));

      // Add required edges from cpu_out to all cardinal out ports
      // Also essential for CPU connectivity
      _adj.set_bit(node_idx(chip_x, chip_y, cpu_out),
                   node_idx(chip_x, chip_y, north_out));
      _adj.set_bit(node_idx(chip_x, chip_y, cpu_out),
                   node_idx(chip_x, chip_y, east_out));
      _adj.set_bit(node_idx(chip_x, chip_y, cpu_out),
                   node_idx(chip_x, chip_y, south_out));
      _adj.set_bit(node_idx(chip_x, chip_y, cpu_out),
                   node_idx(chip_x, chip_y, west_out));
    }
  }
}

// Initialize grid connections
void digraph::initialize_grid_connections() {
  for (int chip_x = 0; chip_x < _num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < _num_chips_y; chip_y++) {
      // Connect to north neighbor if not on top edge
      if (chip_y > 0) {
        size_t current = node_idx(chip_x, chip_y, north_out);
        size_t neighbor = node_idx(chip_x, chip_y - 1, south_in);
        _viable_edges.set_bit(current, neighbor);
        _adj.set_bit(current, neighbor);
      }

      // Connect to south neighbor if not on bottom edge
      if (chip_y < _num_chips_y - 1) {
        size_t current = node_idx(chip_x, chip_y, south_out);
        size_t neighbor = node_idx(chip_x, chip_y + 1, north_in);
        _viable_edges.set_bit(current, neighbor);
        _adj.set_bit(current, neighbor);
      }

      // Connect to west neighbor if not on left edge
      if (chip_x > 0) {
        size_t current = node_idx(chip_x, chip_y, west_out);
        size_t neighbor = node_idx(chip_x - 1, chip_y, east_in);
        _viable_edges.set_bit(current, neighbor);
        _adj.set_bit(current, neighbor);
      }

      // Connect to east neighbor if not on right edge
      if (chip_x < _num_chips_x - 1) {
        size_t current = node_idx(chip_x, chip_y, east_out);
        size_t neighbor = node_idx(chip_x + 1, chip_y, west_in);
        _viable_edges.set_bit(current, neighbor);
        _adj.set_bit(current, neighbor);
      }
    }
  }
}

// Public methods
// Initialize with basic grid connectivity but no turns
digraph::digraph(unsigned num_chips_x, unsigned num_chips_y)
    : digraph(num_chips_x * num_chips_y * total_nodes) {
  _num_chips_x = num_chips_x;
  _num_chips_y = num_chips_y;

  initialize_viable_edges();
  initialize_grid_connections();
  compute_paths();
}

// Initialize with XY routing
digraph digraph::create_xy_routing(unsigned num_chips_x, unsigned num_chips_y) {
  digraph d(num_chips_x, num_chips_y);

  // Add turns for XY routing (first X, then Y)
  for (int chip_x = 0; chip_x < num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < num_chips_y; chip_y++) {
      // XY routing principle: first route in X direction, then in Y direction

      // East input to North/South output (after completing X movement)
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, south_out));

      // West input to North/South output (after completing X movement)
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, south_out));

      // Allow straight-through paths
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, west_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, east_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, north_in),
                        d.node_idx(chip_x, chip_y, south_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, south_in),
                        d.node_idx(chip_x, chip_y, north_out));

      // Since we're using XY routing, we don't add Y→X turns (North/South
      // input to East/West output) This restriction is what prevents cycles
      // in XY routing

      // Add cardinal inputs to CPU_in to ensure CPU connectivity
      // These need to be direct connections in the adjacency matrix
      d._adj.set_bit(d.node_idx(chip_x, chip_y, north_in),
                     d.node_idx(chip_x, chip_y, cpu_in));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, east_in),
                     d.node_idx(chip_x, chip_y, cpu_in));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, south_in),
                     d.node_idx(chip_x, chip_y, cpu_in));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, west_in),
                     d.node_idx(chip_x, chip_y, cpu_in));

      // Add CPU out to all directions
      // These also need to be direct connections in the adjacency matrix
      d._adj.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                     d.node_idx(chip_x, chip_y, north_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                     d.node_idx(chip_x, chip_y, east_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                     d.node_idx(chip_x, chip_y, south_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                     d.node_idx(chip_x, chip_y, west_out));
    }
  }

  // Need to recompute paths after setting the adjacency matrix directly
  d.compute_paths();

  return d;
}

// Initialize with West-first routing
digraph digraph::create_west_first_routing(unsigned num_chips_x,
                                           unsigned num_chips_y) {
  // Create a completely new routing algorithm that combines West-first
  // principles with acyclicity but ensures full CPU connectivity

  // Instead of modifying an existing digraph, start from scratch for clarity
  // This way we have full control over which edges are added

  // Create digraph with empty adjacency matrix
  digraph d(num_chips_x, num_chips_y);
  d._num_chips_x = num_chips_x;
  d._num_chips_y = num_chips_y;

  // Step 3: Add turn connections within each router
  for (int chip_x = 0; chip_x < num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < num_chips_y; chip_y++) {
      // Standard west-first routing turns:

      // 1. West input can go to any output (the "West-first" principle)
      d._adj.set_bit(d.node_idx(chip_x, chip_y, west_in),
                     d.node_idx(chip_x, chip_y, north_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, west_in),
                     d.node_idx(chip_x, chip_y, east_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, west_in),
                     d.node_idx(chip_x, chip_y, south_out));

      // 2. North/South can always turn East (standard west-first allows this)
      d._adj.set_bit(d.node_idx(chip_x, chip_y, north_in),
                     d.node_idx(chip_x, chip_y, east_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, south_in),
                     d.node_idx(chip_x, chip_y, east_out));

      // 3. East input can go North/South (standard west-first allows this)
      d._adj.set_bit(d.node_idx(chip_x, chip_y, east_in),
                     d.node_idx(chip_x, chip_y, north_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, east_in),
                     d.node_idx(chip_x, chip_y, south_out));

      // 4. Straight-through paths for all directions
      d._adj.set_bit(d.node_idx(chip_x, chip_y, north_in),
                     d.node_idx(chip_x, chip_y, south_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, south_in),
                     d.node_idx(chip_x, chip_y, north_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, east_in),
                     d.node_idx(chip_x, chip_y, west_out));
      d._adj.set_bit(d.node_idx(chip_x, chip_y, west_in),
                     d.node_idx(chip_x, chip_y, east_out));
    }
  }

  // Ensure viable_edges matrix is populated for future addacyclic_edge calls
  d._viable_edges = bit_matrix(d._n, d._n);
  for (int chip_x = 0; chip_x < num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < num_chips_y; chip_y++) {
      // Add all possible turns to viable edges
      // North in -> any out
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, north_in),
                              d.node_idx(chip_x, chip_y, east_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, north_in),
                              d.node_idx(chip_x, chip_y, south_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, north_in),
                              d.node_idx(chip_x, chip_y, west_out));

      // East in -> any out
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, east_in),
                              d.node_idx(chip_x, chip_y, north_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, east_in),
                              d.node_idx(chip_x, chip_y, south_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, east_in),
                              d.node_idx(chip_x, chip_y, west_out));

      // South in -> any out
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, south_in),
                              d.node_idx(chip_x, chip_y, north_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, south_in),
                              d.node_idx(chip_x, chip_y, east_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, south_in),
                              d.node_idx(chip_x, chip_y, west_out));

      // West in -> any out
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, west_in),
                              d.node_idx(chip_x, chip_y, north_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, west_in),
                              d.node_idx(chip_x, chip_y, east_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, west_in),
                              d.node_idx(chip_x, chip_y, south_out));

      // CPU connections
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, north_in),
                              d.node_idx(chip_x, chip_y, cpu_in));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, east_in),
                              d.node_idx(chip_x, chip_y, cpu_in));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, south_in),
                              d.node_idx(chip_x, chip_y, cpu_in));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, west_in),
                              d.node_idx(chip_x, chip_y, cpu_in));

      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                              d.node_idx(chip_x, chip_y, north_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                              d.node_idx(chip_x, chip_y, east_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                              d.node_idx(chip_x, chip_y, south_out));
      d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, cpu_out),
                              d.node_idx(chip_x, chip_y, west_out));

      // External links
      if (chip_y > 0) {
        d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, north_out),
                                d.node_idx(chip_x, chip_y - 1, south_in));
      }
      if (chip_y < num_chips_y - 1) {
        d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, south_out),
                                d.node_idx(chip_x, chip_y + 1, north_in));
      }
      if (chip_x > 0) {
        d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, west_out),
                                d.node_idx(chip_x - 1, chip_y, east_in));
      }
      if (chip_x < num_chips_x - 1) {
        d._viable_edges.set_bit(d.node_idx(chip_x, chip_y, east_out),
                                d.node_idx(chip_x + 1, chip_y, west_in));
      }
    }
  }

  // Compute paths to update path matrix
  d.compute_paths();

  return d;
}

// Initialize with Odd-Even routing
digraph digraph::create_odd_even_routing(unsigned num_chips_x,
                                         unsigned num_chips_y) {
  digraph d(num_chips_x, num_chips_y);

  // Add turns for Odd-Even routing
  for (int chip_x = 0; chip_x < num_chips_x; chip_x++) {
    for (int chip_y = 0; chip_y < num_chips_y; chip_y++) {
      // Odd-Even routing rules depend on column
      bool is_even_column = (chip_x % 2 == 0);

      // East-to-North/South turns allowed in all columns
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, south_out));

      // West-to-North/South turns allowed in all columns
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, south_out));

      // North-to-East turns allowed in all columns
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, north_in),
                        d.node_idx(chip_x, chip_y, east_out));

      // South-to-East turns allowed in all columns
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, south_in),
                        d.node_idx(chip_x, chip_y, east_out));

      // North-to-West turns allowed only in even columns
      if (is_even_column) {
        d.addacyclic_edge(d.node_idx(chip_x, chip_y, north_in),
                          d.node_idx(chip_x, chip_y, west_out));
      }

      // South-to-West turns allowed only in even columns
      if (is_even_column) {
        d.addacyclic_edge(d.node_idx(chip_x, chip_y, south_in),
                          d.node_idx(chip_x, chip_y, west_out));
      }

      // East-to-West turns are generally restricted, but allow them in even
      // columns for better connectivity
      if (is_even_column) {
        d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                          d.node_idx(chip_x, chip_y, west_out));
      }

      // Straight-through paths
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, north_in),
                        d.node_idx(chip_x, chip_y, south_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, south_in),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, west_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, east_out));

      // Add CPU connectivity
      // CPU in from all cardinal directions
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, north_in),
                        d.node_idx(chip_x, chip_y, cpu_in));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, east_in),
                        d.node_idx(chip_x, chip_y, cpu_in));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, south_in),
                        d.node_idx(chip_x, chip_y, cpu_in));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, west_in),
                        d.node_idx(chip_x, chip_y, cpu_in));

      // CPU out to all cardinal directions
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, cpu_out),
                        d.node_idx(chip_x, chip_y, north_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, cpu_out),
                        d.node_idx(chip_x, chip_y, east_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, cpu_out),
                        d.node_idx(chip_x, chip_y, south_out));
      d.addacyclic_edge(d.node_idx(chip_x, chip_y, cpu_out),
                        d.node_idx(chip_x, chip_y, west_out));
    }
  }

  return d;
}

// Warshall's algorithm
void digraph::compute_paths() {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Initialize path matrix with adjacency matrix
  _path = _adj;

  // For each intermediate vertex k
  for (unsigned k = 0; k < _n; k++) {
    // For each source vertex i
    for (unsigned i = 0; i < _n; i++) {
      // If there's a path from i to k
      if (_path.get_bit(i, k)) {
        // For each destination vertex j
        for (unsigned j = 0; j < _n; j++) {
          // If there's a path from k to j, then there's a path from i to j
          if (_path.get_bit(k, j)) {
            _path.set_bit(i, j);
          }
        }
      }
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.path_computation_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);
  _perf_stats.path_computations++;
}

void digraph::addedge(unsigned i, unsigned j) {
  auto start_time = std::chrono::high_resolution_clock::now();

  if (i >= _n || j >= _n || _adj.get_bit(i, j)) {
    return;
  }

  unsigned old_count = _adj.count_ones();
  _adj.set_bit(i, j);
  unsigned new_count = _adj.count_ones();

  // Recompute paths after adding edge
  update_path(i, j);

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.edge_add_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);
  _perf_stats.edge_additions++;
}

bool digraph::addacyclic_edge(unsigned i, unsigned j) {
  if (i >= _n || j >= _n) {
    return false;
  }
  if (_adj.get_bit(i, j)) {
    return false;
  }
  if (_path.get_bit(j, i)) {
    return false;
  }
  if (!_viable_edges.get_bit(i, j)) {
    return false;
  }

  // If we get here, the edge is safe to add
  addedge(i, j);
  return true;
}

bool digraph::check_cycle() const {
  // Don't track performance in const method to avoid modifying member variables
  bool has_cycle = false;
  for (unsigned i = 0; i < _n; i++) {
    if (_path.get_bit(i, i)) {
      has_cycle = true;
      break;
    }
  }

  return has_cycle;
}

// Non-const version for performance tracking
bool digraph::check_cycle_and_track() {
  auto start_time = std::chrono::high_resolution_clock::now();

  bool has_cycle = check_cycle();

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.cycle_check_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);
  _perf_stats.cycle_checks++;

  return has_cycle;
}

bit_matrix digraph::get_addable_edges() const {
  // Start with viable edges that aren't already in _adj
  bit_matrix result = _viable_edges;

  bit_matrix nonadj = ~_adj;

  result = result & nonadj;

  // Remove edges that would create cycles
  unsigned removed = 0;
  for (unsigned v = 0; v < _n; v++) {
    for (unsigned u = 0; u < _n; u++) {
      if (_path.get_bit(u, v)) {
        if (result.get_bit(v, u)) {
          result.clear_bit(v, u);
          removed++;
        }
      }
    }
  }

  // Debug: Print some example addable edges
  if (result.count_ones() > 0) {
    unsigned count = 0;
    for (unsigned i = 0; i < _n && count < 5; i++) {
      for (unsigned j = 0; j < _n && count < 5; j++) {
        if (result.get_bit(i, j)) {
          count++;
        }
      }
    }
  }

  return result;
}

bool digraph::path(unsigned i, unsigned j) const {
  return i < _n && j < _n && _path.get_bit(i, j);
}

unsigned digraph::count_edges() const { return _adj.count_ones(); }

unsigned digraph::count_edge_closure() const { return _path.count_ones(); }

void digraph::print_matrices() const {
  std::cout << "\nAdjacency Matrix:\n";
  _adj.print_compact();
  std::cout << "\nPath Matrix:\n";
  _path.print_compact();
  std::cout << "\nViable Edges Matrix:\n";
  _viable_edges.print_compact();
}

void digraph::debug_print_matrices() const {
  std::cout << "=== Matrix Debug Info ===\n";
  std::cout << "Total nodes: " << _n << "\n";
  std::cout << "Grid size: " << _num_chips_x << "x" << _num_chips_y << "\n";
  std::cout << "Edge count: " << count_edges() << "\n\n";

  std::cout << "Adjacency Matrix:\n";
  _adj.print_compact();
  std::cout << "\nPath Matrix:\n";
  _path.print_compact();
  std::cout << "\nViable Edges Matrix:\n";
  _viable_edges.print_compact();
  std::cout << "=====================\n";
}

void digraph::verify_edge_add(unsigned i, unsigned j) const {
  std::cout << "Verifying edge (" << i << ", " << j << "):\n";
  std::cout << "In bounds: " << (i < _n && j < _n) << "\n";
  std::cout << "Already in adj: " << _adj.get_bit(i, j) << "\n";
  std::cout << "Would create cycle: " << _path.get_bit(j, i) << "\n";
  std::cout << "Is viable: " << _viable_edges.get_bit(i, j) << "\n";
}

bool digraph::has_cpu_path(size_t src_x, size_t src_y, size_t dst_x,
                           size_t dst_y) const {
  size_t src = node_idx(src_x, src_y, cpu_out);
  size_t dst = node_idx(dst_x, dst_y, cpu_in);
  return path(src, dst);
}

// Check if all required CPU-to-CPU paths exist
bool digraph::check_all_cpu_paths() const {
  for (size_t src_x = 0; src_x < _num_chips_x; src_x++) {
    for (size_t src_y = 0; src_y < _num_chips_y; src_y++) {
      for (size_t dst_x = 0; dst_x < _num_chips_x; dst_x++) {
        for (size_t dst_y = 0; dst_y < _num_chips_y; dst_y++) {
          if (src_x == dst_x && src_y == dst_y)
            continue; // Skip same chip
          if (!has_cpu_path(src_x, src_y, dst_x, dst_y)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// Add all possible edges while maintaining required properties
void digraph::add_all_possible_edges() {
  bool added_any;
  do {
    added_any = false;
    bit_matrix addable = get_addable_edges();

    // Try adding each possible edge
    for (unsigned i = 0; i < _n; i++) {
      for (unsigned j = 0; j < _n; j++) {
        if (addable.get_bit(i, j)) {
          // Create temporary copy of graph to test edge addition
          digraph temp = *this;
          if (temp.addacyclic_edge(i, j) && temp.check_all_cpu_paths()) {
            // If edge maintains required properties, add it to actual graph
            addacyclic_edge(i, j);
            added_any = true;
            break; // Start over with new addable edges matrix
          }
        }
      }
      if (added_any)
        break;
    }
  } while (added_any);
}

// Add edges randomly while maintaining required properties
void digraph::add_random_edges(std::mt19937_64 &rng) {
  bool added_any;
  do {
    added_any = false;
    bit_matrix addable = get_addable_edges();

    // Count total addable edges
    unsigned total_addable = addable.count_ones();
    if (total_addable == 0) {
      break;
    }

    // Generate random index
    std::uniform_int_distribution<unsigned> dist(0, total_addable - 1);
    unsigned target = dist(rng);

    // Find the target edge
    unsigned count = 0;
    for (unsigned i = 0; i < _n && !added_any; i++) {
      for (unsigned j = 0; j < _n && !added_any; j++) {
        if (addable.get_bit(i, j)) {
          if (count == target) {
            if (addacyclic_edge(i, j)) {
              added_any = true;
            }
            break;
          }
          count++;
        }
      }
    }
  } while (added_any);
}

// Count internal turn edges for a specific chip
unsigned digraph::count_chip_turn_edges(size_t chip_x, size_t chip_y) const {
  unsigned count = 0;
  internal_node in_ports[] = {north_in, east_in, south_in, west_in};
  internal_node out_ports[] = {north_out, east_out, south_out, west_out};

  // Check each possible in->out combination
  for (auto in_port : in_ports) {
    for (auto out_port : out_ports) {
      // Skip invalid turns (going back in same direction)
      if ((in_port == north_in && out_port == south_out) ||
          (in_port == south_in && out_port == north_out) ||
          (in_port == east_in && out_port == west_out) ||
          (in_port == west_in && out_port == east_out)) {
        continue;
      }

      if (_adj.get_bit(node_idx(chip_x, chip_y, in_port),
                       node_idx(chip_x, chip_y, out_port))) {
        count++;
      }
    }
  }
  return count;
}

// Count all internal turn edges across all chips
unsigned digraph::count_all_turn_edges() const {
  unsigned total = 0;
  for (size_t x = 0; x < _num_chips_x; x++) {
    for (size_t y = 0; y < _num_chips_y; y++) {
      total += count_chip_turn_edges(x, y);
    }
  }
  return total;
}

// Get statistics about the current graph
void digraph::print_stats() const {
  std::cout << "Graph Statistics:\n";
  std::cout << "Dimensions: " << _num_chips_x << "x" << _num_chips_y
            << " chips\n";
  std::cout << "Total nodes: " << _n << "\n";
  std::cout << "Total edges: " << count_edges() << "\n";
  std::cout << "Path closure size: " << count_edge_closure() << "\n";
  std::cout << "Is acyclic: " << (!check_cycle() ? "yes" : "no") << "\n";
  std::cout << "All CPU paths exist: " << (check_all_cpu_paths() ? "yes" : "no")
            << "\n";
  std::cout << "Total internal turn edges: " << count_all_turn_edges() << "\n";
}

void digraph::print_perf_stats() const { _perf_stats.print(); }

const digraph::PerformanceStats &digraph::get_perf_stats() const {
  return _perf_stats;
}

void digraph::reset_perf_stats() { _perf_stats.reset(); }

unsigned digraph::NumChipsX() const { return _num_chips_x; }
unsigned digraph::NumChipsY() const { return _num_chips_y; }

// Convert internal_node enum to string
const char *digraph::node_type_to_string(internal_node node) const {
  switch (node) {
  case north_in:
    return "north_in";
  case north_out:
    return "north_out";
  case east_in:
    return "east_in";
  case east_out:
    return "east_out";
  case south_in:
    return "south_in";
  case south_out:
    return "south_out";
  case west_in:
    return "west_in";
  case west_out:
    return "west_out";
  case cpu_in:
    return "cpu_in";
  case cpu_out:
    return "cpu_out";
  default:
    return "unknown";
  }
}

// Log turn edges to a file
bool digraph::log_turn_edges(const std::string &filename,
                             const std::string &routing_type,
                             bool optimized) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  // Write header
  file << "# DIGRAPH-TURNS 1.0\n";
  file << "# rows: " << _num_chips_y << "\n";
  file << "# cols: " << _num_chips_x << "\n";
  file << "# routing: " << routing_type << "\n";
  file << "# optimized: " << (optimized ? "true" : "false") << "\n";
  file << "# turn_edges: " << count_all_turn_edges() << "\n\n";

  // Write format description
  file << "# Format: <chip_row>,<chip_col>: <from_port> -> <to_port>\n";

  // Check all possible internal turn edges in each chip
  internal_node in_ports[] = {north_in, east_in, south_in, west_in};
  internal_node out_ports[] = {north_out, east_out, south_out, west_out};

  for (size_t y = 0; y < _num_chips_y; y++) {
    for (size_t x = 0; x < _num_chips_x; x++) {
      // Check each possible in->out combination
      for (auto in_port : in_ports) {
        for (auto out_port : out_ports) {
          // Skip straight-through connections
          if ((in_port == north_in && out_port == south_out) ||
              (in_port == south_in && out_port == north_out) ||
              (in_port == east_in && out_port == west_out) ||
              (in_port == west_in && out_port == east_out)) {
            continue;
          }

          // If this turn edge exists, log it
          if (_adj.get_bit(node_idx(x, y, in_port), node_idx(x, y, out_port))) {
            file << y << "," << x << ": " << node_type_to_string(in_port)
                 << " -> " << node_type_to_string(out_port) << "\n";
          }
        }
      }
    }
  }

  file.close();
  return true;
}

// Export network to BookSim2 anynet format
bool digraph::export_booksim2_anynet(const std::string &filename,
                                     const std::string &routing_type,
                                     bool optimized) const {
  // Create two files: main config file and the anynet topology file
  std::string anynet_file = filename + "_anynet_file";

  // Create the anynet topology file first
  std::ofstream anynet_os(anynet_file);
  if (!anynet_os.is_open()) {
    return false;
  }

  // Create router ID mapping
  std::vector<std::vector<int>> router_ids(_num_chips_y,
                                           std::vector<int>(_num_chips_x));
  for (size_t y = 0; y < _num_chips_y; y++) {
    for (size_t x = 0; x < _num_chips_x; x++) {
      router_ids[y][x] = y * _num_chips_x + x;
    }
  }

  // Process all routers and their connections
  for (size_t y = 0; y < _num_chips_y; y++) {
    for (size_t x = 0; x < _num_chips_x; x++) {
      // Router ID for current chip
      int router_id = router_ids[y][x];

      // Start a line for this router and its node
      anynet_os << "router " << router_id << " node " << router_id;

      // Check connection to north neighbor
      if (y > 0 && _adj.get_bit(node_idx(x, y, north_out),
                                node_idx(x, y - 1, south_in))) {
        int neighbor_id = router_ids[y - 1][x];
        anynet_os << " router " << neighbor_id;
      }

      // Check connection to east neighbor
      if (x < _num_chips_x - 1 &&
          _adj.get_bit(node_idx(x, y, east_out), node_idx(x + 1, y, west_in))) {
        int neighbor_id = router_ids[y][x + 1];
        anynet_os << " router " << neighbor_id;
      }

      // Check connection to south neighbor
      if (y < _num_chips_y - 1 && _adj.get_bit(node_idx(x, y, south_out),
                                               node_idx(x, y + 1, north_in))) {
        int neighbor_id = router_ids[y + 1][x];
        anynet_os << " router " << neighbor_id;
      }

      // Check connection to west neighbor
      if (x > 0 &&
          _adj.get_bit(node_idx(x, y, west_out), node_idx(x - 1, y, east_in))) {
        int neighbor_id = router_ids[y][x - 1];
        anynet_os << " router " << neighbor_id;
      }

      anynet_os << std::endl;
    }
  }

  anynet_os.close();

  // Now create the main configuration file
  std::ofstream config_os(filename);
  if (!config_os.is_open()) {
    return false;
  }

  // Write the BookSim2 configuration
  config_os << "// BookSim2 anynet configuration generated from digraph"
            << std::endl;
  config_os << "// Routing type: " << routing_type
            << ", Optimized: " << (optimized ? "true" : "false") << std::endl;
  config_os << "// Grid size: " << _num_chips_y << " rows x " << _num_chips_x
            << " columns" << std::endl;
  config_os << std::endl;

  // Add standard BookSim2 configuration parameters
  config_os << "// Standard BookSim2 configuration parameters" << std::endl;
  config_os << "topology = anynet;" << std::endl;
  config_os << "routing_function = min;" << std::endl;
  config_os << "network_file = " << anynet_file << ";" << std::endl;
  config_os << "num_vcs = 4;" << std::endl;
  config_os << "vc_buf_size = 4;" << std::endl;
  config_os << "wait_for_tail_credit = 1;" << std::endl;
  config_os << std::endl;
  config_os << "// Traffic pattern" << std::endl;
  config_os << "traffic = uniform;" << std::endl;
  config_os << "injection_rate = 0.1;" << std::endl;
  config_os << "sim_type = latency;" << std::endl;
  config_os << "sample_period = 10000;" << std::endl;
  config_os << "sim_count = 1;" << std::endl;
  config_os << "max_samples = 10;" << std::endl;

  config_os.close();
  return true;
}

void digraph::PerformanceStats::reset() {
  total_time = std::chrono::duration<int64_t, std::micro>{0};
  path_computation_time = std::chrono::duration<int64_t, std::micro>{0};
  edge_add_time = std::chrono::duration<int64_t, std::micro>{0};
  cycle_check_time = std::chrono::duration<int64_t, std::micro>{0};
  update_path_loop1_time = std::chrono::duration<int64_t, std::micro>{0};
  update_path_loop2_time = std::chrono::duration<int64_t, std::micro>{0};
  edge_additions = 0;
  path_computations = 0;
  cycle_checks = 0;
  update_path_loop1_calls = 0;
  update_path_loop2_calls = 0;
}

void digraph::PerformanceStats::print() const {
  std::cout << "Digraph Performance Stats:\n";
  std::cout << "  Total time: " << total_time.count() / 1000.0 << " ms\n";
  std::cout << "  Path computation: " << path_computation_time.count() / 1000.0
            << " ms (" << path_computations << " calls)\n";
  std::cout << "  Edge additions: " << edge_add_time.count() / 1000.0 << " ms ("
            << edge_additions << " calls)\n";
  std::cout << "  Cycle checks: " << cycle_check_time.count() / 1000.0
            << " ms (" << cycle_checks << " calls)\n";
  std::cout << "  update_path loop1: "
            << update_path_loop1_time.count() / 1000.0 << " ms ("
            << update_path_loop1_calls << " calls)\n";
  std::cout << "  update_path loop2: "
            << update_path_loop2_time.count() / 1000.0 << " ms ("
            << update_path_loop2_calls << " calls)\n";
}
