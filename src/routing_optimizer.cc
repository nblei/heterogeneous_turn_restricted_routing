#include "routing_optimizer.hh"
#include <stack>
#include <unordered_set>
#include <omp.h>
#include <iostream>

void PerformanceStats::reset() {
  // Execution times
  total_time = std::chrono::duration<int64_t, std::micro>{0};
  path_analysis_time = std::chrono::duration<int64_t, std::micro>{0};
  edge_prioritization_time = std::chrono::duration<int64_t, std::micro>{0};
  path_completion_time = std::chrono::duration<int64_t, std::micro>{0};
  calculate_edge_priority_time = std::chrono::duration<int64_t, std::micro>{0};
  optimize_additional_edges_time = std::chrono::duration<int64_t, std::micro>{0};
  get_addable_edges_time = std::chrono::duration<int64_t, std::micro>{0};

  // Counters
  edges_added = 0;
  path_analyses = 0;
  edge_prioritizations = 0;
  path_completions = 0;
  edge_priority_calculations = 0;
  get_addable_edges_calls = 0;
}

void PerformanceStats::print() const {
  std::cout << "Routing Optimizer Performance Stats:\n";
  std::cout << "  Total time: " << total_time.count() / 1000.0 << " ms\n";
  std::cout << "  Path analysis: " << path_analysis_time.count() / 1000.0
            << " ms (" << path_analyses << " calls)\n";
  std::cout << "  Edge prioritization: "
            << edge_prioritization_time.count() / 1000.0 << " ms ("
            << edge_prioritizations << " calls)\n";
  std::cout << "  └─ Calculate edge priority: "
            << calculate_edge_priority_time.count() / 1000.0 << " ms ("
            << edge_priority_calculations << " calls)\n";
  std::cout << "  Path completion: " << path_completion_time.count() / 1000.0
            << " ms (" << path_completions << " calls)\n";
  std::cout << "  Get addable edges: "
            << get_addable_edges_time.count() / 1000.0 << " ms ("
            << get_addable_edges_calls << " calls)\n";
  std::cout << "  Edges added: " << edges_added << "\n";
}

bool RoutingOptimizer::PathTarget::operator==(const PathTarget &other) const {
  return src_x == other.src_x && src_y == other.src_y &&
          dst_x == other.dst_x && dst_y == other.dst_y;
}

bool RoutingOptimizer::EdgeCandidate::operator<(const EdgeCandidate &other) const {
  return priority < other.priority;
}

RoutingOptimizer::RoutingOptimizer(digraph &g) : graph(g) {}

void RoutingOptimizer::print_perf_stats() const { 
  _perf_stats.print(); 
}

const PerformanceStats &RoutingOptimizer::get_perf_stats() const { 
  return _perf_stats; 
}

void RoutingOptimizer::reset_perf_stats() { 
  _perf_stats.reset(); 
}

void RoutingOptimizer::analyze_missing_paths() {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Create thread-local vectors to collect missing paths
  std::vector<std::vector<PathTarget>> thread_missing_paths;

  ////#pragma omp parallel
  {
    // Initialize thread-local storage
    // #pragma omp single
    {
      thread_missing_paths.resize(omp_get_num_threads());
    }

    int thread_id = omp_get_thread_num();

    // Parallelize the nested loops
    // #pragma omp for collapse(2) schedule(dynamic)
    for (size_t src_x = 0; src_x < graph.NumChipsX(); src_x++) {
      for (size_t src_y = 0; src_y < graph.NumChipsY(); src_y++) {
        // These loops can remain sequential within each thread
        for (size_t dst_x = 0; dst_x < graph.NumChipsX(); dst_x++) {
          for (size_t dst_y = 0; dst_y < graph.NumChipsY(); dst_y++) {
            if (src_x == dst_x && src_y == dst_y)
              continue;

            if (!graph.has_cpu_path(src_x, src_y, dst_x, dst_y)) {
              thread_missing_paths[thread_id].push_back(
                  {src_x, src_y, dst_x, dst_y});
            }
          }
        }
      }
    }
  }

  // Combine all thread-local results
  missing_paths.clear();
  for (const auto &thread_paths : thread_missing_paths) {
    missing_paths.insert(missing_paths.end(), thread_paths.begin(),
                        thread_paths.end());
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.path_analysis_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                          start_time);
  _perf_stats.path_analyses++;
}

void RoutingOptimizer::prioritize_edges() {
  auto start_time = std::chrono::high_resolution_clock::now();

  bit_matrix addable = graph.get_addable_edges();

  // Use bit-parallel scanning to identify potential edges
  // First determine approximate count of edges to process
  size_t total_addable_edges = 0;

  // Use SIMD to count total bits set in addable matrix
  // #pragma omp parallel reduction(+ : total_addable_edges)
  {
    // #pragma omp for
    for (unsigned i = 0; i < addable.data().size(); i++) {
      total_addable_edges += addable.data()[i].popcount();
    }
  }

  // Pre-allocate vector for better performance - avoids reallocations
  std::vector<std::pair<unsigned, unsigned>> potential_edges;
  potential_edges.reserve(total_addable_edges);

  // Create thread-local vectors to collect edges in parallel
  std::vector<std::vector<std::pair<unsigned, unsigned>>> thread_edges;

  // Use bit-parallel scanning to collect edges efficiently
  // #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();

    // Initialize thread-local storage
    // #pragma omp single
    {
      thread_edges.resize(omp_get_num_threads());
      for (auto &vec : thread_edges) {
        vec.reserve(total_addable_edges / omp_get_num_threads() +
                    100); // Add some buffer
      }
    }

    // Process rows in parallel
    // #pragma omp for schedule(dynamic, 16)
    for (unsigned i = 0; i < graph._n; i++) {
      // Get all bits set in this row using SIMD operations
      const auto row_data = addable.get_row_data(i);

      // Process each block in the row
      for (unsigned block_idx = 0; block_idx < row_data.size(); block_idx++) {
        // Skip empty blocks for efficiency
        if (row_data[block_idx].popcount() == 0)
          continue;

        // Process each 64-bit segment in the block (4 segments per 256-bit
        // AVX block)
        const uint64_t *segments =
            reinterpret_cast<const uint64_t *>(&row_data[block_idx]);

        for (unsigned seg = 0; seg < 4; seg++) {
          uint64_t segment = segments[seg];
          if (segment == 0)
            continue; // Skip empty segments

          // Base column index for this segment
          unsigned base_col = block_idx * 256 + seg * 64;

          // Process all bits set in this segment
          while (segment) {
            // Find position of least significant bit that is set
            unsigned bit_pos = __builtin_ctzll(segment);

            // Calculate actual column index
            unsigned j = base_col + bit_pos;

            // Add edge if within matrix bounds
            if (j < graph._n) {
              thread_edges[thread_id].emplace_back(i, j);
            }

            // Clear the bit we just processed
            segment &= (segment - 1); // Clear the least significant bit
          }
        }
      }
    }
  }

  // Combine thread-local edge vectors
  for (const auto &edges : thread_edges) {
    potential_edges.insert(potential_edges.end(), edges.begin(), edges.end());
  }

  // Create thread-local priority queues to avoid contention
  std::vector<std::priority_queue<EdgeCandidate>> thread_candidates;

  // #pragma omp parallel
  {
    // Initialize thread-local storage
    // #pragma omp single
    {
      thread_candidates.resize(omp_get_num_threads());
    }

    int thread_id = omp_get_thread_num();

    // Process edges in parallel with better cache locality
    // Group edges by source to improve cache utilization
    // #pragma omp for schedule(dynamic, 64)
    for (size_t idx = 0; idx < potential_edges.size(); idx++) {
      auto [i, j] = potential_edges[idx];

      // Prefetch data for the next edge calculation
      if (idx + 1 < potential_edges.size()) {
        auto [next_i, next_j] = potential_edges[idx + 1];
        __builtin_prefetch(&graph._path.data()[next_i * graph._path.width()],
                          0, 0);
        __builtin_prefetch(&graph._path.data()[next_j * graph._path.width()],
                          0, 0);
      }

      double priority = calculate_edge_priority(i, j);
      thread_candidates[thread_id].push({i, j, priority});
    }
  }

  // Merge all thread-local queues into the main queue
  // Use a more efficient merge algorithm to avoid repeated push/pop
  // operations
  edge_candidates = std::priority_queue<EdgeCandidate>();
  std::vector<EdgeCandidate> all_candidates;
  all_candidates.reserve(total_addable_edges);

  for (auto &local_queue : thread_candidates) {
    while (!local_queue.empty()) {
      all_candidates.push_back(local_queue.top());
      local_queue.pop();
    }
  }

  // Build final priority queue in one operation
  for (const auto &candidate : all_candidates) {
    edge_candidates.push(candidate);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.edge_prioritization_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                          start_time);
  _perf_stats.edge_prioritizations++;
}

double RoutingOptimizer::calculate_edge_priority(unsigned src, unsigned dst) {
  auto start_time = std::chrono::high_resolution_clock::now();

  double priority = 0.0;

  // Early optimization: Cache path test results for source and destination
  const auto &src_path_row = graph._path.get_row_data(src);
  const auto &dst_path_col =
      graph._path.get_row_data(dst); // We'll use transpose later

  // Constants for SIMD processing
  constexpr size_t VECTOR_SIZE = 8; // Process 8 paths at once with AVX2

  // Prepare batched processing for SIMD
  const size_t num_targets = missing_paths.size();
  const size_t num_full_batches = num_targets / VECTOR_SIZE;
  const size_t remaining = num_targets % VECTOR_SIZE;

  // Process targets in SIMD-friendly batches
  // #pragma omp parallel reduction(+ : priority)
  {
    double thread_priority = 0.0;

    // #pragma omp for schedule(dynamic, 4)
    for (size_t batch = 0; batch < num_full_batches; batch++) {
      // Process 8 targets at once
      alignas(32) int batch_src_x[VECTOR_SIZE];
      alignas(32) int batch_src_y[VECTOR_SIZE];
      alignas(32) int batch_dst_x[VECTOR_SIZE];
      alignas(32) int batch_dst_y[VECTOR_SIZE];
      alignas(32) size_t batch_cpu_src[VECTOR_SIZE];
      alignas(32) size_t batch_cpu_dst[VECTOR_SIZE];
      alignas(32) bool batch_has_path_to_src[VECTOR_SIZE];
      alignas(32) bool batch_has_path_from_dst[VECTOR_SIZE];
      alignas(32) int batch_manhattan_dist[VECTOR_SIZE];

      // Load data for this batch
      for (size_t i = 0; i < VECTOR_SIZE; i++) {
        const size_t idx = batch * VECTOR_SIZE + i;
        const auto &target = missing_paths[idx];

        // Store coordinates
        batch_src_x[i] = target.src_x;
        batch_src_y[i] = target.src_y;
        batch_dst_x[i] = target.dst_x;
        batch_dst_y[i] = target.dst_y;

        // Compute CPU indices
        batch_cpu_src[i] =
            graph.node_idx(target.src_x, target.src_y, digraph::cpu_out);
        batch_cpu_dst[i] =
            graph.node_idx(target.dst_x, target.dst_y, digraph::cpu_in);

        // Prefetch path data for next iteration
        if (idx + VECTOR_SIZE < num_targets) {
          const auto &next_target = missing_paths[idx + VECTOR_SIZE];
          size_t next_cpu_src = graph.node_idx(
              next_target.src_x, next_target.src_y, digraph::cpu_out);
          size_t next_cpu_dst = graph.node_idx(
              next_target.dst_x, next_target.dst_y, digraph::cpu_in);

          __builtin_prefetch(
              &graph._path.data()[next_cpu_src * graph._path.width()], 0, 0);
          __builtin_prefetch(
              &graph._path.data()[next_cpu_dst * graph._path.width()], 0, 0);
        }
      }

// Compute manhattan distances using SIMD
#ifdef __AVX2__
      // Load source coordinates into AVX registers
      __m256i vx_src = _mm256_load_si256((__m256i *)batch_src_x);
      __m256i vy_src = _mm256_load_si256((__m256i *)batch_src_y);

      // Load destination coordinates into AVX registers
      __m256i vx_dst = _mm256_load_si256((__m256i *)batch_dst_x);
      __m256i vy_dst = _mm256_load_si256((__m256i *)batch_dst_y);

      // Calculate differences: dst_x - src_x and dst_y - src_y
      __m256i vx_diff = _mm256_sub_epi32(vx_dst, vx_src);
      __m256i vy_diff = _mm256_sub_epi32(vy_dst, vy_src);

      // Calculate absolute values
      __m256i vx_abs = _mm256_abs_epi32(vx_diff);
      __m256i vy_abs = _mm256_abs_epi32(vy_diff);

      // Sum absolutes to get manhattan distance
      __m256i vdist = _mm256_add_epi32(vx_abs, vy_abs);

      // Store results
      _mm256_store_si256((__m256i *)batch_manhattan_dist, vdist);
#else
      // Fallback to scalar code
      for (size_t i = 0; i < VECTOR_SIZE; i++) {
        batch_manhattan_dist[i] =
            std::abs((int)batch_dst_x[i] - (int)batch_src_x[i]) +
            std::abs((int)batch_dst_y[i] - (int)batch_src_y[i]);
      }
#endif

      // Check path connectivity in parallel for all targets in batch
      for (size_t i = 0; i < VECTOR_SIZE; i++) {
        batch_has_path_to_src[i] = graph.path(batch_cpu_src[i], src);
        batch_has_path_from_dst[i] = graph.path(dst, batch_cpu_dst[i]);
      }

      // Compute priorities for all targets in batch
      for (size_t i = 0; i < VECTOR_SIZE; i++) {
        // Check if this edge could be part of a path between CPUs
        if (batch_has_path_to_src[i] && batch_has_path_from_dst[i]) {
          // Higher priority if it connects previously disconnected components
          thread_priority += 10.0;
        }

        // Add distance-based priority, avoiding division by zero
        if (batch_manhattan_dist[i] > 0) {
          thread_priority += 1.0 / batch_manhattan_dist[i];
        }
      }
    }

    // Add to priority
    priority += thread_priority;
  }

  // Handle remaining targets (less than a full batch)
  for (size_t i = num_full_batches * VECTOR_SIZE; i < num_targets; i++) {
    const auto &target = missing_paths[i];
    size_t cpu_src =
        graph.node_idx(target.src_x, target.src_y, digraph::cpu_out);
    size_t cpu_dst =
        graph.node_idx(target.dst_x, target.dst_y, digraph::cpu_in);

    // Check if this edge could be part of a path between CPUs
    if (graph.path(cpu_src, src) && graph.path(dst, cpu_dst)) {
      // Higher priority if it connects previously disconnected components
      priority += 10.0;
    }

    // Add distance-based priority
    int manhattan_dist = std::abs((int)target.dst_x - (int)target.src_x) +
                          std::abs((int)target.dst_y - (int)target.src_y);
    if (manhattan_dist > 0) {
      priority += 1.0 / manhattan_dist;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.calculate_edge_priority_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                          start_time);
  _perf_stats.edge_priority_calculations++;

  return priority;
}

bool RoutingOptimizer::try_complete_path(const PathTarget &target) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::stack<unsigned> edge_stack;
  size_t src_cpu =
      graph.node_idx(target.src_x, target.src_y, digraph::cpu_out);
  size_t dst_cpu =
      graph.node_idx(target.dst_x, target.dst_y, digraph::cpu_in);

  // Try to find a sequence of edges that completes this path
  while (!graph.path(src_cpu, dst_cpu)) {
    // Get highest priority edge that might help
    if (edge_candidates.empty()) {
      auto end_time = std::chrono::high_resolution_clock::now();
      _perf_stats.path_completion_time +=
          std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                start_time);
      _perf_stats.path_completions++;
      return false;
    }

    auto candidate = edge_candidates.top();
    edge_candidates.pop();

    // Try adding the edge
    if (graph.addacyclic_edge(candidate.src, candidate.dst)) {
      _perf_stats.edges_added++;
      edge_stack.push(candidate.src);
      edge_stack.push(candidate.dst);
    }

    // If we've tried too many edges without success, backtrack
    if (edge_stack.size() > 20) { // Arbitrary limit, tune based on testing
      while (!edge_stack.empty()) {
        // Would need to add edge removal functionality to digraph
        edge_stack.pop();
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      _perf_stats.path_completion_time +=
          std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                start_time);
      _perf_stats.path_completions++;
      return false;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.path_completion_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                            start_time);
  _perf_stats.path_completions++;
  return true;
}

bool RoutingOptimizer::is_turn_edge(unsigned src, unsigned dst) const {
  // Helper to determine if an edge is a turn edge within a chip
  for (size_t x = 0; x < graph._num_chips_x; x++) {
    for (size_t y = 0; y < graph._num_chips_y; y++) {
      // Check if src and dst are in the same chip
      size_t chip_start = (x + graph._num_chips_x * y) * digraph::total_nodes;
      size_t chip_end = chip_start + digraph::total_nodes;

      if (src >= chip_start && src < chip_end && dst >= chip_start &&
          dst < chip_end) {

        // Check if it's a valid turn (in_port -> out_port)
        size_t src_offset = src - chip_start;
        size_t dst_offset = dst - chip_start;

        bool is_in_port = src_offset <= digraph::west_in;
        bool is_out_port = dst_offset >= digraph::north_out &&
                            dst_offset <= digraph::west_out;

        // Exclude "reverse" turns
        if (is_in_port && is_out_port) {
          // Check it's not a reverse direction
          if ((src_offset == digraph::north_in &&
                dst_offset == digraph::south_out) ||
              (src_offset == digraph::south_in &&
                dst_offset == digraph::north_out) ||
              (src_offset == digraph::east_in &&
                dst_offset == digraph::west_out) ||
              (src_offset == digraph::west_in &&
                dst_offset == digraph::east_out)) {
            return false;
          }
          return true;
        }
      }
    }
  }
  return false;
}

unsigned RoutingOptimizer::count_turn_edges() {
  unsigned total_count = 0;

  // #pragma omp parallel reduction(+ : total_count)
  {
    unsigned thread_count = 0;

    // #pragma omp for collapse(2) schedule(dynamic)
    for (unsigned i = 0; i < graph._n; i++) {
      for (unsigned j = 0; j < graph._n; j++) {
        if (graph._adj.get_bit(i, j) && is_turn_edge(i, j)) {
          thread_count++;
        }
      }
    }

    total_count += thread_count;
  }

  return total_count;
}

bool RoutingOptimizer::optimize_random() {
  auto overall_start_time = std::chrono::high_resolution_clock::now();
  _perf_stats.reset();
  graph.reset_perf_stats();

  // Get the initial number of turn edges
  unsigned initial_turn_count = count_turn_edges();

  // Setup random number generator with a fixed seed for reproducibility
  std::mt19937_64 rng(42);

  // Keep adding random edges until no more edges can be added without
  // creating cycles
  bool added_any;
  unsigned total_edges_added = 0;

  do {
    added_any = false;

    // Get all potential edges that could be added without creating cycles
    auto get_edges_start = std::chrono::high_resolution_clock::now();
    bit_matrix addable = graph.get_addable_edges();
    auto get_edges_end = std::chrono::high_resolution_clock::now();
    _perf_stats.get_addable_edges_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            get_edges_end - get_edges_start);
    _perf_stats.get_addable_edges_calls++;

    // Count total addable edges
    unsigned total_addable = addable.count_ones();
    if (total_addable == 0) {
      break; // No more edges can be added without creating cycles
    }

    // Generate random index
    std::uniform_int_distribution<unsigned> dist(0, total_addable - 1);
    unsigned target = dist(rng);

    // Find the target edge
    unsigned count = 0;
    unsigned block_idx = 0;

    for (const auto block : addable.data()) {
      unsigned increment = block.popcount();
      unsigned next_count = count + increment;
      if (next_count > target) {
        // Found the block containing our target edge
        // Calculate how many bits we need to skip within this block
        unsigned target_in_block = target - count;
        
        // Get row and column based on block index
        unsigned row = block_idx / addable.width();
        unsigned base_col = (block_idx % addable.width()) * 256; // 256 bits per AVX block
        
        // Extract the bits from the block to find the target bit
        const uint64_t *segments = reinterpret_cast<const uint64_t *>(&block);
        
        // Track position within the block
        unsigned pos_in_block = 0;
        unsigned seg = 0;
        
        // Scan through segments in the block until we find our target bit
        while (seg < 4) { // Each AVX block has 4 64-bit segments
          uint64_t segment = segments[seg];
          unsigned segment_pop = _mm_popcnt_u64(segment);
          
          if (pos_in_block + segment_pop > target_in_block) {
            // Target bit is in this segment
            unsigned bit_to_skip = target_in_block - pos_in_block;
            
            // Find the position of the target bit
            uint64_t mask = segment;
            for (unsigned i = 0; i < bit_to_skip; i++) {
              // Clear the least significant set bit
              mask &= mask - 1;
            }
            
            // Get the position of the least significant bit that's still set
            unsigned bit_pos = __builtin_ctzll(mask);
            
            // Calculate final column
            unsigned col = base_col + seg * 64 + bit_pos;
            
            // Add this edge to the graph
            if (graph.addacyclic_edge(row, col)) {
              added_any = true;
              total_edges_added++;
              
              // Print progress every 100 edges added
              if (total_edges_added % 100 == 0) {
                std::cout << "Added " << total_edges_added << " edges so far...\n";
              }
            }
            
            break;
          }
          
          pos_in_block += segment_pop;
          seg++;
        }
        
        break;
      }
      block_idx += 1;
      count = next_count;
    }
  } while (added_any);

  _perf_stats.edges_added = total_edges_added;

  // Report how many edges were added
  unsigned final_turn_count = count_turn_edges();
  std::cout << "Total edges added: " << total_edges_added << "\n";
  if (final_turn_count > initial_turn_count) {
    std::cout << "Turn edges: " << initial_turn_count << " -> "
              << final_turn_count << " (+"
              << (final_turn_count - initial_turn_count) << ")\n";
  }

  auto overall_end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          overall_end_time - overall_start_time);

  return graph.check_all_cpu_paths();
}

bool RoutingOptimizer::optimize() {
  auto overall_start_time = std::chrono::high_resolution_clock::now();
  _perf_stats.reset();
  graph.reset_perf_stats();

  // Get the initial number of turn edges
  unsigned initial_turn_count = count_turn_edges();

  // Phase 1: Establish full connectivity
  bool progress = true;
  while (progress) {
    progress = false;

    // Analyze current state
    analyze_missing_paths();
    if (missing_paths.empty())
      break; // We have full connectivity

    // Prioritize potential edges
    prioritize_edges();

    // Try to complete paths in parallel
    // Use a shared atomic flag to track if any thread made progress
    std::atomic<bool> any_progress{false};

    // Process paths in chunks to enable parallel exploration
    // while still maintaining some ordering preference (higher priority paths
    // first)
    const size_t chunk_size = std::min(size_t(16), missing_paths.size());

    for (size_t chunk_start = 0;
          chunk_start < missing_paths.size() && !any_progress;
          chunk_start += chunk_size) {

      size_t chunk_end =
          std::min(chunk_start + chunk_size, missing_paths.size());

      // //#pragma omp parallel for schedule(dynamic, 1)
      for (size_t i = chunk_start; i < chunk_end; i++) {
        // Skip if another thread has already made progress
        if (!any_progress.load(std::memory_order_relaxed)) {
          if (try_complete_path(missing_paths[i])) {
            any_progress.store(true, std::memory_order_relaxed);
          }
        }
      }

      // If any thread made progress, we'll break out of the outer loop and
      // start over
      if (any_progress.load()) {
        progress = true;
        break;
      }
    }
  }

  // Phase 2: Add more turn edges to maximize connectivity
  auto optimize_start = std::chrono::high_resolution_clock::now();

  // Get all potential edges that could be added without creating cycles
  bit_matrix addable = graph.get_addable_edges();

  // Collect all potential turn edges first to enable parallel processing
  std::vector<std::pair<unsigned, unsigned>> potential_edges;

  // First pass to determine size needed
  size_t potential_count = 0;
  // //#pragma omp parallel for collapse(2) reduction(+ : potential_count)            \
//     schedule(dynamic)
  for (unsigned i = 0; i < graph._n; i++) {
    for (unsigned j = 0; j < graph._n; j++) {
      if (addable.get_bit(i, j) && is_turn_edge(i, j)) {
        potential_count++;
      }
    }
  }

  // Pre-allocate storage for better performance
  potential_edges.reserve(potential_count);

  // Create thread-local vectors to avoid contention
  std::vector<std::vector<std::pair<unsigned, unsigned>>> thread_edges;

  // //#pragma omp parallel
  {
    // Initialize thread-local storage
    // //#pragma omp single
    {
      thread_edges.resize(omp_get_num_threads());
    }

    int thread_id = omp_get_thread_num();

    // Collect edges in parallel
    // //#pragma omp for collapse(2) schedule(dynamic)
    for (unsigned i = 0; i < graph._n; i++) {
      for (unsigned j = 0; j < graph._n; j++) {
        if (addable.get_bit(i, j) && is_turn_edge(i, j)) {
          thread_edges[thread_id].emplace_back(i, j);
        }
      }
    }
  }

  // Combine all thread-local results
  for (const auto &local_edges : thread_edges) {
    potential_edges.insert(potential_edges.end(), local_edges.begin(),
                          local_edges.end());
  }

  // Use atomic counter to track added edges across threads
  std::atomic<int> added_count{0};

  // Process edges in parallel using OpenMP
  // //#pragma omp parallel
  {
    // Each thread will have its own local edge counter
    int thread_added = 0;

    // Distribute iterations across threads
    // //#pragma omp for schedule(dynamic, 64)
    for (size_t idx = 0; idx < potential_edges.size(); idx++) {
      auto [i, j] = potential_edges[idx];
      if (graph.addacyclic_edge(i, j)) {
        thread_added++;
      }
    }

    // Atomically add the thread's count to the total
    added_count += thread_added;
  }

  // Update performance stats with the total count
  _perf_stats.edges_added = added_count.load();

  auto optimize_end = std::chrono::high_resolution_clock::now();
  _perf_stats.optimize_additional_edges_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(optimize_end -
                                                          optimize_start);

  // Report how many edges were added
  unsigned final_turn_count = count_turn_edges();
  if (final_turn_count > initial_turn_count) {
    std::cout << "Added " << (final_turn_count - initial_turn_count)
              << " additional turn edges (from " << initial_turn_count
              << " to " << final_turn_count << " turns)\n";
  }

  auto overall_end_time = std::chrono::high_resolution_clock::now();
  _perf_stats.total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          overall_end_time - overall_start_time);

  return graph.check_all_cpu_paths();
}