#ifndef __ROUTING_OPTIMIZER_HH__
#define __ROUTING_OPTIMIZER_HH__

#include "digraph.hh"
#include <chrono>
#include <queue>
#include <stack>
#include <unordered_set>

struct PerformanceStats {
  std::chrono::duration<int64_t, std::micro> total_time{0};
  std::chrono::duration<int64_t, std::micro> path_analysis_time{0};
  std::chrono::duration<int64_t, std::micro> edge_prioritization_time{0};
  std::chrono::duration<int64_t, std::micro> path_completion_time{0};
  std::chrono::duration<int64_t, std::micro> calculate_edge_priority_time{0};
  std::chrono::duration<int64_t, std::micro> optimize_additional_edges_time{0};
  
  size_t edges_added{0};
  size_t path_analyses{0};
  size_t edge_prioritizations{0};
  size_t path_completions{0};
  size_t edge_priority_calculations{0};
  
  void reset() {
    total_time = std::chrono::duration<int64_t, std::micro>{0};
    path_analysis_time = std::chrono::duration<int64_t, std::micro>{0};
    edge_prioritization_time = std::chrono::duration<int64_t, std::micro>{0};
    path_completion_time = std::chrono::duration<int64_t, std::micro>{0};
    calculate_edge_priority_time = std::chrono::duration<int64_t, std::micro>{0};
    optimize_additional_edges_time = std::chrono::duration<int64_t, std::micro>{0};
    
    edges_added = 0;
    path_analyses = 0;
    edge_prioritizations = 0;
    path_completions = 0;
    edge_priority_calculations = 0;
  }
  
  void print() const {
    std::cout << "Routing Optimizer Performance Stats:\n";
    std::cout << "  Total time: " << total_time.count() / 1000.0 << " ms\n";
    std::cout << "  Path analysis: " << path_analysis_time.count() / 1000.0 
              << " ms (" << path_analyses << " calls)\n";
    std::cout << "  Edge prioritization: " << edge_prioritization_time.count() / 1000.0 
              << " ms (" << edge_prioritizations << " calls)\n";
    std::cout << "  └─ Calculate edge priority: " << calculate_edge_priority_time.count() / 1000.0
              << " ms (" << edge_priority_calculations << " calls)\n";
    std::cout << "  Path completion: " << path_completion_time.count() / 1000.0 
              << " ms (" << path_completions << " calls)\n";
    std::cout << "  Edges added: " << edges_added << "\n";
  }
};

class RoutingOptimizer {
private:
  struct PathTarget {
    size_t src_x, src_y;
    size_t dst_x, dst_y;
    bool operator==(const PathTarget &other) const {
      return src_x == other.src_x && src_y == other.src_y &&
             dst_x == other.dst_x && dst_y == other.dst_y;
    }
  };

  struct EdgeCandidate {
    unsigned src, dst;
    double priority;

    bool operator<(const EdgeCandidate &other) const {
      return priority < other.priority;
    }
  };

  digraph &graph;
  std::vector<PathTarget> missing_paths;
  std::priority_queue<EdgeCandidate> edge_candidates;
  PerformanceStats _perf_stats;

  void analyze_missing_paths() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    missing_paths.clear();

    // Find all CPU pairs without paths
    for (size_t src_x = 0; src_x < graph.NumChipsX(); src_x++) {
      for (size_t src_y = 0; src_y < graph.NumChipsY(); src_y++) {
        for (size_t dst_x = 0; dst_x < graph.NumChipsX(); dst_x++) {
          for (size_t dst_y = 0; dst_y < graph.NumChipsY(); dst_y++) {
            if (src_x == dst_x && src_y == dst_y)
              continue;

            if (!graph.has_cpu_path(src_x, src_y, dst_x, dst_y)) {
              missing_paths.push_back({src_x, src_y, dst_x, dst_y});
            }
          }
        }
      }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    _perf_stats.path_analysis_time += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    _perf_stats.path_analyses++;
  }

  void prioritize_edges() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    edge_candidates = std::priority_queue<EdgeCandidate>();
    bit_matrix addable = graph.get_addable_edges();

    // For each addable edge, calculate its priority based on:
    // 1. How many missing paths it might help complete
    // 2. Manhattan distance to target CPUs
    // 3. Current path closure properties
    for (unsigned i = 0; i < graph._n; i++) {
      for (unsigned j = 0; j < graph._n; j++) {
        if (addable.get_bit(i, j)) {
          double priority = calculate_edge_priority(i, j);
          edge_candidates.push({i, j, priority});
        }
      }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    _perf_stats.edge_prioritization_time += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    _perf_stats.edge_prioritizations++;
  }

  double calculate_edge_priority(unsigned src, unsigned dst) {
    auto start_time = std::chrono::high_resolution_clock::now();
    _perf_stats.edge_priority_calculations++;
    
    double priority = 0.0;

    // Analyze how this edge might help complete missing paths
    for (const auto &target : missing_paths) {
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
      priority += 1.0 / manhattan_dist;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    _perf_stats.calculate_edge_priority_time += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
    return priority;
  }

  bool try_complete_path(const PathTarget &target) {
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
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
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
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        _perf_stats.path_completions++;
        return false;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    _perf_stats.path_completion_time += 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    _perf_stats.path_completions++;
    return true;
  }
  
  bool is_turn_edge(unsigned src, unsigned dst) const {
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
  
  // Count turn edges in the graph - to track our progress
  unsigned count_turn_edges() {
    unsigned count = 0;
    for (unsigned i = 0; i < graph._n; i++) {
      for (unsigned j = 0; j < graph._n; j++) {
        if (graph._adj.get_bit(i, j) && is_turn_edge(i, j)) {
          count++;
        }
      }
    }
    return count;
  }

public:
  RoutingOptimizer(digraph &g) : graph(g) {}
  
  void print_perf_stats() const {
    _perf_stats.print();
  }
  
  const PerformanceStats& get_perf_stats() const {
    return _perf_stats;
  }
  
  void reset_perf_stats() {
    _perf_stats.reset();
  }

  bool optimize() {
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    _perf_stats.reset();
    
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

      // Try to complete each missing path
      for (const auto &target : missing_paths) {
        if (try_complete_path(target)) {
          progress = true;
          break;
        }
      }
    }
    
    // Phase 2: Add more turn edges to maximize connectivity
    auto optimize_start = std::chrono::high_resolution_clock::now();
    
    // Get all potential edges that could be added without creating cycles
    bit_matrix addable = graph.get_addable_edges();
    
    // Try to add each viable edge
    for (unsigned i = 0; i < graph._n; i++) {
      for (unsigned j = 0; j < graph._n; j++) {
        if (addable.get_bit(i, j) && is_turn_edge(i, j)) {
          if (graph.addacyclic_edge(i, j)) {
            _perf_stats.edges_added++;
          }
        }
      }
    }
    
    auto optimize_end = std::chrono::high_resolution_clock::now();
    _perf_stats.optimize_additional_edges_time += 
        std::chrono::duration_cast<std::chrono::microseconds>(optimize_end - optimize_start);
    
    // Report how many edges were added
    unsigned final_turn_count = count_turn_edges();
    if (final_turn_count > initial_turn_count) {
      std::cout << "Added " << (final_turn_count - initial_turn_count) 
                << " additional turn edges (from " << initial_turn_count 
                << " to " << final_turn_count << " turns)\n";
    }
    
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    _perf_stats.total_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(overall_end_time - overall_start_time);
    
    return graph.check_all_cpu_paths();
  }
};

#endif
