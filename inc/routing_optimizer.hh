#ifndef __ROUTING_OPTIMIZER_HH__
#define __ROUTING_OPTIMIZER_HH__

#include "digraph.hh"
#include <queue>
#include <stack>
#include <unordered_set>

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

  void analyze_missing_paths() {
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
  }

  void prioritize_edges() {
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
  }

  double calculate_edge_priority(unsigned src, unsigned dst) {
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

    return priority;
  }

  bool try_complete_path(const PathTarget &target) {
    std::stack<unsigned> edge_stack;
    size_t src_cpu =
        graph.node_idx(target.src_x, target.src_y, digraph::cpu_out);
    size_t dst_cpu =
        graph.node_idx(target.dst_x, target.dst_y, digraph::cpu_in);

    // Try to find a sequence of edges that completes this path
    while (!graph.path(src_cpu, dst_cpu)) {
      // Get highest priority edge that might help
      if (edge_candidates.empty())
        return false;

      auto candidate = edge_candidates.top();
      edge_candidates.pop();

      // Try adding the edge
      if (graph.addacyclic_edge(candidate.src, candidate.dst)) {
        edge_stack.push(candidate.src);
        edge_stack.push(candidate.dst);
      }

      // If we've tried too many edges without success, backtrack
      if (edge_stack.size() > 20) { // Arbitrary limit, tune based on testing
        while (!edge_stack.empty()) {
          // Would need to add edge removal functionality to digraph
          edge_stack.pop();
        }
        return false;
      }
    }

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

public:
  RoutingOptimizer(digraph &g) : graph(g) {}

  bool optimize() {
    bool progress = true;
    while (progress) {
      progress = false;

      // Analyze current state
      analyze_missing_paths();
      if (missing_paths.empty())
        return true;

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

    return graph.check_all_cpu_paths();
  }
};

#endif
