#ifndef __ROUTING_OPTIMIZER_HH__
#define __ROUTING_OPTIMIZER_HH__

#include "digraph.hh"
#include <chrono>
#include <atomic>
#include <queue>
#include <random>

struct PerformanceStats {
  // Execution times
  std::chrono::duration<int64_t, std::micro> total_time{0};
  std::chrono::duration<int64_t, std::micro> path_analysis_time{0};
  std::chrono::duration<int64_t, std::micro> edge_prioritization_time{0};
  std::chrono::duration<int64_t, std::micro> path_completion_time{0};
  std::chrono::duration<int64_t, std::micro> calculate_edge_priority_time{0};
  std::chrono::duration<int64_t, std::micro> optimize_additional_edges_time{0};
  std::chrono::duration<int64_t, std::micro> get_addable_edges_time{0};

  // Counters
  size_t edges_added{0};
  size_t path_analyses{0};
  size_t edge_prioritizations{0};
  size_t path_completions{0};
  size_t edge_priority_calculations{0};
  size_t get_addable_edges_calls{0};

  void reset();
  void print() const;
};

class RoutingOptimizer {
private:
  struct PathTarget {
    size_t src_x, src_y;
    size_t dst_x, dst_y;
    bool operator==(const PathTarget &other) const;
  };

  struct EdgeCandidate {
    unsigned src, dst;
    double priority;

    bool operator<(const EdgeCandidate &other) const;
  };

  digraph &graph;
  std::vector<PathTarget> missing_paths;
  std::priority_queue<EdgeCandidate> edge_candidates;
  PerformanceStats _perf_stats;

  void analyze_missing_paths();
  void prioritize_edges();
  double calculate_edge_priority(unsigned src, unsigned dst);
  bool try_complete_path(const PathTarget &target);
  bool is_turn_edge(unsigned src, unsigned dst) const;
  unsigned count_turn_edges();

public:
  RoutingOptimizer(digraph &g);

  void print_perf_stats() const;
  const PerformanceStats &get_perf_stats() const;
  void reset_perf_stats();

  bool optimize_random();
  bool optimize();
};

#endif