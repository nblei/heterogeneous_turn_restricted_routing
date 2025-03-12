#ifndef __DIGRAPH_HH__
#define __DIGRAPH_HH__
#include "matrix.hh"
#include <chrono>
#include <random>
#include <string>
#include <vector>

class RoutingOptimzer;
class digraph {
public:
  typedef enum : size_t {
    north_in,
    north_out,
    east_in,
    east_out,
    south_in,
    south_out,
    west_in,
    west_out,
    cpu_in,
    cpu_out,
    total_nodes,
  } internal_node;

  struct PerformanceStats {
    // Execution times
    std::chrono::duration<int64_t, std::micro> total_time{0};
    std::chrono::duration<int64_t, std::micro> digraph_create_time{0};
    std::chrono::duration<int64_t, std::micro> initialization_time{0};
    std::chrono::duration<int64_t, std::micro> path_computation_time{0};
    std::chrono::duration<int64_t, std::micro> edge_add_time{0};
    std::chrono::duration<int64_t, std::micro> cycle_check_time{0};
    std::chrono::duration<int64_t, std::micro> update_path_loop1_time{0};
    std::chrono::duration<int64_t, std::micro> update_path_loop2_time{0};
    std::chrono::duration<int64_t, std::micro> get_addable_edges_time{0};

    // Counters
    size_t edge_additions{0};
    size_t path_computations{0};
    size_t cycle_checks{0};
    size_t update_path_loop1_calls{0};
    size_t update_path_loop2_calls{0};
    size_t get_addable_edges_calls{0};

    void reset();
    void print() const;
  };

  friend class RoutingOptimizer;

private:
  unsigned _n;
  unsigned _num_chips_x;
  unsigned _num_chips_y;
  bit_matrix _adj;
  bit_matrix _path;
  bit_matrix _viable_edges;
  PerformanceStats _perf_stats;

  void update_path(unsigned i, unsigned j);
  size_t chip_idx(size_t chip_x, size_t chip_y) const;
  size_t node_idx(size_t chip_x, size_t chip_y, internal_node n) const;
  digraph(unsigned n);
  void initialize_viable_edges();
  void initialize_grid_connections();

public:
  // Initialize with basic grid connectivity but no turns
  digraph(unsigned num_chips_x, unsigned num_chips_y);

  // Initialize with different routing strategies
  static digraph create_xy_routing(unsigned num_chips_x, unsigned num_chips_y);
  static digraph create_west_first_routing(unsigned num_chips_x,
                                           unsigned num_chips_y);
  static digraph create_odd_even_routing(unsigned num_chips_x,
                                         unsigned num_chips_y);

  // Core algorithms
  void compute_paths();
  void addedge(unsigned i, unsigned j);
  bool addacyclic_edge(unsigned i, unsigned j);
  bool check_cycle() const;
  bool check_cycle_and_track();
  bit_matrix get_addable_edges() const;
  bool path(unsigned i, unsigned j) const;

  // Utility methods
  unsigned count_edges() const;
  unsigned count_edge_closure() const;
  void print_matrices() const;
  void debug_print_matrices() const;
  void verify_edge_add(unsigned i, unsigned j) const;
  bool has_cpu_path(size_t src_x, size_t src_y, size_t dst_x,
                    size_t dst_y) const;
  bool check_all_cpu_paths() const;
  void add_all_possible_edges();
  void add_random_edges(std::mt19937_64 &rng);
  unsigned count_chip_turn_edges(size_t chip_x, size_t chip_y) const;
  unsigned count_all_turn_edges() const;
  void print_stats() const;
  void print_perf_stats() const;
  const PerformanceStats &get_perf_stats() const;
  void reset_perf_stats();
  unsigned NumChipsX() const;
  unsigned NumChipsY() const;
  const char *node_type_to_string(internal_node node) const;

  // File I/O
  bool log_turn_edges(const std::string &filename,
                      const std::string &routing_type, bool optimized) const;
  bool export_booksim2_anynet(const std::string &filename,
                              const std::string &routing_type,
                              bool optimized) const;
};
#endif
