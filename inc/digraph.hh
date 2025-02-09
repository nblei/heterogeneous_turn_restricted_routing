#ifndef __DIGRAPH_HH__
#define __DIGRAPH_HH__
#include "matrix.hh"
#include <cstring>
#include <iostream>
#include <random>

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

  friend class RoutingOptimizer;

private:
  unsigned _n;
  unsigned _num_chips_x;
  unsigned _num_chips_y;
  bit_matrix _adj;
  bit_matrix _path;
  bit_matrix _viable_edges;

  void update_path(unsigned i, unsigned j) {
    // Update paths from i based on paths from j
    for (unsigned k = 0; k < _n; k++) {
      if (_path.get_bit(j, k)) {
        _path.set_bit(i, k);
      }
    }

    // For each node that had a path to i, update its paths
    for (unsigned k = 0; k < _n; k++) {
      if (_path.get_bit(k, i)) {
        for (unsigned l = 0; l < _n; l++) {
          if (_path.get_bit(i, l)) {
            _path.set_bit(k, l);
          }
        }
      }
    }
  }

  size_t chip_idx(size_t chip_x, size_t chip_y) const {
    return total_nodes * (chip_x + _num_chips_x * chip_y);
  }

  size_t node_idx(size_t chip_x, size_t chip_y, internal_node n) const {
    return chip_idx(chip_x, chip_y) + n;
  }

  digraph(unsigned n) : _n(n), _adj(n, n), _path(n, n), _viable_edges(n, n) {}

public:
  digraph(unsigned num_chips_x, unsigned num_chips_y)
      : digraph(num_chips_x * num_chips_y * total_nodes) {
    _num_chips_x = num_chips_x;
    _num_chips_y = num_chips_y;

    // Add viable chip-internal edges
    for (int chip_x = 0; chip_x < num_chips_x; chip_x++) {
      for (int chip_y = 0; chip_y < num_chips_y; chip_y++) {
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

        // Add required edges from cardinal in ports to cpu_in
        _adj.set_bit(node_idx(chip_x, chip_y, north_in),
                     node_idx(chip_x, chip_y, cpu_in));
        _adj.set_bit(node_idx(chip_x, chip_y, east_in),
                     node_idx(chip_x, chip_y, cpu_in));
        _adj.set_bit(node_idx(chip_x, chip_y, south_in),
                     node_idx(chip_x, chip_y, cpu_in));
        _adj.set_bit(node_idx(chip_x, chip_y, west_in),
                     node_idx(chip_x, chip_y, cpu_in));
      }
    }

    // Add inter-chip edges for the entire grid
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
    compute_paths();
  }

  // Warshall's algorithm
  void compute_paths() {
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
  }

  void addedge(unsigned i, unsigned j) {

    if (i >= _n || j >= _n || _adj.get_bit(i, j)) {
      return;
    }

    unsigned old_count = _adj.count_ones();
    _adj.set_bit(i, j);
    unsigned new_count = _adj.count_ones();

    // Recompute paths after adding edge
    update_path(i, j);
  }

  bool addacyclic_edge(unsigned i, unsigned j) {

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

  bool check_cycle() const {
    for (unsigned i = 0; i < _n; i++) {
      if (_path.get_bit(i, i))
        return true;
    }
    return false;
  }

  bit_matrix get_addable_edges() const {
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

  bool path(unsigned i, unsigned j) const {
    return i < _n && j < _n && _path.get_bit(i, j);
  }

  unsigned count_edges() const { return _adj.count_ones(); }

  unsigned count_edge_closure() const { return _path.count_ones(); }

  void print_matrices() const {
    std::cout << "\nAdjacency Matrix:\n";
    _adj.print_compact();
    std::cout << "\nPath Matrix:\n";
    _path.print_compact();
    std::cout << "\nViable Edges Matrix:\n";
    _viable_edges.print_compact();
  }

  void debug_print_matrices() const {
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

  void verify_edge_add(unsigned i, unsigned j) const {
    std::cout << "Verifying edge (" << i << ", " << j << "):\n";
    std::cout << "In bounds: " << (i < _n && j < _n) << "\n";
    std::cout << "Already in adj: " << _adj.get_bit(i, j) << "\n";
    std::cout << "Would create cycle: " << _path.get_bit(j, i) << "\n";
    std::cout << "Is viable: " << _viable_edges.get_bit(i, j) << "\n";
  }

  bool has_cpu_path(size_t src_x, size_t src_y, size_t dst_x,
                    size_t dst_y) const {
    size_t src = node_idx(src_x, src_y, cpu_out);
    size_t dst = node_idx(dst_x, dst_y, cpu_in);
    return path(src, dst);
  }

  // Check if all required CPU-to-CPU paths exist
  bool check_all_cpu_paths() const {
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
  void add_all_possible_edges() {
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
  void add_random_edges(std::mt19937_64 &rng) {
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
  unsigned count_chip_turn_edges(size_t chip_x, size_t chip_y) const {
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
  unsigned count_all_turn_edges() const {
    unsigned total = 0;
    for (size_t x = 0; x < _num_chips_x; x++) {
      for (size_t y = 0; y < _num_chips_y; y++) {
        total += count_chip_turn_edges(x, y);
      }
    }
    return total;
  }

  // Get statistics about the current graph
  void print_stats() const {
    std::cout << "Graph Statistics:\n";
    std::cout << "Dimensions: " << _num_chips_x << "x" << _num_chips_y
              << " chips\n";
    std::cout << "Total nodes: " << _n << "\n";
    std::cout << "Total edges: " << count_edges() << "\n";
    std::cout << "Path closure size: " << count_edge_closure() << "\n";
    std::cout << "Is acyclic: " << (!check_cycle() ? "yes" : "no") << "\n";
    std::cout << "All CPU paths exist: "
              << (check_all_cpu_paths() ? "yes" : "no") << "\n";
    std::cout << "Total internal turn edges: " << count_all_turn_edges()
              << "\n";
  }

  unsigned NumChipsX() const { return _num_chips_x; }
  unsigned NumChipsY() const { return _num_chips_y; }
};
#endif
