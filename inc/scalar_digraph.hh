#include <vector>

class scalar_digraph {
private:
  unsigned _n;
  std::vector<std::vector<bool>> _adj;
  std::vector<std::vector<bool>> _path;

  void update_path(unsigned i, unsigned j) {
    // Update i with paths from j
    for (unsigned k = 0; k < _n; ++k) {
      if (_path[j][k])
        _path[i][k] = true;
    }

    // Update all nodes that had path to i
    for (unsigned k = 0; k < _n; ++k) {
      if (_path[k][i]) {
        for (unsigned l = 0; l < _n; ++l) {
          if (_path[i][l])
            _path[k][l] = true;
        }
      }
    }
  }

public:
  scalar_digraph(unsigned n)
      : _n(n), _adj(n, std::vector<bool>(n, false)),
        _path(n, std::vector<bool>(n, false)) {}

  void addedge(unsigned i, unsigned j) {
    if (i >= _n || j >= _n || _adj[i][j])
      return;
    _adj[i][j] = true;
    _path[i][j] = true;
    update_path(i, j);
  }

  bool addacyclic_edge(unsigned i, unsigned j) {
    if (i >= _n || j >= _n || _adj[i][j] || _path[j][i])
      return false;
    addedge(i, j);
    return true;
  }

  bool check_cycle() const {
    for (unsigned i = 0; i < _n; i++) {
      if (_path[i][i])
        return true;
    }
    return false;
  }

  bool path(unsigned i, unsigned j) const {
    return i < _n && j < _n && _path[i][j];
  }

  unsigned count_edges() const {
    unsigned count = 0;
    for (unsigned i = 0; i < _n; i++) {
      for (unsigned j = 0; j < _n; j++) {
        if (_adj[i][j])
          count++;
      }
    }
    return count;
  }

  unsigned count_edge_closure() const {
    unsigned count = 0;
    for (unsigned i = 0; i < _n; i++) {
      for (unsigned j = 0; j < _n; j++) {
        if (_path[i][j])
          count++;
      }
    }
    return count;
  }
};
