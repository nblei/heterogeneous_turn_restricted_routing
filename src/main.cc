#include "digraph.hh"
#include "routing_optimizer.hh"
#include "scalar_digraph.hh"
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>
void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name << " rows cols\n"
            << "  rows: Number of rows in the mesh (positive integer)\n"
            << "  cols: Number of columns in the mesh (positive integer)\n";
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse arguments
  int rows = std::atoi(argv[1]);
  int cols = std::atoi(argv[2]);

  digraph d(rows, cols); // Create a 4x4 mesh
  // Print initial state
  std::cout << "\nInitial state:\n";
  d.print_stats();

  // Set up random number generator
  std::random_device rd;
  std::mt19937_64 rng(rd());
  RoutingOptimizer opt(d);

  opt.optimize();

  // Print final state
  std::cout << "\nAfter adding random edges:\n";
  d.print_stats();
  std::cout << "Best Homogeneous: " << rows * cols * 6 << "\n";
  return 0;
}
