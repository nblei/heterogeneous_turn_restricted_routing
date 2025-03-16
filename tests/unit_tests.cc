#include "digraph.hh"
#include "matrix.hh"
#include <gtest/gtest.h>

// Test bit_matrix basic operations
class BitMatrixTest : public ::testing::Test {
protected:
  bit_matrix m{4, 4}; // 4x4 matrix for basic tests
};

TEST_F(BitMatrixTest, InitialState) {
  for (unsigned i = 0; i < 4; i++) {
    for (unsigned j = 0; j < 4; j++) {
      EXPECT_FALSE(m.get_bit(i, j));
    }
  }
}

TEST_F(BitMatrixTest, SetAndGetBit) {
  m.set_bit(1, 2);
  EXPECT_TRUE(m.get_bit(1, 2));
  EXPECT_FALSE(m.get_bit(2, 1));
}

TEST_F(BitMatrixTest, BitwiseOperations) {
  bit_matrix a(2, 2), b(2, 2);

  a.set_bit(0, 0);
  a.set_bit(1, 1);
  b.set_bit(0, 0);
  b.set_bit(0, 1);

  // Test AND
  bit_matrix c = a & b;
  EXPECT_TRUE(c.get_bit(0, 0));
  EXPECT_FALSE(c.get_bit(0, 1));
  EXPECT_FALSE(c.get_bit(1, 1));

  // Test OR
  bit_matrix d = a | b;
  EXPECT_TRUE(d.get_bit(0, 0));
  EXPECT_TRUE(d.get_bit(0, 1));
  EXPECT_TRUE(d.get_bit(1, 1));

  // Test NOT
  bit_matrix e = ~a;
  EXPECT_FALSE(e.get_bit(0, 0));
  EXPECT_TRUE(e.get_bit(0, 1));
  EXPECT_TRUE(e.get_bit(1, 0));
  EXPECT_FALSE(e.get_bit(1, 1));
}

TEST_F(BitMatrixTest, CountOnes) {
  m.set_bit(0, 0);
  m.set_bit(1, 1);
  m.set_bit(2, 2);
  EXPECT_EQ(m.count_ones(), 3);
}

// Test digraph functionality
class DigraphTest : public ::testing::Test {
protected:
  digraph d{2, 2}; // 2x2 mesh for basic tests

  // Helper function to get node indices
  size_t node_idx(size_t chip_x, size_t chip_y, digraph::internal_node n) {
    return (chip_x + 2 * chip_y) * 10 + n;
  }
};

TEST_F(DigraphTest, InitialState) {
  // Test that required edges are present
  EXPECT_GT(d.count_edges(), 0);
  EXPECT_FALSE(d.check_cycle());
}

TEST_F(DigraphTest, AddAcyclicEdge) {
  // Try to add a valid routing edge: north_in -> east_out in chip (0,0)
  size_t src = node_idx(0, 0, digraph::north_in);
  size_t dst = node_idx(0, 0, digraph::east_out);

  std::cout << "Testing edge from north_in to east_out in chip (0,0)\n";
  std::cout << "Source node: " << src << " (north_in)\n";
  std::cout << "Dest node: " << dst << " (east_out)\n";

  auto addable = d.get_addable_edges();
  std::cout << "Addable edges matrix:\n";
  addable.print_compact();

  unsigned initial_edges = d.count_edges();
  bool success = d.addacyclic_edge(src, dst);

  EXPECT_TRUE(success);
  EXPECT_EQ(d.count_edges(), initial_edges + 1);
}

TEST_F(DigraphTest, PreventCycles) {
  // Try to create a cycle using valid routing edges
  size_t north_in_00 = node_idx(0, 0, digraph::north_in);
  size_t east_out_00 = node_idx(0, 0, digraph::east_out);
  size_t west_in_10 = node_idx(1, 0, digraph::west_in);
  size_t south_out_10 = node_idx(1, 0, digraph::south_out);

  // Add first edge
  EXPECT_TRUE(d.addacyclic_edge(north_in_00, east_out_00));

  // Add second edge
  EXPECT_TRUE(d.addacyclic_edge(west_in_10, south_out_10));

  // Try to add edge that would create cycle
  // This should fail as it would create a cycle
  EXPECT_FALSE(d.addacyclic_edge(south_out_10, north_in_00));
}

TEST_F(DigraphTest, PathComputation) {
  // Test path computation with valid routing edges
  size_t north_in_00 = node_idx(0, 0, digraph::north_in);
  size_t east_out_00 = node_idx(0, 0, digraph::east_out);
  size_t west_in_10 = node_idx(1, 0, digraph::west_in);
  size_t south_out_10 = node_idx(1, 0, digraph::south_out);

  // Add edges to create a path
  EXPECT_TRUE(d.addacyclic_edge(north_in_00, east_out_00));
  EXPECT_TRUE(d.addacyclic_edge(west_in_10, south_out_10));

  // The complete path requires three segments:
  // 1. north_in_00 to east_out_00 (added manually)
  // 2. east_out_00 to west_in_10 (added during grid initialization)
  // 3. west_in_10 to south_out_10 (added manually)
  //
  // We've fixed the issue by ensuring that the direct path bits are set
  // when edges are added, which allows transitive paths to be correctly calculated.

  // Check the path exists
  EXPECT_TRUE(d.path(north_in_00, south_out_10));
}
