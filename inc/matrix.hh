#ifndef __BIT_MATRIX_HH__
#define __BIT_MATRIX_HH__
#include <cstdint>
#include <vector>
#include <immintrin.h>

struct alignas(32) avx_block {
  __m256i val;
  avx_block() : val(_mm256_setzero_si256()) {}
  avx_block(const __m256i &v) : val(v) {}
  operator __m256i() const { return val; }
  uint64_t popcount() const;
};

class bit_matrix {
private:
  unsigned _rows;
  unsigned _cols;
  unsigned _width; // Width in AVX blocks
  std::vector<avx_block> _data;
  static constexpr size_t avx_size = 256;
  static constexpr size_t quad_size = 64;

  // Returns block offset within a row --- see get_idx
  inline unsigned get_block(unsigned col) const;
  inline unsigned get_offset(unsigned col) const;
  inline unsigned get_idx(unsigned row, unsigned col) const;
  inline unsigned get_subarray_idx(unsigned col) const;
  inline unsigned long long get_mask(unsigned col) const;

public:
  bit_matrix(unsigned rows, unsigned cols);

  // Get the raw data
  std::vector<avx_block> &data();
  const std::vector<avx_block> &data() const;

  // Basic operations
  void set_bit(unsigned row, unsigned col);
  void clear_bit(unsigned row, unsigned col);
  bool get_bit(unsigned row, unsigned col) const;

  // Matrix operations
  bit_matrix operator&(const bit_matrix &other) const;
  bit_matrix operator|(const bit_matrix &other) const;
  bit_matrix operator~() const;
  bit_matrix andnot(const bit_matrix &other) const;

  // Utility functions
  unsigned count_ones() const;
  void print() const;
  void print_compact() const;

  // Dimensions
  unsigned rows() const;
  unsigned cols() const;
  unsigned width() const;

  // Get a reference to a specific row's data
  const std::vector<avx_block> get_row_data(unsigned row) const;

  /**
   * Fast collect indices of all set bits in a column
   * @param col The column to check
   * @param indices Vector to store row indices where bits are set
   */
  void collect_column_indices(unsigned col, std::vector<unsigned> &indices) const;

  /**
   * Fast collect indices of all set bits in a column using SIMD
   * @param col The column to check
   * @param indices Vector to store row indices where bits are set
   */
  void collect_column_indices_fast(unsigned col, std::vector<unsigned> &indices) const;

  // Check if a bit is set in a row data vector
  bool is_bit_set_in_row_data(const std::vector<avx_block> &row_data, unsigned col) const;

  /**
   * Thread-safe bit setter when each thread operates on a different row.
   * IMPORTANT: This method is only thread-safe when multiple threads set bits
   * in different rows. Concurrent writes to the same row require external
   * locking.
   */
  void set_bit_thread_safe_row(unsigned row, unsigned col);

  /**
   * SIMD-accelerated row operation - perform a bitwise OR of one row into
   * another This is thread-safe as long as different threads operate on
   * different target rows
   */
  void row_bitwise_or(unsigned target_row, unsigned source_row);

  /**
   * Conditional row OR operation - where we only set bits that match a mask
   * @param target_row Row to update (destination)
   * @param source_row Row containing bits to be copied
   * @param mask_row Row containing a mask (only copy bits where mask is set)
   */
  void row_masked_or(unsigned target_row, unsigned source_row, const std::vector<avx_block> &mask_data);

  /**
   * Apply a source row as a mask to update a target row
   * This is effectively: if any bit is set in source_row, set the same bit in
   * target_row This performs the same operation as: for (each col) { if
   * (source_row.get_bit(col)) target_row.set_bit(col); } But uses SIMD for
   * performance
   */
  void apply_row_mask(unsigned target_row, const std::vector<avx_block> &source_data);

  /**
   * Optimized version that only updates bits that aren't already set
   * Only performs writes if there's new bits to set, reducing unnecessary
   * memory writes
   *
   * @param target_row Row to update
   * @param source_data Source row data to copy bits from
   * @return true if any bits were updated, false if no changes made
   */
  bool apply_row_mask_efficient(unsigned target_row, const std::vector<avx_block> &source_data);
};

#endif