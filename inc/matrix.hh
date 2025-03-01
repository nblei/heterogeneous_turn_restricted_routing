#ifndef __BIT_MATRIX_HH__
#define __BIT_MATRIX_HH__
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <vector>

struct alignas(32) avx_block {
  __m256i val;
  avx_block() : val(_mm256_setzero_si256()) {}
  avx_block(const __m256i &v) : val(v) {}
  operator __m256i() const { return val; }
  uint64_t popcount() const {
    const uint64_t *values = (const uint64_t *)&val;
    return _mm_popcnt_u64(values[0]) + _mm_popcnt_u64(values[1]) +
           _mm_popcnt_u64(values[2]) + _mm_popcnt_u64(values[3]);
  };
};

class bit_matrix {
private:
  unsigned _rows;
  unsigned _cols;
  unsigned _width; // Width in AVX blocks
  std::vector<avx_block> _data;
  static constexpr size_t avx_size = 256;
  static constexpr size_t quad_size = 64;

  inline unsigned get_block(unsigned col) const { return col / avx_size; }
  inline unsigned get_offset(unsigned col) const { return col % avx_size; }
  inline unsigned get_idx(unsigned row, unsigned col) const {
    return row * _width + get_block(col);
  }
  inline unsigned get_subarray_idx(unsigned col) const {
    return get_offset(col) / quad_size;
  }
  inline unsigned long long get_mask(unsigned col) const {
    return 1ULL << (get_offset(col) % 64);
  }

public:
  bit_matrix(unsigned rows, unsigned cols)
      : _rows(rows), _cols(cols), _width((cols + avx_size - 1) / avx_size) {
    _data.resize(_rows * _width);
    std::fill(_data.begin(), _data.end(), _mm256_setzero_si256());
  }

  // Get the raw data
  std::vector<avx_block> &data() { return _data; }
  const std::vector<avx_block> &data() const { return _data; }

  // Basic operations
  void set_bit(unsigned row, unsigned col) {
    if (row >= _rows || col >= _cols)
      return;
    unsigned idx = get_idx(row, col);
    unsigned long long mask = get_mask(col);
    avx_block current = _data[idx];
    unsigned long long *values = (unsigned long long *)&current;
    values[get_subarray_idx(col)] |= mask;
    _data[idx] = current;
  }

  // Basic operations
  void clear_bit(unsigned row, unsigned col) {
    if (row >= _rows || col >= _cols)
      return;
    unsigned idx = get_idx(row, col);
    unsigned long long mask = ~get_mask(col);
    avx_block current = _data[idx];
    unsigned long long *values = (unsigned long long *)&current;
    values[get_subarray_idx(col)] &= mask;
    _data[idx] = current;
  }

  bool get_bit(unsigned row, unsigned col) const {
    if (row >= _rows || col >= _cols)
      return false;
    unsigned idx = get_idx(row, col);
    unsigned long long mask = get_mask(col);
    const unsigned long long *values = (const unsigned long long *)&_data[idx];
    return values[get_subarray_idx(col)] & mask;
  }

  // Matrix operations
  bit_matrix operator&(const bit_matrix &other) const {
    bit_matrix result(_rows, _cols);
    for (unsigned i = 0; i < _data.size(); i++) {
      result._data[i].val = _mm256_and_si256(_data[i].val, other._data[i].val);
    }
    return result;
  }

  bit_matrix operator|(const bit_matrix &other) const {
    bit_matrix result(_rows, _cols);
    for (unsigned i = 0; i < _data.size(); i++) {
      result._data[i].val = _mm256_or_si256(_data[i].val, other._data[i].val);
    }
    return result;
  }

  bit_matrix operator~() const {
    bit_matrix result(_rows, _cols);
    for (unsigned i = 0; i < _data.size(); i++) {
      result._data[i].val =
          _mm256_xor_si256(_data[i].val, _mm256_set1_epi32(-1));
    }
    return result;
  }

  bit_matrix andnot(const bit_matrix &other) const {
    bit_matrix result(_rows, _cols);
    for (unsigned i = 0; i < _data.size(); i++) {
      result._data[i].val =
          _mm256_andnot_si256(_data[i].val, other._data[i].val);
    }
    return std::move(result);
  }

  // Utility functions
  unsigned count_ones() const {
    unsigned count = 0;
    for (const auto &block : _data) {
      count += block.popcount();
    }
    return count;
  }

  void print() const {
    std::cout << "Matrix " << _rows << "x" << _cols << ":\n";
    for (unsigned i = 0; i < _rows; i++) {
      for (unsigned j = 0; j < _cols; j++) {
        std::cout << (get_bit(i, j) ? "1" : "0");
        if (j < _cols - 1)
          std::cout << " ";
      }
      std::cout << "\n";
    }
  }

  void print_compact() const {
    std::cout << "Matrix " << _rows << "x" << _cols
              << " (showing only non-zero elements):\n";
    for (unsigned i = 0; i < _rows; i++) {
      bool row_has_ones = false;
      for (unsigned j = 0; j < _cols; j++) {
        if (get_bit(i, j)) {
          if (!row_has_ones) {
            std::cout << "Row " << std::setw(4) << i << ": ";
            row_has_ones = true;
          }
          std::cout << std::setw(4) << j << " ";
        }
      }
      if (row_has_ones)
        std::cout << "\n";
    }
  }

  // Dimensions
  unsigned rows() const { return _rows; }
  unsigned cols() const { return _cols; }
  unsigned width() const { return _width; }
  
  // Get a reference to a specific row's data
  const std::vector<avx_block> get_row_data(unsigned row) const {
    if (row >= _rows) {
      return std::vector<avx_block>();
    }
    
    std::vector<avx_block> row_data(_width);
    for (unsigned i = 0; i < _width; i++) {
      row_data[i] = _data[row * _width + i];
    }
    return row_data;
  }
  
  /**
   * Fast collect indices of all set bits in a column
   * @param col The column to check
   * @param indices Vector to store row indices where bits are set
   */
  void collect_column_indices(unsigned col, std::vector<unsigned>& indices) const {
    if (col >= _cols) {
      return;
    }
    
    unsigned block_idx = get_block(col);
    unsigned long long mask = get_mask(col);
    unsigned subarray_idx = get_subarray_idx(col);
    
    // Process each row
    for (unsigned row = 0; row < _rows; row++) {
      unsigned idx = row * _width + block_idx;
      const unsigned long long *values = (const unsigned long long *)&_data[idx];
      if (values[subarray_idx] & mask) {
        indices.push_back(row);
      }
    }
  }
  
  /**
   * Fast collect indices of all set bits in a column using SIMD
   * @param col The column to check
   * @param indices Vector to store row indices where bits are set
   */
  void collect_column_indices_fast(unsigned col, std::vector<unsigned>& indices) const {
    if (col >= _cols) {
      return;
    }
    
    unsigned block_idx = get_block(col);
    unsigned bit_offset = get_offset(col);
    unsigned subarray_idx = get_subarray_idx(col);
    unsigned bit_in_subarray = bit_offset % 64;
    unsigned long long mask = 1ULL << bit_in_subarray;
    
    // For larger matrices with many set bits, this can be faster
    // Use a temporary buffer to avoid frequent reallocation
    indices.reserve(_rows / 4); // Reserve some space - assume 1/4th of rows have the bit set
    
    // Use a larger batch size for better SIMD utilization
    constexpr unsigned batch_size = 16;
    constexpr unsigned prefetch_distance = 4; // Number of batches to prefetch ahead
    
    // Process rows in batches
    unsigned row;
    for (row = 0; row + batch_size <= _rows; row += batch_size) {
      // Prefetch data for upcoming batches to reduce cache misses
      if (row + batch_size + prefetch_distance * batch_size <= _rows) {
        for (unsigned p = 1; p <= prefetch_distance; p++) {
          unsigned prefetch_row = row + p * batch_size;
          unsigned prefetch_idx = prefetch_row * _width + block_idx;
          __builtin_prefetch(&_data[prefetch_idx], 0, 0); // Read-only, low temporal locality
        }
      }
      
      unsigned long long batch_results = 0;
      
      // Check rows in a batch
      for (unsigned offset = 0; offset < batch_size; offset++) {
        unsigned idx = (row + offset) * _width + block_idx;
        const unsigned long long *values = (const unsigned long long *)&_data[idx];
        if (values[subarray_idx] & mask) {
          batch_results |= (1ULL << offset);
        }
      }
      
      // Process the results
      while (batch_results) {
        unsigned offset = __builtin_ctzll(batch_results); // Count trailing zeros
        indices.push_back(row + offset);
        batch_results &= ~(1ULL << offset); // Clear the bit
      }
    }
    
    // Handle remaining rows
    for (; row < _rows; row++) {
      unsigned idx = row * _width + block_idx;
      const unsigned long long *values = (const unsigned long long *)&_data[idx];
      if (values[subarray_idx] & mask) {
        indices.push_back(row);
      }
    }
  }
  
  // Check if a bit is set in a row data vector
  bool is_bit_set_in_row_data(const std::vector<avx_block>& row_data, unsigned col) const {
    if (col >= _cols) {
      return false;
    }
    
    unsigned block = get_block(col);
    unsigned long long mask = get_mask(col);
    const unsigned long long *values = (const unsigned long long *)&row_data[block];
    return values[get_subarray_idx(col)] & mask;
  }
  
  /**
   * Thread-safe bit setter when each thread operates on a different row.
   * IMPORTANT: This method is only thread-safe when multiple threads set bits
   * in different rows. Concurrent writes to the same row require external locking.
   */
  void set_bit_thread_safe_row(unsigned row, unsigned col) {
    // This method is identical to set_bit() but makes thread-safety assumptions explicit
    // It documents that it's safe when threads operate on different rows
    // If the implementation changes, this method should be updated or removed
    set_bit(row, col);
  }
  
  /**
   * SIMD-accelerated row operation - perform a bitwise OR of one row into another
   * This is thread-safe as long as different threads operate on different target rows
   */
  void row_bitwise_or(unsigned target_row, unsigned source_row) {
    if (target_row >= _rows || source_row >= _rows) {
      return;
    }
    
    // Starting indices for the target and source rows
    unsigned target_idx = target_row * _width;
    unsigned source_idx = source_row * _width;
    
    // Use AVX instructions to OR entire blocks at once
    for (unsigned i = 0; i < _width; i++) {
      // Load the AVX blocks
      __m256i target_block = _data[target_idx + i].val;
      __m256i source_block = _data[source_idx + i].val;
      
      // Perform OR operation with AVX instruction
      _data[target_idx + i].val = _mm256_or_si256(target_block, source_block);
    }
  }
  
  /**
   * Conditional row OR operation - where we only set bits that match a mask
   * @param target_row Row to update (destination)
   * @param source_row Row containing bits to be copied
   * @param mask_row Row containing a mask (only copy bits where mask is set)
   */
  void row_masked_or(unsigned target_row, unsigned source_row, const std::vector<avx_block>& mask_data) {
    if (target_row >= _rows || source_row >= _rows || mask_data.size() < _width) {
      return;
    }
    
    // Starting indices for the target and source rows
    unsigned target_idx = target_row * _width;
    unsigned source_idx = source_row * _width;
    
    // Use AVX instructions to perform masked OR operation
    for (unsigned i = 0; i < _width; i++) {
      // Load the AVX blocks
      __m256i target_block = _data[target_idx + i].val;
      __m256i source_block = _data[source_idx + i].val;
      __m256i mask_block = mask_data[i].val;
      
      // AND the source with the mask, then OR with target
      __m256i masked_source = _mm256_and_si256(source_block, mask_block);
      _data[target_idx + i].val = _mm256_or_si256(target_block, masked_source);
    }
  }
  
  /**
   * Apply a source row as a mask to update a target row
   * This is effectively: if any bit is set in source_row, set the same bit in target_row
   * This performs the same operation as:
   *   for (each col) { if (source_row.get_bit(col)) target_row.set_bit(col); }
   * But uses SIMD for performance
   */
  void apply_row_mask(unsigned target_row, const std::vector<avx_block>& source_data) {
    if (target_row >= _rows || source_data.size() < _width) {
      return;
    }
    
    // Starting index for the target row
    unsigned target_idx = target_row * _width;
    
    // Use AVX instructions to apply the mask
    for (unsigned i = 0; i < _width; i++) {
      // Load the AVX blocks
      __m256i target_block = _data[target_idx + i].val;
      __m256i source_block = source_data[i].val;
      
      // Perform OR operation with AVX instruction
      _data[target_idx + i].val = _mm256_or_si256(target_block, source_block);
    }
  }
  
  /**
   * Optimized version that only updates bits that aren't already set
   * Only performs writes if there's new bits to set, reducing unnecessary memory writes
   *
   * @param target_row Row to update
   * @param source_data Source row data to copy bits from
   * @return true if any bits were updated, false if no changes made
   */
  bool apply_row_mask_efficient(unsigned target_row, const std::vector<avx_block>& source_data) {
    if (target_row >= _rows || source_data.size() < _width) {
      return false;
    }
    
    bool changed = false;
    
    // Starting index for the target row
    unsigned target_idx = target_row * _width;
    
    // Use AVX instructions to apply the mask
    for (unsigned i = 0; i < _width; i++) {
      // Load the AVX blocks
      __m256i target_block = _data[target_idx + i].val;
      __m256i source_block = source_data[i].val;
      
      // See what bits would be newly set (~target_block & source_block)
      __m256i target_complement = _mm256_xor_si256(target_block, _mm256_set1_epi32(-1));
      __m256i new_bits = _mm256_and_si256(source_block, target_complement);
      
      // Only update if there are new bits to set
      if (!_mm256_testz_si256(new_bits, new_bits)) {
        // Perform OR operation with AVX instruction
        _data[target_idx + i].val = _mm256_or_si256(target_block, source_block);
        changed = true;
      }
    }
    
    return changed;
  }
};

#endif
