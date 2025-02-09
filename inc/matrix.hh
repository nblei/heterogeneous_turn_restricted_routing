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
};

#endif
