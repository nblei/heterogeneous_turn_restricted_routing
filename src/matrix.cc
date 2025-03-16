#include "matrix.hh"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

uint64_t avx_block::popcount() const {
  const uint64_t *values = (const uint64_t *)&val;
  return _mm_popcnt_u64(values[0]) + _mm_popcnt_u64(values[1]) +
         _mm_popcnt_u64(values[2]) + _mm_popcnt_u64(values[3]);
}

inline unsigned bit_matrix::get_block(unsigned col) const { 
  return col / avx_size; 
}

inline unsigned bit_matrix::get_offset(unsigned col) const { 
  return col % avx_size; 
}

inline unsigned bit_matrix::get_idx(unsigned row, unsigned col) const {
  return row * _width + get_block(col);
}

inline unsigned bit_matrix::get_subarray_idx(unsigned col) const {
  return get_offset(col) / quad_size;
}

inline unsigned long long bit_matrix::get_mask(unsigned col) const {
  return 1ULL << (get_offset(col) % 64);
}

bit_matrix::bit_matrix(unsigned rows, unsigned cols)
    : _rows(rows), _cols(cols), _width((cols + avx_size - 1) / avx_size) {
  _data.resize(_rows * _width);
  std::fill(_data.begin(), _data.end(), _mm256_setzero_si256());
}

std::vector<avx_block> &bit_matrix::data() { 
  return _data; 
}

const std::vector<avx_block> &bit_matrix::data() const { 
  return _data; 
}

void bit_matrix::set_bit(unsigned row, unsigned col) {
  if (row >= _rows || col >= _cols)
    return;
  unsigned idx = get_idx(row, col);
  unsigned long long mask = get_mask(col);
  avx_block current = _data[idx];
  unsigned long long *values = (unsigned long long *)&current;
  values[get_subarray_idx(col)] |= mask;
  _data[idx] = current;
}

void bit_matrix::clear_bit(unsigned row, unsigned col) {
  if (row >= _rows || col >= _cols)
    return;
  unsigned idx = get_idx(row, col);
  unsigned long long mask = ~get_mask(col);
  avx_block current = _data[idx];
  unsigned long long *values = (unsigned long long *)&current;
  values[get_subarray_idx(col)] &= mask;
  _data[idx] = current;
}

bool bit_matrix::get_bit(unsigned row, unsigned col) const {
  if (row >= _rows || col >= _cols)
    return false;
  unsigned idx = get_idx(row, col);
  unsigned long long mask = get_mask(col);
  const unsigned long long *values = (const unsigned long long *)&_data[idx];
  return values[get_subarray_idx(col)] & mask;
}

bit_matrix bit_matrix::operator&(const bit_matrix &other) const {
  bit_matrix result(_rows, _cols);
  for (unsigned i = 0; i < _data.size(); i++) {
    result._data[i].val = _mm256_and_si256(_data[i].val, other._data[i].val);
  }
  return result;
}

bit_matrix bit_matrix::operator|(const bit_matrix &other) const {
  bit_matrix result(_rows, _cols);
  for (unsigned i = 0; i < _data.size(); i++) {
    result._data[i].val = _mm256_or_si256(_data[i].val, other._data[i].val);
  }
  return result;
}

bit_matrix bit_matrix::operator~() const {
  bit_matrix result(_rows, _cols);
  for (unsigned i = 0; i < _data.size(); i++) {
    result._data[i].val =
        _mm256_xor_si256(_data[i].val, _mm256_set1_epi32(-1));
  }
  return result;
}

bit_matrix bit_matrix::andnot(const bit_matrix &other) const {
  bit_matrix result(_rows, _cols);
  for (unsigned i = 0; i < _data.size(); i++) {
    result._data[i].val =
        _mm256_andnot_si256(_data[i].val, other._data[i].val);
  }
  return std::move(result);
}

unsigned bit_matrix::count_ones() const {
  unsigned count = 0;
  for (const auto &block : _data) {
    count += block.popcount();
  }
  return count;
}

void bit_matrix::print() const {
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

void bit_matrix::print_compact() const {
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

unsigned bit_matrix::rows() const { 
  return _rows; 
}

unsigned bit_matrix::cols() const { 
  return _cols; 
}

unsigned bit_matrix::width() const { 
  return _width; 
}

const std::vector<avx_block> bit_matrix::get_row_data(unsigned row) const {
  if (row >= _rows) {
    return std::vector<avx_block>();
  }

  std::vector<avx_block> row_data(_width);
  for (unsigned i = 0; i < _width; i++) {
    row_data[i] = _data[row * _width + i];
  }
  return row_data;
}

void bit_matrix::collect_column_indices(unsigned col,
                                        std::vector<unsigned> &indices) const {
  if (col >= _cols) {
    return;
  }

  unsigned block_idx = get_block(col);
  unsigned long long mask = get_mask(col);
  unsigned subarray_idx = get_subarray_idx(col);

  // Process each row
  for (unsigned row = 0; row < _rows; row++) {
    unsigned idx = row * _width + block_idx;
    const unsigned long long *values =
        (const unsigned long long *)&_data[idx];
    if (values[subarray_idx] & mask) {
      indices.push_back(row);
    }
  }
}

void bit_matrix::collect_column_indices_fast(unsigned col,
                                            std::vector<unsigned> &indices) const {
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
  indices.reserve(
      _rows /
      4); // Reserve some space - assume 1/4th of rows have the bit set

  // Use a larger batch size for better SIMD utilization
  constexpr unsigned batch_size = 16;
  constexpr unsigned prefetch_distance =
      4; // Number of batches to prefetch ahead

  // Process rows in batches
  unsigned row;
  for (row = 0; row + batch_size <= _rows; row += batch_size) {
    // Prefetch data for upcoming batches to reduce cache misses
    if (row + batch_size + prefetch_distance * batch_size <= _rows) {
      for (unsigned p = 1; p <= prefetch_distance; p++) {
        unsigned prefetch_row = row + p * batch_size;
        unsigned prefetch_idx = prefetch_row * _width + block_idx;
        __builtin_prefetch(&_data[prefetch_idx], 0,
                          0); // Read-only, low temporal locality
      }
    }

    unsigned long long batch_results = 0;

    // Check rows in a batch
    for (unsigned offset = 0; offset < batch_size; offset++) {
      unsigned idx = (row + offset) * _width + block_idx;
      const unsigned long long *values =
          (const unsigned long long *)&_data[idx];
      if (values[subarray_idx] & mask) {
        batch_results |= (1ULL << offset);
      }
    }

    // Process the results
    while (batch_results) {
      unsigned offset =
          __builtin_ctzll(batch_results); // Count trailing zeros
      indices.push_back(row + offset);
      batch_results &= ~(1ULL << offset); // Clear the bit
    }
  }

  // Handle remaining rows
  for (; row < _rows; row++) {
    unsigned idx = row * _width + block_idx;
    const unsigned long long *values =
        (const unsigned long long *)&_data[idx];
    if (values[subarray_idx] & mask) {
      indices.push_back(row);
    }
  }
}

bool bit_matrix::is_bit_set_in_row_data(const std::vector<avx_block> &row_data,
                                       unsigned col) const {
  if (col >= _cols) {
    return false;
  }

  unsigned block = get_block(col);
  unsigned long long mask = get_mask(col);
  const unsigned long long *values =
      (const unsigned long long *)&row_data[block];
  return values[get_subarray_idx(col)] & mask;
}

void bit_matrix::set_bit_thread_safe_row(unsigned row, unsigned col) {
  // This method is identical to set_bit() but makes thread-safety assumptions
  // explicit It documents that it's safe when threads operate on different
  // rows If the implementation changes, this method should be updated or
  // removed
  set_bit(row, col);
}

void bit_matrix::row_bitwise_or(unsigned target_row, unsigned source_row) {
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

void bit_matrix::row_masked_or(unsigned target_row, unsigned source_row,
                              const std::vector<avx_block> &mask_data) {
  if (target_row >= _rows || source_row >= _rows ||
      mask_data.size() < _width) {
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

void bit_matrix::apply_row_mask(unsigned target_row,
                               const std::vector<avx_block> &source_data) {
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

bool bit_matrix::apply_row_mask_efficient(unsigned target_row,
                                         const std::vector<avx_block> &source_data) {
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
    __m256i target_complement =
        _mm256_xor_si256(target_block, _mm256_set1_epi32(-1));
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