#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

enum class Major { row_major, col_major };

struct Matrix {
  int rows;
  int cols;
  Major layout;

  std::vector<double> data;

  double &operator()(int r, int c) {
    return data[layout == Major::row_major ? r * cols + c : c * rows + r];
  }

  double operator()(int r, int c) const {
    return data[layout == Major::row_major ? r * cols + c : c * rows + r];
  }

  Matrix block(int r0, int c0, int block_rows, int block_cols) const {

    if (r0 < 0 || c0 < 0 || r0 + block_rows > rows || c0 + block_cols > cols) {
      std::cerr << "block Out of bounds" << std::endl;
    }
    Matrix B{block_rows, block_cols, Major::row_major,
             std::vector<double>(block_rows * block_cols)};
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        B(i, j) = (*this)(r0 + i, c0 + j);
      }
    }
    return B;
  }

  void add_inplace(const Matrix &Y) {
    if (rows != Y.rows || cols != Y.cols) {
      throw std::invalid_argument("add size mismatch");
    }
    for (size_t i = 0; i < data.size(); i++) {
      data[i] += Y.data[i];
    }
  }

  void set_block(int r0, int c0, const Matrix &B) {
    for (int i = 0; i < B.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        (*this)(r0 + i, c0 + j) = B(i, j);
      }
    }
  }
};
