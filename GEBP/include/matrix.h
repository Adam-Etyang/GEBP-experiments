#pragma once

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
};
