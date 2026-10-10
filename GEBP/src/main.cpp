#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix.h"

std::vector<std::pair<int, int>> block_ranges(int total, int n) {
  std::vector<std::pair<int, int>> ranges;
  int rem = total;
  int start = 0;
  for (int i = 0; i < n; i++) {
    int sz = rem / (n - i);
    ranges.push_back({start, sz});
    start += sz;
    rem -= sz;
  }
  return ranges;
}

Matrix matmul(const Matrix &A, const Matrix &B) {
  if (A.cols != B.rows) {
    throw std::invalid_argument("matrix dimensions do not match for GEMM");
  }

  Matrix result{A.rows, B.cols, Major::row_major,
                std::vector<double>(static_cast<std::size_t>(A.rows) * B.cols,
                                    0.0)};
  for (int i = 0; i < A.rows; i++) {
    for (int k = 0; k < A.cols; k++) {
      for (int j = 0; j < B.cols; j++) {
        result(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return result;
}

Matrix matmul(std::vector<Matrix> mats) {
  if (mats.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }
  Matrix result = std::move(mats.front());
  for (size_t i = 1; i < mats.size(); i++) {
    result = matmul(std::move(result), std::move(mats[i]));
  }
  return result;
}

int main(int argc, char *argv[]) {
  int size = 512;
  if (argc > 2) {
    std::cerr << "usage: gebp [positive-matrix-size]\n";
    return 1;
  }
  if (argc > 1) {
    try {
      size = std::stoi(argv[1]);
    } catch (const std::exception &) {
      std::cerr << "usage: gebp [positive-matrix-size]\n";
      return 1;
    }
  }
  if (size <= 0) {
    std::cerr << "matrix size must be positive\n";
    return 1;
  }

  const auto element_count = static_cast<std::size_t>(size) * size;
  Matrix A{size, size, Major::row_major, std::vector<double>(element_count)};
  Matrix B{size, size, Major::row_major, std::vector<double>(element_count)};
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A(i, j) = static_cast<double>((i + j) % 100) / 100.0;
      B(i, j) = static_cast<double>((i * 3 + j * 7) % 100) / 100.0;
    }
  }

  std::vector<Matrix> matrices;
  matrices.reserve(2);
  matrices.push_back(std::move(A));
  matrices.push_back(std::move(B));
  const Matrix result = matmul(std::move(matrices));

  double checksum = 0.0;
  for (double value : result.data) {
    checksum += value;
  }

  std::cout << "Matrix size: " << size << " x " << size << '\n'
            << "Checksum: " << checksum << '\n';

  return 0;
}
