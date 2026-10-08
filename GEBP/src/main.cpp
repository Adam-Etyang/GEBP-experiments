#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "matrix.h"

int main(int argc, char *argv[]) {
  int size = 512;
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

  if (A.cols != B.rows) {
    std::cerr << "dimension error" << std::endl;
    return -1;
  }

  Matrix C{A.rows, B.cols, Major::row_major,
           std::vector<double>(element_count, 0.0)};

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A(i, j) = static_cast<double>((i + j) % 100) / 100.0;
      B(i, j) = static_cast<double>((i * 3 + j * 7) % 100) / 100.0;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < B.cols; j++) {
      for (int k = 0; k < A.cols; k++) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();

  double checksum = 0.0;
  for (double value : C.data) {
    checksum += value;
  }

  const std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Matrix size: " << size << " x " << size << '\n'
            << "GEMM time: " << elapsed.count() << " ms\n"
            << "Checksum: " << checksum << '\n';

  return 0;
}
