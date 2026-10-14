#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <initializer_list>

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

  Matrix result{
      A.rows, B.cols, Major::row_major,
      std::vector<double>(static_cast<std::size_t>(A.rows) * B.cols, 0.0)};

  for (int i = 0; i < A.rows; i++) {
    for (int k = 0; k < A.cols; k++) {
      for (int j = 0; j < B.cols; j++) {
        result(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return result;
}

Matrix matmul(std::initializer_list<Matrix> matrix) {
  if (matrix.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }
  auto it = matrix.begin();
  Matrix result = *it;
  for (it++; it != matrix.end(); it++) {
    result = matmul(result, *it);
  }
  return result;
}

Matrix matmul(std::vector<Matrix> mats) {
  if (mats.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }
  Matrix result = std::move(mats.front());
  for (size_t i = 1; i < mats.size(); i++) {
    result = matmul(result, mats[i]);
  }
  return result;
}

int main(int argc, char *argv[]) {
  int size = 512;
  std::string order = "vector";
  if (argc > 3) {
    std::cerr
        << "usage: gebp [positive-matrix-size] [vector|ijk|ikj|kij|wrappers]\n";
    return 1;
  }
  if (argc > 1) {
    try {
      size = std::stoi(argv[1]);
    } catch (const std::exception &) {
      std::cerr << "usage: gebp [positive-matrix-size] "
                   "[vector|ijk|ikj|kij|wrappers]\n";
      return 1;
    }
  }
  if (argc > 2) {
    order = argv[2];
  }
  if (size <= 0) {
    std::cerr << "matrix size must be positive\n";
    return 1;
  }
  if (order != "vector" && order != "ijk" && order != "ikj" && order != "kij" &&
      order != "wrappers") {
    std::cerr << "mode must be vector, ijk, ikj, kij, or wrappers\n";
    return 1;
  }

  const auto element_count = static_cast<std::size_t>(size) * size;
  Matrix A{size, size, Major::row_major, std::vector<double>(element_count)};
  Matrix B{size, size, Major::row_major, std::vector<double>(element_count)};

  if (A.cols != B.rows) {
    std::cerr << "dimension error" << std::endl;
    return -1;
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A(i, j) = static_cast<double>((i + j) % 100) / 100.0;
      B(i, j) = static_cast<double>((i * 3 + j * 7) % 100) / 100.0;
    }
  }

  if (order == "wrappers") {
    const std::vector<Matrix> matrices{A, B};

    const auto list_start = std::chrono::steady_clock::now();
    const Matrix list_result = matmul(std::initializer_list<Matrix>{A, B});
    const auto list_end = std::chrono::steady_clock::now();

    const auto vector_start = std::chrono::steady_clock::now();
    const Matrix vector_result = matmul(matrices);
    const auto vector_end = std::chrono::steady_clock::now();

    double list_checksum = 0.0;
    for (double value : list_result.data) {
      list_checksum += value;
    }
    double vector_checksum = 0.0;
    for (double value : vector_result.data) {
      vector_checksum += value;
    }

    const std::chrono::duration<double, std::milli> list_elapsed =
        list_end - list_start;
    const std::chrono::duration<double, std::milli> vector_elapsed =
        vector_end - vector_start;
    std::cout << "Matrix size: " << size << " x " << size << '\n'
              << "Initializer-list wrapper: " << list_elapsed.count()
              << " ms, checksum " << list_checksum << '\n'
              << "Vector wrapper: " << vector_elapsed.count()
              << " ms, checksum " << vector_checksum << '\n';
    return 0;
  }

  if (order == "vector") {
    std::vector<Matrix> matrices;
    matrices.reserve(2);
    matrices.push_back(std::move(A));
    matrices.push_back(std::move(B));

    const auto start = std::chrono::steady_clock::now();
    const Matrix result = matmul(std::move(matrices));
    const auto end = std::chrono::steady_clock::now();

    double checksum = 0.0;
    for (double value : result.data) {
      checksum += value;
    }
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Matrix size: " << size << " x " << size << '\n'
              << "Matmul API: vector wrapper (i-k-j kernel)\n"
              << "GEMM time: " << elapsed.count() << " ms\n"
              << "Checksum: " << checksum << '\n';
    return 0;
  }

  Matrix C{A.rows, B.cols, Major::row_major,
           std::vector<double>(element_count, 0.0)};

  const auto multiply_ijk = [&] {
    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < B.cols; j++) {
        for (int k = 0; k < A.cols; k++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }
  };
  const auto multiply_ikj = [&] {
    for (int i = 0; i < A.rows; i++) {
      for (int k = 0; k < A.cols; k++) {
        for (int j = 0; j < B.cols; j++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }
  };
  const auto multiply_kij = [&] {
    for (int k = 0; k < A.cols; k++) {
      for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }
  };

  const auto start = std::chrono::steady_clock::now();
  if (order == "ijk") {
    multiply_ijk();
  } else if (order == "ikj") {
    multiply_ikj();
  } else {
    multiply_kij();
  }
  const auto end = std::chrono::steady_clock::now();

  double checksum = 0.0;
  for (double value : C.data) {
    checksum += value;
  }

  const std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Matrix size: " << size << " x " << size << '\n'
            << "Loop order: " << order << '\n'
            << "GEMM time: " << elapsed.count() << " ms\n"
            << "Checksum: " << checksum << '\n';

  return 0;
}
