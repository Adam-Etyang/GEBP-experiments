#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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

void matmul_into(const Matrix &A, const Matrix &B, Matrix &C) {
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) {
    throw std::invalid_argument("matrix dimensions do not match for GEMM");
  }

  std::fill(C.data.begin(), C.data.end(), 0.0);
  for (int i = 0; i < A.rows; i++) {
    for (int k = 0; k < A.cols; k++) {
      const double a_ik = A(i, k);
      for (int j = 0; j < B.cols; j++) {
        C(i, j) += a_ik * B(k, j);
      }
    }
  }
}

void matmul_into_blocked(const Matrix &A, const Matrix &B, Matrix &C,
                         int block_size = 32) {
  if (block_size <= 0) {
    throw std::invalid_argument("block size must be positive");
  }
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) {
    throw std::invalid_argument("matrix dimensions do not match for GEMM");
  }

  std::fill(C.data.begin(), C.data.end(), 0.0);
  for (int ii = 0; ii < A.rows; ii += block_size) {
    const int i_end = std::min(ii + block_size, A.rows);

    for (int jj = 0; jj < B.cols; jj += block_size) {
      const int j_end = std::min(jj + block_size, B.cols);

      for (int kk = 0; kk < A.cols; kk += block_size) {
        const int k_end = std::min(kk + block_size, A.cols);

        for (int i = ii; i < i_end; i++) {
          for (int k = kk; k < k_end; k++) {
            const double a_ik = A(i, k);
            for (int j = jj; j < j_end; j++) {
              C(i, j) += a_ik * B(k, j);
            }
          }
        }
      }
    }
  }
}

Matrix matmul(const Matrix &A, const Matrix &B) {
  if (A.cols != B.rows) {
    throw std::invalid_argument("matrix dimensions do not match for GEMM");
  }

  Matrix result{
      A.rows, B.cols, Major::row_major,
      std::vector<double>(static_cast<std::size_t>(A.rows) * B.cols, 0.0)};

  matmul_into_blocked(A, B, result);
  return result;
}

Matrix matmul(std::vector<Matrix> mats) {
  if (mats.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }

  if (mats.size() == 2) {
    return matmul(mats[0], mats[1]);
  }

  int result_cols = mats.front().cols;
  std::size_t max_elements = 0;
  for (std::size_t i = 1; i < mats.size(); i++) {
    if (result_cols != mats[i].rows) {
      throw std::invalid_argument("matrix dimensions do not match for GEMM");
    }
    result_cols = mats[i].cols;
    max_elements =
        std::max(max_elements,
                 static_cast<std::size_t>(mats.front().rows) * result_cols);
  }

  Matrix result = std::move(mats.front());
  result.data.reserve(max_elements);
  Matrix scratch{result.rows, mats[1].cols, Major::row_major, {}};
  scratch.data.reserve(max_elements);

  for (std::size_t i = 1; i < mats.size(); i++) {
    scratch.rows = result.rows;
    scratch.cols = mats[i].cols;
    scratch.layout = Major::row_major;
    scratch.data.resize(static_cast<std::size_t>(scratch.rows) * scratch.cols);
    matmul_into_blocked(result, mats[i], scratch);
    std::swap(result, scratch);
  }
  return result;
}

Matrix matmul_allocating(std::vector<Matrix> mats) {
  if (mats.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }

  Matrix result = std::move(mats.front());
  for (std::size_t i = 1; i < mats.size(); i++) {
    result = matmul(result, mats[i]);
  }
  return result;
}

double checksum(const Matrix &matrix) {
  double sum = 0.0;
  for (double value : matrix.data) {
    sum += value;
  }
  return sum;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
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
  Matrix C{size, size, Major::row_major, std::vector<double>(element_count)};

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A(i, j) = static_cast<double>((i + j) % 100) / 100.0;
      B(i, j) = static_cast<double>((i * 3 + j * 7) % 100) / 100.0;
      C(i, j) = static_cast<double>((i * 5 + j * 2 + 1) % 100) / 100.0;
    }
  }

  std::vector<Matrix> matrices;
  matrices.reserve(3);
  matrices.push_back(std::move(A));
  matrices.push_back(std::move(B));
  matrices.push_back(std::move(C));

  const auto pair_result_elements =
      static_cast<std::size_t>(matrices[0].rows) * matrices[1].cols;
  Matrix unblocked_check{matrices[0].rows, matrices[1].cols,
                         Major::row_major,
                         std::vector<double>(pair_result_elements)};
  Matrix blocked_check{matrices[0].rows, matrices[1].cols, Major::row_major,
                       std::vector<double>(pair_result_elements)};
  matmul_into(matrices[0], matrices[1], unblocked_check);
  matmul_into_blocked(matrices[0], matrices[1], blocked_check);
  if (unblocked_check.data != blocked_check.data) {
    std::cerr << "blocked GEMM result differs from unblocked result\n";
    return 1;
  }

  const Matrix allocating_check = matmul_allocating(matrices);
  const Matrix reuse_check = matmul(matrices);
  if (allocating_check.data != reuse_check.data) {
    std::cerr << "buffer-reuse result differs from allocating result\n";
    return 1;
  }

  constexpr int repetitions = 5;
  std::vector<double> allocating_times;
  std::vector<double> reuse_times;
  allocating_times.reserve(repetitions);
  reuse_times.reserve(repetitions);

  auto measure_allocating = [&] {
    auto inputs = matrices;
    const auto start = std::chrono::steady_clock::now();
    const Matrix result = matmul_allocating(std::move(inputs));
    const auto end = std::chrono::steady_clock::now();
    return std::pair{
        std::chrono::duration<double, std::milli>(end - start).count(),
        checksum(result)};
  };
  auto measure_reuse = [&] {
    auto inputs = matrices;
    const auto start = std::chrono::steady_clock::now();
    const Matrix result = matmul(std::move(inputs));
    const auto end = std::chrono::steady_clock::now();
    return std::pair{
        std::chrono::duration<double, std::milli>(end - start).count(),
        checksum(result)};
  };

  double allocating_checksum = 0.0;
  double reuse_checksum = 0.0;
  for (int repetition = 0; repetition < repetitions; repetition++) {
    if (repetition % 2 == 0) {
      const auto [elapsed, sum] = measure_allocating();
      allocating_times.push_back(elapsed);
      allocating_checksum = sum;
      const auto [reuse_elapsed, reuse_sum] = measure_reuse();
      reuse_times.push_back(reuse_elapsed);
      reuse_checksum = reuse_sum;
    } else {
      const auto [elapsed, sum] = measure_reuse();
      reuse_times.push_back(elapsed);
      reuse_checksum = sum;
      const auto [allocating_elapsed, allocating_sum] = measure_allocating();
      allocating_times.push_back(allocating_elapsed);
      allocating_checksum = allocating_sum;
    }
  }

  const double allocating_ms = median(allocating_times);
  const double reuse_ms = median(reuse_times);
  const double operations = 4.0 * size * size * size;
  std::cout << std::fixed << std::setprecision(4)
            << "Three-matrix chain: " << size << " x " << size << '\n'
            << "Allocating median: " << allocating_ms << " ms ("
            << operations / (allocating_ms * 1.0e6) << " GFLOP/s)\n"
            << "Buffer-reuse median: " << reuse_ms << " ms ("
            << operations / (reuse_ms * 1.0e6) << " GFLOP/s)\n"
            << "Speedup: " << allocating_ms / reuse_ms << "x\n"
            << "Results match: "
            << (allocating_checksum == reuse_checksum ? "yes" : "no")
            << ", checksum " << reuse_checksum << '\n';

  return 0;
}
