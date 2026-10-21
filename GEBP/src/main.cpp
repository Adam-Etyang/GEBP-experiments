#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "matrix.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

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
  constexpr int micro_rows = 4;
  constexpr int micro_cols = 2;
  for (int ii = 0; ii < A.rows; ii += block_size) {
    const int i_end = std::min(ii + block_size, A.rows);
    for (int jj = 0; jj < B.cols; jj += block_size) {
      const int j_end = std::min(jj + block_size, B.cols);

      for (int i0 = ii; i0 < i_end; i0 += micro_rows) {
        const int rows = std::min(micro_rows, i_end - i0);
        for (int j0 = jj; j0 < j_end; j0 += micro_cols) {
          const int cols = std::min(micro_cols, j_end - j0);

          if (rows == micro_rows && cols == micro_cols) {
            double c00 = 0.0, c01 = 0.0;
            double c10 = 0.0, c11 = 0.0;
            double c20 = 0.0, c21 = 0.0;
            double c30 = 0.0, c31 = 0.0;

            for (int kk = 0; kk < A.cols; kk += block_size) {
              const int k_end = std::min(kk + block_size, A.cols);
              for (int k = kk; k < k_end; k++) {
                const double b0 = B(k, j0);
                const double b1 = B(k, j0 + 1);
                const double a0 = A(i0, k);
                const double a1 = A(i0 + 1, k);
                const double a2 = A(i0 + 2, k);
                const double a3 = A(i0 + 3, k);
                c00 += a0 * b0;
                c01 += a0 * b1;
                c10 += a1 * b0;
                c11 += a1 * b1;
                c20 += a2 * b0;
                c21 += a2 * b1;
                c30 += a3 * b0;
                c31 += a3 * b1;
              }
            }

            C(i0, j0) = c00;
            C(i0, j0 + 1) = c01;
            C(i0 + 1, j0) = c10;
            C(i0 + 1, j0 + 1) = c11;
            C(i0 + 2, j0) = c20;
            C(i0 + 2, j0 + 1) = c21;
            C(i0 + 3, j0) = c30;
            C(i0 + 3, j0 + 1) = c31;
          } else {
            double accum[micro_rows][micro_cols] = {};
            for (int k = 0; k < A.cols; k++) {
              for (int r = 0; r < rows; r++) {
                const double a_ik = A(i0 + r, k);
                for (int c = 0; c < cols; c++) {
                  accum[r][c] += a_ik * B(k, j0 + c);
                }
              }
            }
            for (int r = 0; r < rows; r++) {
              for (int c = 0; c < cols; c++) {
                C(i0 + r, j0 + c) = accum[r][c];
              }
            }
          }
        }
      }
    }
  }
}

void matmul_into_neon(const Matrix &A, const Matrix &B, Matrix &C,
                      int block_size = 32) {
  if (block_size <= 0) {
    throw std::invalid_argument("block size must be positive");
  }
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) {
    throw std::invalid_argument("matrix dimensions do not match for GEMM");
  }

#if defined(__aarch64__)
  if (A.layout != Major::row_major || B.layout != Major::row_major ||
      C.layout != Major::row_major) {
    matmul_into_blocked(A, B, C, block_size);
    return;
  }

  std::fill(C.data.begin(), C.data.end(), 0.0);
  constexpr int micro_rows = 4;
  constexpr int micro_cols = 2;
  for (int ii = 0; ii < A.rows; ii += block_size) {
    const int i_end = std::min(ii + block_size, A.rows);
    for (int jj = 0; jj < B.cols; jj += block_size) {
      const int j_end = std::min(jj + block_size, B.cols);
      for (int i0 = ii; i0 < i_end; i0 += micro_rows) {
        const int rows = std::min(micro_rows, i_end - i0);
        for (int j0 = jj; j0 < j_end; j0 += micro_cols) {
          const int cols = std::min(micro_cols, j_end - j0);

          if (rows == micro_rows && cols == micro_cols) {
            float64x2_t c0 = vdupq_n_f64(0.0);
            float64x2_t c1 = vdupq_n_f64(0.0);
            float64x2_t c2 = vdupq_n_f64(0.0);
            float64x2_t c3 = vdupq_n_f64(0.0);

            for (int kk = 0; kk < A.cols; kk += block_size) {
              const int k_end = std::min(kk + block_size, A.cols);
              for (int k = kk; k < k_end; k++) {
                const double *b_row =
                    B.data.data() + static_cast<std::size_t>(k) * B.cols + j0;
                const float64x2_t b_vec = vld1q_f64(b_row);
                c0 = vfmaq_f64(c0, vdupq_n_f64(A(i0, k)), b_vec);
                c1 = vfmaq_f64(c1, vdupq_n_f64(A(i0 + 1, k)), b_vec);
                c2 = vfmaq_f64(c2, vdupq_n_f64(A(i0 + 2, k)), b_vec);
                c3 = vfmaq_f64(c3, vdupq_n_f64(A(i0 + 3, k)), b_vec);
              }
            }

            for (int r = 0; r < micro_rows; r++) {
              double *c_row =
                  C.data.data() + static_cast<std::size_t>(i0 + r) * C.cols + j0;
              const float64x2_t c_vec = r == 0 ? c0 : r == 1 ? c1 : r == 2 ? c2 : c3;
              vst1q_f64(c_row, c_vec);
            }
          } else {
            double accum[micro_rows][micro_cols] = {};
            for (int k = 0; k < A.cols; k++) {
              for (int r = 0; r < rows; r++) {
                const double a_ik = A(i0 + r, k);
                for (int c = 0; c < cols; c++) {
                  accum[r][c] += a_ik * B(k, j0 + c);
                }
              }
            }
            for (int r = 0; r < rows; r++) {
              double *c_row =
                  C.data.data() + static_cast<std::size_t>(i0 + r) * C.cols + j0;
              for (int c = 0; c < cols; c++) {
                c_row[c] = accum[r][c];
              }
            }
          }
        }
      }
    }
  }
#else
  matmul_into_blocked(A, B, C, block_size);
#endif
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

template <typename Kernel>
Matrix matmul_reusing(std::vector<Matrix> mats, Kernel kernel) {
  if (mats.size() < 2) {
    throw std::invalid_argument("Need at least 2 matrices");
  }

  if (mats.size() == 2) {
    Matrix result{mats[0].rows, mats[1].cols, Major::row_major,
                  std::vector<double>(
                      static_cast<std::size_t>(mats[0].rows) * mats[1].cols)};
    kernel(mats[0], mats[1], result);
    return result;
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
    kernel(result, mats[i], scratch);
    std::swap(result, scratch);
  }
  return result;
}

Matrix matmul(std::vector<Matrix> mats) {
  return matmul_reusing(
      std::move(mats), [](const Matrix &A, const Matrix &B, Matrix &C) {
        matmul_into_blocked(A, B, C);
      });
}

Matrix matmul_neon(std::vector<Matrix> mats) {
  return matmul_reusing(
      std::move(mats), [](const Matrix &A, const Matrix &B, Matrix &C) {
        matmul_into_neon(A, B, C);
      });
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
  const Matrix neon_check = matmul_neon(matrices);
  double max_relative_error = 0.0;
  for (std::size_t i = 0; i < reuse_check.data.size(); i++) {
    const double scale = std::max(1.0, std::abs(reuse_check.data[i]));
    max_relative_error =
        std::max(max_relative_error,
                 std::abs(neon_check.data[i] - reuse_check.data[i]) / scale);
  }
  if (max_relative_error > 1.0e-10) {
    std::cerr << "NEON result differs from blocked result\n";
    return 1;
  }

  constexpr int repetitions = 5;
  std::vector<double> allocating_times;
  std::vector<double> reuse_times;
  std::vector<double> neon_times;
  allocating_times.reserve(repetitions);
  reuse_times.reserve(repetitions);
  neon_times.reserve(repetitions);

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
  auto measure_neon = [&] {
    auto inputs = matrices;
    const auto start = std::chrono::steady_clock::now();
    const Matrix result = matmul_neon(std::move(inputs));
    const auto end = std::chrono::steady_clock::now();
    return std::pair{
        std::chrono::duration<double, std::milli>(end - start).count(),
        checksum(result)};
  };

  double allocating_checksum = 0.0;
  double reuse_checksum = 0.0;
  double neon_checksum = 0.0;
  auto record_allocating = [&] {
    const auto [elapsed, sum] = measure_allocating();
    allocating_times.push_back(elapsed);
    allocating_checksum = sum;
  };
  auto record_reuse = [&] {
    const auto [elapsed, sum] = measure_reuse();
    reuse_times.push_back(elapsed);
    reuse_checksum = sum;
  };
  auto record_neon = [&] {
    const auto [elapsed, sum] = measure_neon();
    neon_times.push_back(elapsed);
    neon_checksum = sum;
  };
  for (int repetition = 0; repetition < repetitions; repetition++) {
    switch (repetition % 3) {
    case 0:
      record_allocating();
      record_reuse();
      record_neon();
      break;
    case 1:
      record_reuse();
      record_neon();
      record_allocating();
      break;
    default:
      record_neon();
      record_allocating();
      record_reuse();
      break;
    }
  }

  const double allocating_ms = median(allocating_times);
  const double reuse_ms = median(reuse_times);
  const double neon_ms = median(neon_times);
  const double operations = 4.0 * size * size * size;
  std::cout << std::fixed << std::setprecision(4)
            << "Three-matrix chain: " << size << " x " << size << '\n'
            << "Allocating median: " << allocating_ms << " ms ("
            << operations / (allocating_ms * 1.0e6) << " GFLOP/s)\n"
            << "Register-blocked C++ median: " << reuse_ms << " ms ("
            << operations / (reuse_ms * 1.0e6) << " GFLOP/s)\n"
            << "Register-blocked NEON median: " << neon_ms << " ms ("
            << operations / (neon_ms * 1.0e6) << " GFLOP/s)\n"
            << "NEON speedup over C++: " << reuse_ms / neon_ms << "x\n"
            << "Results match: "
            << (allocating_checksum == reuse_checksum &&
                        max_relative_error <= 1.0e-10
                    ? "yes"
                    : "no")
            << ", max relative error " << std::scientific << std::setprecision(3)
            << max_relative_error << ", checksum " << std::fixed
            << std::setprecision(4) << neon_checksum << '\n';

  return 0;
}
