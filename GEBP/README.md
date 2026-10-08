# GEBP

A C++20 project built with CMake.

## Build and benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gebp
```

The benchmark defaults to 512 × 512 matrices and the `k-i-j` loop ordering.
Pass a positive matrix size and an optional loop ordering (`ijk`, `ikj`, or
`kij`):

```sh
./build/gebp 512 ikj
./build/gebp 256 kij
```

Only the multiplication is timed; the result checksum is printed afterward to
make the computed result observable without the cost of printing the matrix.

## Sample performance

For square GEMM of size `N`, the approximate work is `2N³` floating-point
operations (one multiply and one add per inner-loop iteration). The following
table compares the three loop orderings using medians of three Release-build
runs on this machine. Timings and throughput vary with hardware and compiler.

Each loop-order cell shows `time in ms / GFLOP/s`.

| Matrix size | Approx. operations | `i-j-k` | `i-k-j` | `k-i-j` |
| ----------: | -----------------: | ------: | ------: | ------: |
|     64 × 64 |            524,288 | 0.1617 / 3.24 | 0.0249 / 21.04 | 0.0229 / 22.88 |
|   128 × 128 |          4,194,304 | 1.4552 / 2.88 | 0.1860 / 22.54 | 0.2140 / 19.60 |
|   256 × 256 |         33,554,432 | 11.1621 / 3.01 | 1.9903 / 16.86 | 2.7170 / 12.35 |
|   512 × 512 |        268,435,456 | 106.1290 / 2.53 | 17.1585 / 15.64 | 20.2953 / 13.23 |
| 1024 × 1024 |      2,147,483,648 | 961.3070 / 2.23 | 139.4430 / 15.40 | 172.4330 / 12.45 |

GFLOP/s is calculated as `operations / (time in seconds × 10⁹)`.

### Earlier measurements retained for reference

These earlier medians are kept for comparison with the original `i-j-k` and
`i-k-j` results:

| Matrix size | `i-j-k` (ms / GFLOP/s) | `i-k-j` (ms / GFLOP/s) |
| ----------: | ---------------------: | ---------------------: |
|     64 × 64 | 0.224 / 2.35 | 0.0553 / 9.48 |
|   128 × 128 | 1.490 / 2.82 | 0.3623 / 11.58 |
|   256 × 256 | 11.019 / 3.05 | 3.1340 / 10.71 |
|   512 × 512 | 102.833 / 2.61 | 23.0264 / 11.66 |
| 1024 × 1024 | 834.541 / 2.57 | 186.9110 / 11.49 |
