# GEBP

A C++20 matrix multiplication project built with CMake.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gebp
```

The program compares an allocating matrix-chain implementation with the
buffer-reuse implementation. Both multiply the same chain of three
deterministic square matrices, 512 × 512 by default. Pass a positive integer to
choose another size:

```sh
./build/gebp 256
```

Both implementations use the same blocked `i-k-j` GEMM kernel with a 32-element
block size. The allocating version creates a new result for each pairwise
product; the buffer-reuse version alternates two output buffers through
`matmul_into_blocked`. Inputs are copied before timing, and the program checks
the blocked kernel against the unblocked kernel as well as comparing the two
chain results. Timings are medians of five Release-build runs. A three-matrix
chain performs approximately `4N³` floating-point operations.

Each pair below is `allocating / buffer reuse` time in milliseconds. The
blocked-loop columns name the order of the outer tile loops:

| Matrix size | Approx. operations | Unblocked `i-k-j` | Blocked `ii-kk-jj` | Blocked `ii-jj-kk` |
| ----------: | -----------------: | ---------------: | ----------------: | ----------------: |
|     64 × 64 |          1,048,576 | 0.1665 / 0.1283 | 0.1923 / 0.1901 | 0.1220 / 0.1235 |
|   128 × 128 |          8,388,608 | 0.7148 / 0.7213 | 0.9865 / 1.0485 | 0.8194 / 0.8106 |
|   256 × 256 |         67,108,864 | 4.7460 / 4.5891 | 5.1473 / 5.0332 | 4.8716 / 5.0189 |
|   512 × 512 |        536,870,912 | 34.6567 / 34.7944 | 49.9877 / 49.8824 | 50.3337 / 50.5103 |
| 1024 × 1024 |      4,294,967,296 | 283.8997 / 283.6955 | 442.7710 / 438.9061 | 437.8094 / 437.9946 |

Compared with `ii-kk-jj`, the new `ii-jj-kk` order improves allocating medians
by about 37% at 64 × 64 and 17% at 128 × 128; buffer-reuse medians improve by
about 35% and 23%. For 512 × 512 and 1024 × 1024 the two blocked orders are
within about 1% of each other. The unblocked results remain faster at larger
sizes in these runs. Results match for all measured sizes, including a 37 × 37
check for partial edge tiles. These are separate five-run medians, so small
differences can vary with system load.
