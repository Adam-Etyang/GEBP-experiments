# GEBP

A C++20 matrix multiplication project built with CMake.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gebp
```

The program multiplies deterministic square matrices, 512 × 512 by default.
Pass a positive integer to use a different size:

```sh
./build/gebp 256
```

Matrix products use the vector-based matrix-chain interface and an `i-k-j`
GEMM kernel. The program prints the result dimensions and checksum.
