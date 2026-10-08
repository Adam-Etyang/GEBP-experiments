# GEBP

A C++20 project built with CMake.

## Build and benchmark

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/gebp
```

The benchmark multiplies 1024 × 1024 matrices by default. Pass a positive
integer to choose another square matrix size:

```sh
./build/gebp 512
```

Only the multiplication is timed; the result checksum is printed afterward to
make the computed result observable without the cost of printing the matrix.

```text
Matrix size: 1024 x 1024
GEMM time: 850.177 ms
Checksum: 2.6302e+08

Matrix size: 512 x 512
GEMM time: 132.017 ms
Checksum: 3.28732e+07

Matrix size: 64 x 64
GEMM time: 0.323333 ms
Checksum: 69905.4
```
