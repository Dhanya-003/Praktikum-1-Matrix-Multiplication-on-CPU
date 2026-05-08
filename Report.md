# Lab Report – Matrix Multiplication on CPU

**Course:** AI Accelerators (AIA)

**Lab:** Praktikum 1

**Team members:** Dhanyashri Mohanarangan, Dakshil Rajeshbhai Vora

**Date:** 08/05/2026

---

## Task 1 – System Characterisation

> Fill in the details of your machine. Use tools such as `lscpu`, `lstopo`, `/proc/cpuinfo`.

| Property | Value |
|---|---|
| CPU model | AMD Ryzen 7 7445HS w/ Radeon 740M Graphics |
| Number of cores / threads | 6 cores / 12 threads |
| Base / Boost clock speed (GHz) | ~4.0 GHz base / up to ~4.7 GHz boost |
| SIMD ISA | SSE, SSE2, SSE4.1, SSE4.2, AVX, AVX2, AVX-512, FMA |
| SIMD width (bits / floats per vector) | 256 bits (8 FP32 floats/vector)|
| MAC units per core | 2 FMA units/core |
| L1 cache size (per core) | 32 KB L1d + 32 KB L1i |
| L2 cache size (per core) | 1 MB |
| L3 cache size (shared) | 16 MB |
| Peak theoretical throughput (GFLOP/s) | ~902 GFLOP/s (FP32 AVX2 FMA peak) |

**How did you calculate peak throughput?**

For FP32 matrix multiplication:

AVX2 vector width = 256 bits

FP32 float = 32 bits

So vector lanes:

256/32=8 floats per vector

FMA performs multiply + add together:

1 FMA=2 FLOPs per element

The CPU has:

6 cores

~4.7 GHz boost

2 FMA units/core

8 FP32 elements/vector

Therefore:

6×4.7×8×2×2 ≈ 902 GFLOP/s

Explanation:

6 → cores

4.7 → GHz

8 → floats/vector

first 2 → multiply + add

second 2 → two FMA pipelines/core

Gist:

The peak FP32 throughput was estimated using the number of cores, clock frequency, SIMD vector width, and FMA capability. AVX2 processes 8 FP32 values per vector instruction, and FMA performs a multiplication and addition simultaneously, corresponding to 2 FLOPs per element. The Ryzen 7 7445HS provides two FMA units per core, giving an estimated theoretical peak of approximately 902 GFLOP/s.

---

## Task 2 – Loop Reordering

> Measure each loop ordering for matrix sizes 64, 128, 256, 512, 1024, 2048, 4096.

| Sizes | i-j-k (naive) | i-k-j | j-k-i | k-i-j | Tiled | Parallel |
|---|---|---|---|---|---|---|
| N=64 | 7.18 |  9.04 |  2.11 |  14.98  | 9.12  | 9.12 |
| N=128 | 1.50 |  13.98 |   0.72  | 14.22 |  8.65 |   9.08 |
| N=256 | 1.37 | 17.49 | 0.39 |  16.93 |  8.66  | 8.26 |
| N=512 | 0.39 | 15.81 | 0.13 |  15.78 |  7.67 |  7.67 |
| N=1024 | 0.17 | 15.54 |  0.05 | 15.97 | 7.86 | 7.81 |
| N=2048 | 

**Best ordering found:**  i-k-j

**Why does this ordering perform best?**

_(Explain in terms of spatial locality and cache reuse of A, B, and C)_

The i-k-j ordering performed best because it accesses matrix B and matrix C sequentially in memory, improving spatial locality and cache efficiency. In row-major storage, iterating over j in the innermost loop allows contiguous memory access for B[k][j] and C[i][j], which reduces cache misses and improves hardware prefetching. Additionally, the value A[i][k] can be reused across the entire inner loop, increasing temporal locality.

## Task 3 – Vectorization

> List the compiler flags you tested and their effect.

| Flags added | N=1024 (GFLOP/s) (Parallel) | Speedup vs. naive |
|---|---|---|
| -O3 only (baseline) | 8.64 | 48x |
| -O3 -march=native | 6.98 | 36.73x |
| -O3 -march=native -ffast-math | 6.42  | ~35.66x |
| -O3 -march=native -ffast-math -funroll-loops | 6.97 | ~38.72x |
| -O3 -march=native -ffast-math -fopenmp-simd | 6.50 | ~38.2x |

**Did you add any `#pragma` hints to the source?** If yes, which ones?

Yes

#pragma GCC ivdep

for(int j=0;j<N;j++)

    c_row[j]+=a*b_row[j];

**What speedup did you achieve? Why?**

Increasing compiler optimization beyond -O3 does not improve performance for matrix multiplication. In fact, flags such as -march=native, -ffast-math, -funroll-loops, and -fopenmp-simd slightly reduce performance, indicating that the compiler already generates efficient vectorized code at -O3. The observed behavior suggests that the kernel is not compute-bound anymore, but limited by memory hierarchy effects and instruction scheduling overhead.

---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=1024 (GFLOP/s) |
|---|---|
| 32 | ~4 |
| 64 | 6.6 | 
| 128 | 9.29 |
| 256 | 8.63 |

**Best tile size:** ___ 128

**Why does this tile size work best for your machine?**

A tile size of 128 provides the best balance between cache reuse and loop overhead on the tested machine. With loop tiling, matrix sub-blocks are reused multiple times while remaining in the CPU cache, reducing expensive memory accesses to RAM.

For smaller tile sizes such as 32 and 64:

cache usage is efficient,
but the program performs more loop iterations and block management,
increasing loop-control overhead.

For larger tile sizes such as 256:

the working data block becomes too large for efficient cache usage,
causing more cache misses and memory traffic,
which reduces performance.

The 128×128 tile size fits more effectively within the machine’s cache hierarchy (mainly L2 cache), allowing better temporal locality and improved reuse of matrix data. This minimizes memory stalls and results in the highest observed performance of about 9.29 GFLOP/s.

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=1024 (GFLOP/s) | Speedup |
|---|---|---|
| 1 |  11.44  |   1.00x |
| 2 | 22.97   |   2.01x |
| 4 | 45.45   |   3.97x |
| 6 | 73.02   |   6.38x |

**Does throughput scale linearly with threads?** Why / why not?

Throughput scales approximately linearly with the number of threads because matrix multiplication can be divided into independent computations across multiple CPU cores. However, scaling is not perfectly linear due to memory bandwidth limits, cache contention, and thread management overhead.

---

## Task 6 – Performance Analysis

**Comparison vs. PyTorch (N=1024):**

| Implementation   | GFLOP/s | % of PyTorch |
| ---------------- | ------- | ------------ |
| Naive C (approx) | ~0.5–3  | ~1–3%        |
| Best optimized C | 65.10   | ~73%         |
| PyTorch CPU      | 89.29   | 100%         |

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).

## Arithmetic Intensity

Arithmetic Intensity (AI) = FLOPs / Bytes moved

For matrix multiplication (N × N):

FLOPs = 2N³  
Bytes = 3N² × 4 = 12N²  

AI = (2N³) / (12N²) = N / 6  

For N = 1024:  
AI ≈ 170.67  

Matrix multiplication has high arithmetic intensity, meaning it is theoretically compute-bound. However, the naive implementation is memory-bound due to poor cache reuse. Loop reordering, tiling, and multithreading significantly improve cache utilization, increasing arithmetic intensity and shifting performance closer to a compute-bound regime. Despite these optimizations, the implementation still does not reach PyTorch performance due to more advanced low-level optimizations in vendor-tuned libraries.

**What is the gap and why does it exist?**

The performance gap between the optimized C implementation and PyTorch arises from highly optimized BLAS libraries used by PyTorch, which implement architecture-specific microkernels, vectorized SIMD instructions, and advanced cache blocking strategies. While the custom C implementation achieves strong performance through tiling and multithreading, it lacks low-level hardware-specific optimizations, resulting in lower overall throughput.

---

## Task 7 – Key Takeaways

Write 3–5 sentences summarising the most important lessons learned from this lab.

This lab demonstrates that matrix multiplication performance is strongly influenced by memory hierarchy, cache behavior, and data access patterns rather than just the number of floating-point operations. Loop reordering and tiling significantly improve performance by increasing cache reuse and reducing memory latency. Multithreading provides substantial speedup, but scalability is limited by memory bandwidth and overhead rather than purely by available CPU cores. Compiler optimizations and vectorization help, but their impact is secondary once the code is already memory-optimized. Overall, achieving high performance requires combining algorithmic structure with hardware-aware optimizations.


---

