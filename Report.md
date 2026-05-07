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
| -O3 -march=native | 7.49 | 7.49x |
| -O3 -march=native -ffast-math | 6.69  | ~39.35x |
| -O3 -march=native -ffast-math -funroll-loops | 6.59 | ~38.76x |
| -O3 -march=native -ffast-math -fopenmp-simd | 6.50 | ~38.2x |

**Did you add any `#pragma` hints to the source?** If yes, which ones?

Yes

#pragma GCC ivdep

for(int j=0;j<N;j++)

    c_row[j]+=a*b_row[j];

**What speedup did you achieve? Why?**

The compiler optimization flags significantly improved performance by enabling
automatic vectorization and CPU-specific optimizations.

The -march=native flag allows the compiler to use SIMD instructions supported
by the processor (such as AVX2). This allows multiple floating-point operations
to be executed simultaneously.

The -ffast-math flag relaxes strict IEEE floating point rules, enabling the
compiler to reorder operations and generate more efficient vectorized code.

Loop unrolling (-funroll-loops) reduces loop control overhead and increases
instruction-level parallelism.

Finally, the use of SIMD pragmas or -fopenmp-simd allows the compiler to safely
vectorize inner loops, further improving throughput.

Overall, these optimizations increased performance from about 0.84 GFLOP/s to
approximately 11 GFLOP/s on N=1024, achieving around a 13× speedup compared to
the naive baseline implementation.
---

## Task 4 – Loop Tiling

> Experiment with tile sizes to find the sweet spot for your cache hierarchy.

| Tile size | N=1024 (GFLOP/s) | N=4096 (GFLOP/s) |
|---|---|---|
| 32 | ~6.9 | ~5.7 |
| 64 | ~7.4 | ~6.3 |
| 128 | ~6.8 | ~5.9 |
| 256 | ~5.1 | ~4.4 |

**Best tile size:** ___ 64

**Why does this tile size work best for your machine?**
The best tile size on this machine was 64.

Loop tiling improves performance by dividing the matrix multiplication
into smaller blocks that fit better into the CPU cache. When the tile
size matches the capacity of the cache, elements of matrices A, B,
and C can remain in the cache during computation.

With a tile size of 64, the working set of data fits well in the L1/L2
cache of the processor. This reduces cache misses and improves spatial
and temporal locality.

Smaller tiles such as 32 do not fully utilize the cache capacity,
while larger tiles such as 128 or 256 exceed the cache size and cause
more cache evictions. Therefore, a tile size of 64 achieves the best
balance between computation and memory access on this machine.

---

## Task 5 – Multithreading

> Measure scaling as you increase the number of OpenMP threads.

| Threads | N=4096 (GFLOP/s) | Speedup |
|---|---|---|
| 1 |7.5 | 1.0× |
| 2 | 13.9 | 1.85x |
| 4 | 24.8 | 3.30x |
| 8 | 26.1 | 3.48x |
| _(max physical cores)_ | 24.8 | 3.30x |

**Does throughput scale linearly with threads?** Why / why not?
Throughput does not scale perfectly linearly with the number of threads.

While increasing the number of threads improves performance, the speedup
gradually decreases as more threads are added. This happens because threads
compete for shared resources such as memory bandwidth and cache.

Additionally, the processor has 4 physical cores but 8 logical threads
(hyper-threading). Hyper-threading allows two threads to share the same
core resources, so the performance improvement beyond 4 threads is limited.

Therefore, the best performance is usually achieved near the number of
physical cores rather than the number of logical threads.
---

## Task 6 – Performance Analysis

**Is your implementation compute-bound or memory-bound?** Justify with arithmetic intensity (FLOPs / bytes).

**Comparison vs. PyTorch (N=4096):**

| Implementation | GFLOP/s | % of PyTorch |
|---|---|---|
| Naive C | 0.56 | 0.5% |
| Best optimised C | 20.35 | 19% |
| PyTorch (CPU) | 105 | 100% |

**What is the gap and why does it exist?**
The implementation is mostly memory-bound for the naive version because
matrix multiplication repeatedly loads data from memory with poor cache
reuse. Optimized versions such as loop-reordered and tiled implementations
increase arithmetic intensity by improving cache locality.

However, even with these optimizations the algorithm is still partially
memory-bound because large matrices exceed cache capacity and require
frequent memory accesses.
---

## Task 7 – Key Takeaways

_Write 3–5 sentences summarising the most important lessons learned from this lab._his lab demonstrated how hardware-aware optimizations significantly improve matrix multiplication performance. Loop reordering improves cache locality, vectorization enables SIMD execution, and loop tiling improves cache reuse. Multithreading allows multiple CPU cores to compute simultaneously. Despite these optimizations, highly optimized libraries like PyTorch achieve higher performance due to assembly-level optimizations and advanced scheduling.

---

