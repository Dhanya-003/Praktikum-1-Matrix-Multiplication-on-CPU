/*
* Prakitikum 1 – Matrix Multiplication on CPU
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Your task is to progressively optimize this naive C implementation
 * of matrix multiplication (C = A * B) through the steps below.
 * Read README.md carefully before you start!
 *
 * Build:  make
 * Run:    ./matmul <size>      (e.g. ./matmul 512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

/*  #define THREAD_COUNT 4*/
int THREAD_COUNT = 8;
/* If we use int THREAD_COUNT = 8; with few changes then it will use all threads but currently it is using 1 core insted of 4*/
const int num_iterations = 2;
#define JB 64 //Tile size divides matrix size

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static void matmul_tiled_range(const float* A, const float* B, float* C,
                               int row_start, int row_end, int M, int N, int K) {
    for (int ii = row_start; ii < row_end; ii += JB) {
        const int i_end = min_int(ii + JB, row_end);
        for (int kk = 0; kk < K; kk += JB) {
            const int k_end = min_int(kk + JB, K);
            for (int jj = 0; jj < N; jj += JB) {
                const int j_end = min_int(jj + JB, N);
                for (int i = ii; i < i_end; i++) {
                    float* c_row = &C[i * N];
                    for (int k = kk; k < k_end; k++) {
                        const float a_ik = A[i * K + k];
                        const float* b_row = &B[k * N];
                        for (int j = jj; j < j_end; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

#ifdef _WIN32
typedef struct {
    const float* A;
    const float* B;
    float* C;
    int row_start;
    int row_end;
    int M;
    int N;
    int K;
} matmul_thread_args;

DWORD WINAPI matmul_thread_worker(LPVOID arg) {
    matmul_thread_args* args = (matmul_thread_args*)arg;
    matmul_tiled_range(args->A, args->B, args->C,
                       args->row_start, args->row_end,
                       args->M, args->N, args->K);
    return 0;
}
#endif

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            const float a_ik = A[i * K + k];
            const float* b_row = &B[k * N];
            float* c_row = &C[i * N];
            for (int j = 0; j < N; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
/* For fast execution we put c_row outside of K and add #pragma GCC ivdep 

void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++)
    {
        float* c_row = &C[i * N];

        for (int k = 0; k < K; k++)
        {
            float a_ik = A[i * K + k];
            const float* b_row = &B[k * N];

            #pragma GCC ivdep
            for (int j = 0; j < N; j++)
            {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}*/

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================

void matmul_looptiling(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    matmul_tiled_range(A, B, C, 0, M, M, N, K);
}

// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================
void matmul_parallel_ikj(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    #ifdef _WIN32
    HANDLE threads[THREAD_COUNT];
    matmul_thread_args args[THREAD_COUNT];
    const int chunk = (M + THREAD_COUNT - 1) / THREAD_COUNT;
    int launched = 0;

    for (int t = 0; t < THREAD_COUNT; t++) {
        const int row_start = t * chunk;
        const int row_end = min_int(row_start + chunk, M);
        if (row_start >= row_end) {
            break;
        }

        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].M = M;
        args[t].N = N;
        args[t].K = K;

        threads[t] = CreateThread(NULL, 0, matmul_thread_worker, &args[t], 0, NULL);
        if (threads[t] == NULL) {
            for (int j = 0; j < t; j++) {
                WaitForSingleObject(threads[j], INFINITE);
                CloseHandle(threads[j]);
            }
            matmul_tiled_range(A, B, C, row_start, M, M, N, K);
            return;
        }
        launched++;
    }
    // ============================================================================
// IMPLEMENTATION 5: j-k-i
// ============================================================================
void matmul_jki(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

    for(int j=0;j<N;j++)
    {
        for(int k=0;k<K;k++)
        {
            float b=B[k*N+j];

            for(int i=0;i<M;i++)
            {
                C[i*N+j]+=A[i*K+k]*b;
            }
        }
    }
}


// ============================================================================
// IMPLEMENTATION 6: k-i-j
// ============================================================================
void matmul_kij(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

    for(int k=0;k<K;k++)
    {
        for(int i=0;i<M;i++)
        {
            float a=A[i*K+k];

            for(int j=0;j<N;j++)
            {
                C[i*N+j]+=a*B[k*N+j];
            }
        }
    }
}

    WaitForMultipleObjects(launched, threads, TRUE, INFINITE);
    for (int t = 0; t < launched; t++) {
        CloseHandle(threads[t]);
    }
    #else
    matmul_tiled_range(A, B, C, 0, M, M, N, K);
    #endif
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols){
    for (int i = 0; i < rows * cols; i++){
        matrix[i] = rand() % 100;
    }
}

double get_time_ms() {
    #ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
    #else
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    #endif
}

double calculate_gflops(int M, int N, int K, double total_time) {
    if (total_time <= 0.0) {
        return 0.0;
    }
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float* C_ref, const float* C_test, int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float* A, const float* B, float* C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float* A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); //Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m" (C[0]) : : "memory");
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time/num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmakrk
// ============================================================================
int main(int argc, char *argv[]) {
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {1024, 512, 256, 128, 64};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    printf("%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n",
       "Size","i-j-k","i-k-j","j-k-i","k-i-j","Tiled","Parallel");
/*     printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");
This i change bc to run till 4096 */ 
    for (int i = 0; i < n; i++) {
        int M = sizes[i], N = M, K = M;

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));
        float *C_ref = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        matmul_naive(A, B, C_ref, M, N, K);

        // --- 1. Naive ---
        memset(C, 0, M * N * sizeof(float));

        float t_naive = benchmark(matmul_naive, A, B, C, M, N, K);
        double g_naive = calculate_gflops(M, N, K, t_naive);

        // --- 2. Tiled ---
        memset(C, 0, M * N * sizeof(float));
        matmul_looptiling(A, B, C, M, N, K);
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Tiled verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }

        float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
        double g_blocking = calculate_gflops(M, N, K, t_blocking);

        // --- 3. Reordered ---
        memset(C, 0, M * N * sizeof(float));
        matmul_looporder(A, B, C, M, N, K);
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Reordered verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }

        float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
        double g_reorder = calculate_gflops(M, N, K, t_reorder);

        // --- 4. Parallel ---
        memset(C, 0, M * N * sizeof(float));
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Parallel verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }
        memset(C, 0, M * N * sizeof(float));

        float t_jki = benchmark(matmul_jki, A, B, C, M, N, K);
        double g_jki = calculate_gflops(M, N, K, t_jki);
        memset(C, 0, M * N * sizeof(float));
        float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);

        memset(C, 0, M * N * sizeof(float));

        float t_kij = benchmark(matmul_kij, A, B, C, M, N, K);
        double g_kij = calculate_gflops(M, N, K, t_kij);

        printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
                M, g_naive, g_reorder, g_jki, g_kij, g_blocking, g_parallel);

        free(A); free(B); free(C); free(C_ref);
    }

    return 0;
}

/*/*
* Prakitikum 1 – Matrix Multiplication on CPU
 * ============================================
 * AI Accelerators (AIA) – Lab Assignment
 *
 * Your task is to progressively optimize this naive C implementation
 * of matrix multiplication (C = A * B) through the steps below.
 * Read README.md carefully before you start!
 *
 * Build:  make
 * Run:    ./matmul <size>      (e.g. ./matmul 512)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define THREAD_COUNT 4
const int num_iterations = 4;
#define JB 64 //Tile size divides matrix size

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static void matmul_tiled_range(const float* A, const float* B, float* C,
                               int row_start, int row_end, int M, int N, int K) {
    for (int ii = row_start; ii < row_end; ii += JB) {
        const int i_end = min_int(ii + JB, row_end);
        for (int kk = 0; kk < K; kk += JB) {
            const int k_end = min_int(kk + JB, K);
            for (int jj = 0; jj < N; jj += JB) {
                const int j_end = min_int(jj + JB, N);
                for (int i = ii; i < i_end; i++) {
                    float* c_row = &C[i * N];
                    for (int k = kk; k < k_end; k++) {
                        const float a_ik = A[i * K + k];
                        const float* b_row = &B[k * N];
                        for (int j = jj; j < j_end; j++) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

#ifdef _WIN32
typedef struct {
    const float* A;
    const float* B;
    float* C;
    int row_start;
    int row_end;
    int M;
    int N;
    int K;
} matmul_thread_args;

DWORD WINAPI matmul_thread_worker(LPVOID arg) {
    matmul_thread_args* args = (matmul_thread_args*)arg;
    matmul_tiled_range(args->A, args->B, args->C,
                       args->row_start, args->row_end,
                       args->M, args->N, args->K);
    return 0;
}
#endif

// ============================================================================
// IMPLEMENTATION 1: NAIVE MATRIX MULTIPLICATION
// ============================================================================
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// IMPLEMENTATION 2: -03 -ffmastmath, does loop unrolling and vectorization, reorder k,j
// ============================================================================
void matmul_looporder(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            const float a_ik = A[i * K + k];
            const float* b_row = &B[k * N];
            float* c_row = &C[i * N];
            for (int j = 0; j < N; j++) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// ============================================================================
// IMPLEMENTATION 3: Tiling
// ============================================================================

void matmul_looptiling(const float* A, const float* B, float* C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    matmul_tiled_range(A, B, C, 0, M, M, N, K);
}

// ============================================================================
// IMPLEMENTATION 4: Multithreading
// ============================================================================
void matmul_parallel_ikj(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    #ifdef _WIN32
    HANDLE threads[THREAD_COUNT];
    matmul_thread_args args[THREAD_COUNT];
    const int chunk = (M + THREAD_COUNT - 1) / THREAD_COUNT;
    int launched = 0;

    for (int t = 0; t < THREAD_COUNT; t++) {
        const int row_start = t * chunk;
        const int row_end = min_int(row_start + chunk, M);
        if (row_start >= row_end) {
            break;
        }

        args[t].A = A;
        args[t].B = B;
        args[t].C = C;
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].M = M;
        args[t].N = N;
        args[t].K = K;

        threads[t] = CreateThread(NULL, 0, matmul_thread_worker, &args[t], 0, NULL);
        if (threads[t] == NULL) {
            for (int j = 0; j < t; j++) {
                WaitForSingleObject(threads[j], INFINITE);
                CloseHandle(threads[j]);
            }
            matmul_tiled_range(A, B, C, row_start, M, M, N, K);
            return;
        }
        launched++;
    }

    WaitForMultipleObjects(launched, threads, TRUE, INFINITE);
    for (int t = 0; t < launched; t++) {
        CloseHandle(threads[t]);
    }
    #else
    matmul_tiled_range(A, B, C, 0, M, M, N, K);
    #endif
}

// ============================================================================
// Utility functions: Init Matrix, Benchmarking, Calculate Gflops
// ============================================================================
void initialize_matrix(float *matrix, int rows, int cols){
    for (int i = 0; i < rows * cols; i++){
        matrix[i] = rand() % 100;
    }
}

double get_time_ms() {
    #ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
    #else
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
    #endif
}

double calculate_gflops(int M, int N, int K, double total_time) {
    if (total_time <= 0.0) {
        return 0.0;
    }
    double flops = 2.0 * M * N * K;
    double gflops = (flops / ((total_time) / 1000.0)) / 1e9;
    return gflops;
}

int verify_result(const float* C_ref, const float* C_test, int M, int N, float tolerance) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_ref[i] - C_test[i]) > tolerance) {
            printf("Mismatch at index %d: ref=%f, test=%f\n", i, C_ref[i], C_test[i]);
            return 0;
        }
    }
    return 1;
}

typedef void (*matmul_fn)(const float* A, const float* B, float* C, int M, int N, int K);

float benchmark(matmul_fn matmul, const float* A, const float *B, float *C, int M, int N, int K)
{
    matmul(A, B, C, M, N, K); //Warmup
    double total_time = 0.0;
    for (int i = 0; i < num_iterations; i++) {
        double start = get_time_ms();
        matmul(A, B, C, M, N, K);
        __asm__ __volatile__("" : "+m" (C[0]) : : "memory");
        double end = get_time_ms();
        total_time += end - start;
    }

    return total_time/num_iterations;
}

// ============================================================================
// Main: Verify results and performance benchmakrk
// ============================================================================
int main(int argc, char *argv[]) {
    srand(42);
    printf("MatMul Benchmark: Square Matrix\n");

    int sizes[] = {512, 256, 128, 64};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s %-15s %-15s %-15s %-15s\n", "Size", "Naive", "Reordered", "Tiled", "Parallel");
    printf("%-8s %-15s %-15s %-15s %-15s\n", "----", "-----", "---------", "-----", "--------");

    for (int i = 0; i < n; i++) {
        int M = sizes[i], N = M, K = M;

        float *A = (float *)malloc(M * K * sizeof(float));
        float *B = (float *)malloc(K * N * sizeof(float));
        float *C = (float *)malloc(M * N * sizeof(float));
        float *C_ref = (float *)malloc(M * N * sizeof(float));

        initialize_matrix(A, M, K);
        initialize_matrix(B, K, N);

        matmul_naive(A, B, C_ref, M, N, K);

        // --- 1. Naive ---
        memset(C, 0, M * N * sizeof(float));

        float t_naive = benchmark(matmul_naive, A, B, C, M, N, K);
        double g_naive = calculate_gflops(M, N, K, t_naive);

        // --- 2. Tiled ---
        memset(C, 0, M * N * sizeof(float));
        matmul_looptiling(A, B, C, M, N, K);
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Tiled verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }

        float t_blocking = benchmark(matmul_looptiling, A, B, C, M, N, K);
        double g_blocking = calculate_gflops(M, N, K, t_blocking);

        // --- 3. Reordered ---
        memset(C, 0, M * N * sizeof(float));
        matmul_looporder(A, B, C, M, N, K);
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Reordered verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }

        float t_reorder = benchmark(matmul_looporder, A, B, C, M, N, K);
        double g_reorder = calculate_gflops(M, N, K, t_reorder);

        // --- 4. Parallel ---
        memset(C, 0, M * N * sizeof(float));
        matmul_parallel_ikj(A, B, C, M, N, K);
        if (!verify_result(C_ref, C, M, N, 1e-3f)) {
            fprintf(stderr, "Parallel verification failed for size %d\n", M);
            free(A); free(B); free(C); free(C_ref);
            return 1;
        }

        float t_parallel = benchmark(matmul_parallel_ikj, A, B, C, M, N, K);
        double g_parallel = calculate_gflops(M, N, K, t_parallel);

        printf("%d\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\t%.2f GFLOPS\n",
               M, g_naive, g_reorder, g_blocking, g_parallel);

        free(A); free(B); free(C); free(C_ref);
    }

    return 0;
}

*/