/*
Praktikum 1 – Matrix Multiplication on CPU
AI Accelerators (AIA) – Lab Assignment
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

int THREAD_COUNT = 8;
const int num_iterations = 2;

#define JB 64

static inline int min_int(int a,int b){ return a<b?a:b; }

/* =========================================================
   VERIFY RESULTS
   ========================================================= */

int verify_result(const float* C_ref,const float* C_test,int M,int N,float tol)
{
    for(int i=0;i<M*N;i++)
    {
        if(fabs(C_ref[i]-C_test[i])>tol)
        {
            printf("Mismatch at %d : ref=%f test=%f\n",
                   i,C_ref[i],C_test[i]);
            return 0;
        }
    }
    return 1;
}

/* =========================================================
   TILED KERNEL
   ========================================================= */

static void matmul_tiled_range(const float* A,const float* B,float* C,
                               int row_start,int row_end,int M,int N,int K)
{
    for(int ii=row_start;ii<row_end;ii+=JB)
    {
        int i_end=min_int(ii+JB,row_end);

        for(int kk=0;kk<K;kk+=JB)
        {
            int k_end=min_int(kk+JB,K);

            for(int jj=0;jj<N;jj+=JB)
            {
                int j_end=min_int(jj+JB,N);

                for(int i=ii;i<i_end;i++)
                {
                    float* c_row=&C[i*N];

                    for(int k=kk;k<k_end;k++)
                    {
                        float a=A[i*K+k];
                        const float* b_row=&B[k*N];

                        for(int j=jj;j<j_end;j++)
                            c_row[j]+=a*b_row[j];
                    }
                }
            }
        }
    }
}

#ifdef _WIN32

typedef struct{
    const float* A;
    const float* B;
    float* C;
    int row_start,row_end;
    int M,N,K;
} matmul_thread_args;

DWORD WINAPI matmul_thread_worker(LPVOID arg)
{
    matmul_thread_args* args=(matmul_thread_args*)arg;

    matmul_tiled_range(args->A,args->B,args->C,
                       args->row_start,args->row_end,
                       args->M,args->N,args->K);
    return 0;
}
#endif
/* =========================================================
   1. NAIVE i-j-k
   ========================================================= */

void matmul_naive(const float* A,const float* B,float* C,int M,int N,int K)
{
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
        {
            float sum=0;
            for(int k=0;k<K;k++)
                sum+=A[i*K+k]*B[k*N+j];
            C[i*N+j]=sum;
        }
}

/* =========================================================
   2. i-k-j (REORDERED)
   ========================================================= */

void matmul_looporder(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

    for(int i=0;i<M;i++)
    {
        float* c_row=&C[i*N];

        for(int k=0;k<K;k++)
        {
            float a=A[i*K+k];
            const float* b_row=&B[k*N];

            for(int j=0;j<N;j++)
                c_row[j]+=a*b_row[j];
        }
    }
}

/* =========================================================
   3. j-k-i
   ========================================================= */

void matmul_jki(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

    for(int j=0;j<N;j++)
        for(int k=0;k<K;k++)
        {
            float b=B[k*N+j];
            for(int i=0;i<M;i++)
                C[i*N+j]+=A[i*K+k]*b;
        }
}

/* =========================================================
   4. k-i-j
   ========================================================= */

void matmul_kij(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

    for(int k=0;k<K;k++)
        for(int i=0;i<M;i++)
        {
            float a=A[i*K+k];
            const float* b_row=&B[k*N];
            for(int j=0;j<N;j++)
                C[i*N+j]+=a*b_row[j];
        }
}

/* =========================================================
   5. TILING
   ========================================================= */

void matmul_looptiling(const float* A,const float* B,float* C,int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));
    matmul_tiled_range(A,B,C,0,M,M,N,K);
}

/* =========================================================
   6. PARALLEL
   ========================================================= */

void matmul_parallel_ikj(const float* A,const float* B,float* C,
                         int M,int N,int K)
{
    memset(C,0,(size_t)M*N*sizeof(float));

#ifdef _WIN32

    HANDLE threads[THREAD_COUNT];
    matmul_thread_args args[THREAD_COUNT];

    int chunk=(M+THREAD_COUNT-1)/THREAD_COUNT;
    int launched=0;

    for(int t=0;t<THREAD_COUNT;t++)
    {
        int row_start=t*chunk;
        int row_end=min_int(row_start+chunk,M);
        if(row_start>=row_end) break;

        args[t].A=A;
        args[t].B=B;
        args[t].C=C;
        args[t].row_start=row_start;
        args[t].row_end=row_end;
        args[t].M=M;
        args[t].N=N;
        args[t].K=K;

        threads[t]=CreateThread(NULL,0,matmul_thread_worker,&args[t],0,NULL);
        launched++;
    }

    WaitForMultipleObjects(launched,threads,TRUE,INFINITE);

    for(int t=0;t<launched;t++)
        CloseHandle(threads[t]);

#else

    matmul_tiled_range(A,B,C,0,M,M,N,K);

#endif
}

/* =========================================================
   UTILITY
   ========================================================= */

void initialize_matrix(float *matrix,int rows,int cols)
{
    for(int i=0;i<rows*cols;i++)
        matrix[i]=rand()%100;
}

double get_time_ms()
{
#ifdef _WIN32

    static LARGE_INTEGER freq;
    static int init=0;
    LARGE_INTEGER counter;

    if(!init)
    {
        QueryPerformanceFrequency(&freq);
        init=1;
    }

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart*1000.0/(double)freq.QuadPart;

#else

    return (double)clock()*1000.0/CLOCKS_PER_SEC;

#endif
}

double calculate_gflops(int M,int N,int K,double time)
{
    double flops=2.0*M*N*K;
    return (flops/(time/1000.0))/1e9;
}

typedef void (*matmul_fn)(const float*,const float*,float*,int,int,int);

float benchmark(matmul_fn matmul,const float* A,const float* B,float* C,
                int M,int N,int K)
{
    matmul(A,B,C,M,N,K); /* warmup */

    double total=0;

    for(int i=0;i<num_iterations;i++)
    {
        double start=get_time_ms();
        matmul(A,B,C,M,N,K);
        double end=get_time_ms();

        total+=end-start;
    }

    return total/num_iterations;
}

/* =========================================================
   MAIN
   ========================================================= */

int main()
{
    srand(42);

    int sizes[]={64,128,256,512,1024,2048};
    int n=sizeof(sizes)/sizeof(sizes[0]);

    printf("Size   i-j-k   i-k-j   j-k-i   k-i-j   Tiled   Parallel\n");

    for(int s=0;s<n;s++)
    {
        int M=sizes[s],N=M,K=M;
        /*Memory allocation*/
        float* A=malloc(M*K*sizeof(float));
        float* B=malloc(K*N*sizeof(float));
        float* C=malloc(M*N*sizeof(float));
        float* C_ref=malloc(M*N*sizeof(float));
        /*Initialize the Matrix*/
        initialize_matrix(A,M,K);
        initialize_matrix(B,K,N);

        matmul_naive(A,B,C_ref,M,N,K);

        memset(C,0,M*N*sizeof(float));
        double g_naive=calculate_gflops(M,N,K,
            benchmark(matmul_naive,A,B,C,M,N,K));

        memset(C,0,M*N*sizeof(float));
        double g_reorder=calculate_gflops(M,N,K,
            benchmark(matmul_looporder,A,B,C,M,N,K));

        memset(C,0,M*N*sizeof(float));
        double g_jki=calculate_gflops(M,N,K,
            benchmark(matmul_jki,A,B,C,M,N,K));

        memset(C,0,M*N*sizeof(float));
        double g_kij=calculate_gflops(M,N,K,
            benchmark(matmul_kij,A,B,C,M,N,K));

        memset(C,0,M*N*sizeof(float));
        matmul_looptiling(A,B,C,M,N,K);
        if(!verify_result(C_ref,C,M,N,1e-3f)) return 1;

        memset(C,0,M*N*sizeof(float));
        double g_blocking=calculate_gflops(M,N,K,
            benchmark(matmul_looptiling,A,B,C,M,N,K));

        memset(C,0,M*N*sizeof(float));
        matmul_parallel_ikj(A,B,C,M,N,K);
        if(!verify_result(C_ref,C,M,N,1e-3f)) return 1;

        memset(C,0,M*N*sizeof(float));
        double g_parallel=calculate_gflops(M,N,K,
            benchmark(matmul_parallel_ikj,A,B,C,M,N,K));

        printf("%d   %.2f   %.2f   %.2f   %.2f   %.2f   %.2f\n",
               M,g_naive,g_reorder,g_jki,g_kij,g_blocking,g_parallel);

        free(A);
        free(B);
        free(C);
        free(C_ref);
    }

    return 0;
}