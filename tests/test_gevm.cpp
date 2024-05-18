#include "yami_blas.h"
#include <cstring>
#include <openblas/cblas.h>

#define INF     (-std::numeric_limits<f64>::infinity())
#define REPEAT  5

static void random_matrix(const int m, const int n, f32 *a) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            a[i * n + j] = 2 * (((f32) rand() / (f32) RAND_MAX) - 0.5f);
}

static f32 max_diff(const f32 *A, const f32 *B, const int n) {
    f32 max = YAMI_MINUS_INF;
    for (int i = 0; i < n; ++i) {
        const f32 diff = std::abs(A[i] - B[i]);
        if (diff > max) {
            max = diff;
        }
    }
    return max;
}

static void gemm_naive(const int m, const int n, const int k,
                       const f32 *a, const f32 *b, f32 *c) {
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const f32 c_a = a[i * k + p];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += c_a * b[p * n + j];
            }
        }
    }
}

static void transpose(const int m, const int n,
                      const f32 *a, f32 *a_T) noexcept {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a_T[j * m + i] = a[i * n + j];
        }
    }
}

int main() {
    const int m = 1, n = 11008, k = 4096;
    const f64 flops = 2 * m * n * k;

    f32 *a = (f32 *) malloc(m*k*sizeof(f32));
    f32 *b = (f32 *) malloc(k*n*sizeof(f32));
    f32 *b_T = (f32 *) malloc(k*n*sizeof(f32));
    f32 *c_blas = (f32 *) calloc(m*n, sizeof(f32));
    f32 *c_v1 = (f32 *) calloc(m*n, sizeof(f32));
    f32 *c_prefetch = (f32 *) calloc(m*n, sizeof(f32));
    f32 *c_T = (f32 *) calloc(m*n, sizeof(f32));
    f32 *c_T_simd = (f32 *) calloc(m*n, sizeof(f32));

    random_matrix(m, k, a);
    random_matrix(k, n, b);
    transpose(k, n, b, b_T);

//    f64 delay_prefetch = INF, delay_t = INF, delay_v1 = INF, delay_t_simd = INF, delay_blas = INF;
    f64 delay_prefetch = -INF, delay_t = -INF, delay_v1 = -INF, delay_t_simd = -INF, delay_blas = -INF;

    for (int i = 0; i < REPEAT; ++i) {
        memset(c_blas, 0, m*n*sizeof(f32));
        memset(c_v1, 0, m*n*sizeof(f32));
        memset(c_prefetch, 0, m*n*sizeof(f32));
        memset(c_T, 0, m*n*sizeof(f32));
        memset(c_T_simd, 0, m*n*sizeof(f32));
        {
            const f64 start = yami_timer();
            cblas_sgemv(CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, k, n, 1.f, b_T, k, a, 1, 0.f, c_blas, 1);
            const f64 this_delay = yami_timer() - start;
//            delay_blas = this_delay > delay_blas ? this_delay : delay_blas;
            delay_blas = this_delay < delay_blas ? this_delay : delay_blas;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32(n, k, a, b, n, c_v1, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_v1 = this_delay > delay_v1 ? this_delay : delay_v1;
            delay_v1 = this_delay < delay_v1 ? this_delay : delay_v1;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32_prefetch(n, k, a, b, n, c_prefetch, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_prefetch = this_delay > delay_prefetch ? this_delay : delay_prefetch;
            delay_prefetch = this_delay < delay_prefetch ? this_delay : delay_prefetch;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32_T(n, k, a, b_T, k, c_T, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_t = this_delay > delay_t ? this_delay : delay_t;
            delay_t = this_delay < delay_t ? this_delay : delay_t;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32_T_simd(n, k, a, b_T, k, c_T_simd, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_t_simd = this_delay > delay_t_simd ? this_delay : delay_t_simd;
            delay_t_simd = this_delay < delay_t_simd ? this_delay : delay_t_simd;
        }
    }

    printf("(1,%d) x (%d,%d)\n  -> BLAS took: %.2fms (%.2f GFLOPS)\n  -> V1 took: %.2fms (%.2f GFLOPS)\n  -> PREFETCH took: %.2fms (%.2f GFLOPS)\n  -> T took: %.2fms (%.2f GFLOPS)\n  -> T SIMD took: %.2fms (%.2f GFLOPS)\n",
           k, k, n,
           delay_blas * 1e3,
           flops / (delay_blas * 1e9),
           delay_v1 * 1e3,
           flops / (delay_v1 * 1e9),
           delay_prefetch * 1e3,
           flops / (delay_prefetch * 1e9),
           delay_t * 1e3,
           flops / (delay_t * 1e9),
           delay_t_simd * 1e3,
           flops / (delay_t_simd * 1e9)
    );

    const f32 diff_v1 = max_diff(c_blas, c_v1, m*n);
    const f32 diff_prefetch = max_diff(c_blas, c_prefetch, m*n);
    const f32 diff_T = max_diff(c_blas, c_T, m*n);
    const f32 diff_T_simd = max_diff(c_blas, c_T_simd, m*n);

    printf("Max diff V1 = %.2e\nMax diff PREFETCH = %.2e\nMax diff T = %.2e\nMax diff T SIMD = %.2e\n",
           diff_v1,
           diff_prefetch,
           diff_T,
           diff_T_simd
    );

    free(a);
    free(b);
    free(b_T);
    free(c_blas);
    free(c_prefetch);
    free(c_v1);
    free(c_T);
    free(c_T_simd);
}