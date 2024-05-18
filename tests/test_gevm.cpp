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
    f32 *c_T = (f32 *) calloc(m*n, sizeof(f32));
    f32 *c_T_simd = (f32 *) calloc(m*n, sizeof(f32));

    random_matrix(m, k, a);
    random_matrix(k, n, b);
    transpose(k, n, b, b_T);

    f64 delay_t = -INF, delay_t_simd = -INF, delay_blas = -INF;
//    f64 delay_t = INF, delay_t_simd = INF, delay_blas = INF;

    for (int i = 0; i < REPEAT; ++i) {
        memset(c_blas, 0, m*n*sizeof(f32));
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
            yami_gevm_f32(n, k, a, b_T, n, c_T, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_t = this_delay > delay_t ? this_delay : delay_t;
            delay_t = this_delay < delay_t ? this_delay : delay_t;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32_simd(n, k, a, b_T, k, c_T_simd, nullptr);
            const f64 this_delay = yami_timer() - start;
//            delay_t_simd = this_delay > delay_t_simd ? this_delay : delay_t_simd;
            delay_t_simd = this_delay < delay_t_simd ? this_delay : delay_t_simd;
        }
    }

    printf("(1,%d) x (%d,%d)\n  -> BLAS took: %.2fms (%.2f GFLOPS)\n  -> B^T took: %.2fms (%.2f GFLOPS)\n  -> B^T SIMD took: %.2fms (%.2f GFLOPS)\n",
           k, k, n,
           delay_blas * 1e3,
           flops / (delay_blas * 1e9),
           delay_t * 1e3,
           flops / (delay_t * 1e9),
           delay_t_simd * 1e3,
           flops / (delay_t_simd * 1e9)
    );

    const f32 diff_T = max_diff(c_blas, c_T, m*n);
    const f32 diff_T_simd = max_diff(c_blas, c_T_simd, m*n);

    printf("Max diff T = %.2e\nMax diff T SIMD = %.2e\n",
           diff_T,
           diff_T_simd
    );

    free(a);
    free(b);
    free(b_T);
    free(c_blas);
    free(c_T);
    free(c_T_simd);
}