// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#include "yami_blas.h"
#include <cstring>
#include <openblas/cblas.h>


#define INF     std::numeric_limits<f64>::infinity()
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

int main() {
    // Compute C += AB^T where A=(1 x k) and B=(k x n)
    yami_blas_ctx *ctx = yami_blas_init(-1);

    const int m = 1, n = 11008, k = 4096;
    const f64 flops = 2 * m * n * k;

    f32 *a = (f32 *) malloc(m*k*sizeof(f32));
    f32 *b = (f32 *) malloc(k*n*sizeof(f32));
    f32 *c_blas = (f32 *) malloc(m*n*sizeof(f32));
    f32 *c_yami = (f32 *) malloc(m*n*sizeof(f32));
    f32 *c_yami_simd = (f32 *) malloc(m*n*sizeof(f32));

    random_matrix(m, k, a);
    random_matrix(n, k, b);

    f64 delay_yami = INF, delay_yami_simd = INF, delay_blas = INF;
//    f64 delay_yami = -INF, delay_yami_simd = -INF, delay_blas = -INF;

    for (int i = 0; i < REPEAT; ++i) {
        memset(c_blas, 0, m*n*sizeof(f32));
        memset(c_yami, 0, m*n*sizeof(f32));
        memset(c_yami_simd, 0, m*n*sizeof(f32));
        {
            const f64 start = yami_timer();
            cblas_sgemv(CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, k, n, 1.f, b, k, a, 1, 0.f, c_blas, 1);
            const f64 this_delay = yami_timer() - start;
//            delay_blas = this_delay > delay_blas ? this_delay : delay_blas;
            delay_blas = this_delay < delay_blas ? this_delay : delay_blas;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32(ctx, n, k, a, b, k, c_yami);
            const f64 this_delay = yami_timer() - start;
//            delay_t = this_delay > delay_t ? this_delay : delay_t;
            delay_yami = this_delay < delay_yami ? this_delay : delay_yami;
        }
        {
            const f64 start = yami_timer();
            yami_gevm_f32_simd(ctx, n, k, a, b, k, c_yami_simd);
            const f64 this_delay = yami_timer() - start;
//            delay_t_simd = this_delay > delay_t_simd ? this_delay : delay_t_simd;
            delay_yami_simd = this_delay < delay_yami_simd ? this_delay : delay_yami_simd;
        }
    }

    printf("(1,%d) x (%d,%d)\n  -> BLAS took: %.2fms (%.2f GFLOPS)\n  -> YAMI took: %.2fms (%.2f GFLOPS)\n  -> YAMI SIMD took: %.2fms (%.2f GFLOPS)\n",
           k, k, n,
           delay_blas * 1e3,
           flops / (delay_blas * 1e9),
           delay_yami * 1e3,
           flops / (delay_yami * 1e9),
           delay_yami_simd * 1e3,
           flops / (delay_yami_simd * 1e9)
    );

    const f32 diff_yami = max_diff(c_blas, c_yami, m*n);
    const f32 diff_yami_simd = max_diff(c_blas, c_yami_simd, m*n);

    printf("Max diff T = %.2e\nMax diff T SIMD = %.2e\n",
           diff_yami,
           diff_yami_simd
    );

    yami_blas_free(ctx);

    free(a);
    free(b);
    free(c_blas);
    free(c_yami);
    free(c_yami_simd);
}