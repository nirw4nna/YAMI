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
    // Compute C += AB^T where A=(m x k) and B=(k x n)
    yami_blas_ctx *ctx = yami_blas_init(-1);

    const int m = 6, n = 11008, k = 4096;
    const f64 flops = 2 * m * n * k;

    f32 *a = (f32 *) malloc(m*k*sizeof(f32));
    f32 *b = (f32 *) malloc(k*n*sizeof(f32));
    f32 *c_blas = (f32 *) malloc(m*n*sizeof(f32));
    f32 *c_yami = (f32 *) malloc(m*n*sizeof(f32));

    random_matrix(m, k, a);
    random_matrix(n, k, b);

    f64 delay_yami = INF, delay_blas = INF;
//    f64 delay_yami = -INF, delay_blas = -INF;

    for (int i = 0; i < REPEAT; ++i) {
        memset(c_blas, 0, m*n*sizeof(f32));
        memset(c_yami, 0, m*n*sizeof(f32));
        {
            const f64 start = yami_timer();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.f, a, k, b, k, 0.f, c_blas, n);
            const f64 this_delay = yami_timer() - start;
//            delay_blas = this_delay > delay_blas ? this_delay : delay_blas;
            delay_blas = this_delay < delay_blas ? this_delay : delay_blas;
        }
        {
            const f64 start = yami_timer();
            yami_gemm_f32(ctx, m, n, k, a, k, b, k, c_yami, n);
            const f64 this_delay = yami_timer() - start;
//            delay_yami = this_delay > delay_yami ? this_delay : delay_yami;
            delay_yami = this_delay < delay_yami ? this_delay : delay_yami;
        }
    }

    printf("(%d,%d) x (%d,%d)\n"
           "  -> OpenBLAS took: %.2fms (%.2f GFLOPS)\n"
           "  -> YAMI took: %.2fms (%.2f GFLOPS)\n",
           m, k, k, n,
           delay_blas * 1e3,
           flops / (delay_blas * 1e9),
           delay_yami * 1e3,
           flops / (delay_yami * 1e9)
    );

    const f32 diff_yami = max_diff(c_blas, c_yami, m*n);

    printf("Max diff YAMI = %.2e\n",
           diff_yami
    );

    yami_blas_free(ctx);

    free(a);
    free(b);
    free(c_blas);
    free(c_yami);
}