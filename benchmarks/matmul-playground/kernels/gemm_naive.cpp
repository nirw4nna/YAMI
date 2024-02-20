#include "gemm_naive.h"

f64 gemm_naive(usize m, usize n, usize k,
               f32 *a, usize lda,
               f32 *b, usize ldb,
               f32 *c, usize ldc) noexcept{
    const f64 start = now();
    for (usize i = 0; i < m; ++i) {
        for (usize j = 0; j < n; ++j) {
            for (usize p = 0; p < k; ++p) {
                c[i * ldc + j] += a[i * lda + p] * b[p * ldb + j];
            }
        }
    }
    return now() - start;
}
