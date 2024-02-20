#include "gemm_naive_restrict.h"

f64 gemm_naive_restrict(const usize m, const usize n, const usize k,
               const f32 *__restrict a, const usize lda,
               const f32 *__restrict b, const usize ldb,
               f32 *__restrict c, const usize ldc) noexcept {
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
