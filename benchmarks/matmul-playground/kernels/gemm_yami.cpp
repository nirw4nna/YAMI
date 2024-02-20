#include "gemm_yami.h"

#define BLOCK_SIZE 32

f64 gemm_yami(const usize m, const usize n, const usize k,
               const f32 *__restrict a, const usize lda,
               const f32 *__restrict b, const usize ldb,
               f32 *__restrict c, const usize ldc) noexcept {

    const f64 start = now();
    const usize b_size = BLOCK_SIZE * sizeof(f32);
    alignas(32) f32 *packed_a = (f32 *) alloca(b_size);
    alignas(32) f32 *packed_b = (f32 *) alloca(b_size * BLOCK_SIZE);
    alignas(32) f32 *packed_c = (f32 *) alloca(b_size);

    for (usize bj = 0; bj < n; bj += BLOCK_SIZE) {

        const usize max_bj = MIN(bj + BLOCK_SIZE, n);

        for (usize bi = 0; bi < k; bi += BLOCK_SIZE) {

            const usize max_bi = MIN(bi + BLOCK_SIZE, k);

            const usize block_rows = max_bi - bi;
            const usize block_cols = max_bj - bj;
            // Fill the buffer with 0s and copy only those values that are part of the YAMI_BLOCK_SIZE x YAMI_BLOCK_SIZE matrix
            memset(packed_b, 0, b_size * BLOCK_SIZE);
            for (usize ib = 0; ib < block_cols; ++ib) {
                for (usize jb = 0; jb < block_rows; ++jb) {
                    // pack B in column-major order
                    packed_b[ib * BLOCK_SIZE + jb] = b[(bi + jb) * n + (bj + ib)];
                }
            }

            // Take the given subset of rows of A from b_r_start to b_r_stop and multiply by pack_b
            for (usize i = 0; i < m; ++i) {
                memset(packed_a, 0, b_size);
                for (usize tmp = bi; tmp < max_bi; ++tmp) packed_a[tmp - bi] = a[i*k + tmp];

                // Block multiply
                for (usize kk = 0; kk < BLOCK_SIZE; ++kk) {
                    f32 acc = 0.f;
                    for (usize jj = 0; jj < BLOCK_SIZE; ++jj) {
                        acc += packed_a[jj] * packed_b[kk * BLOCK_SIZE + jj];
                    }
                    packed_c[kk] = acc;
                }

                for (usize tmp = bj; tmp < max_bj; ++tmp) {
                    c[i*n + tmp] += packed_c[tmp - bj];
                }
            }
        }
    }
    return now() - start;
}
