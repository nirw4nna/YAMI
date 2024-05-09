#include "yami_blas.h"
#include <cstring>

#if defined(_OPENMP)
#   include <omp.h>
#endif

using f32x8 = __m256;

#define rank1_8x8(A, B, idx)                                    \
    do {                                                        \
        const f32x8 beta_p = _mm256_loadu_ps(&(B)[(idx) * NR]); \
\
        /* Broadcast alpha_0 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 0]);   \
        gamma_0 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_0);   \
\
        /* Broadcast alpha_1 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 1]);   \
        gamma_1 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_1);   \
\
        /* Broadcast alpha_2 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 2]);   \
        gamma_2 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_2);   \
\
        /* Broadcast alpha_3 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 3]);   \
        gamma_3 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_3);   \
\
        /* Broadcast alpha_4 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 4]);   \
        gamma_4 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_4);   \
\
        /* Broadcast alpha_5 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 5]);   \
        gamma_5 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_5);   \
\
        /* Broadcast alpha_6 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 6]);   \
        gamma_6 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_6);   \
\
        /* Broadcast alpha_7 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 7]);   \
        gamma_7 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_7);   \
    } while(0)

// ============================================== Ukernel ==============================================
static YAMI_INLINE void yami_ukernel_8x8_f32(const usize k,
                                             const f32 *__restrict a,
                                             const f32 *__restrict b,
                                             f32 *__restrict c, const usize stride_c) noexcept {
    f32x8 gamma_0 = _mm256_loadu_ps(&c[0 * stride_c]);
    f32x8 gamma_1 = _mm256_loadu_ps(&c[1 * stride_c]);
    f32x8 gamma_2 = _mm256_loadu_ps(&c[2 * stride_c]);
    f32x8 gamma_3 = _mm256_loadu_ps(&c[3 * stride_c]);
    f32x8 gamma_4 = _mm256_loadu_ps(&c[4 * stride_c]);
    f32x8 gamma_5 = _mm256_loadu_ps(&c[5 * stride_c]);
    f32x8 gamma_6 = _mm256_loadu_ps(&c[6 * stride_c]);
    f32x8 gamma_7 = _mm256_loadu_ps(&c[7 * stride_c]);

    f32x8 alpha_pj;

    const usize pb = (k / 4) * 4;
    for (usize p = 0; p < pb; p += 4) {
        rank1_8x8(a, b, p + 0);
        rank1_8x8(a, b, p + 1);
        rank1_8x8(a, b, p + 2);
        rank1_8x8(a, b, p + 3);
    }

    for (usize p = pb; p < k; ++p) {
        rank1_8x8(a, b, p);
    }

    _mm256_storeu_ps(&c[0 * stride_c], gamma_0);
    _mm256_storeu_ps(&c[1 * stride_c], gamma_1);
    _mm256_storeu_ps(&c[2 * stride_c], gamma_2);
    _mm256_storeu_ps(&c[3 * stride_c], gamma_3);
    _mm256_storeu_ps(&c[4 * stride_c], gamma_4);
    _mm256_storeu_ps(&c[5 * stride_c], gamma_5);
    _mm256_storeu_ps(&c[6 * stride_c], gamma_6);
    _mm256_storeu_ps(&c[7 * stride_c], gamma_7);
}
// =====================================================================================================

// ============================================== Packing ==============================================
static YAMI_INLINE void yami_packA_f32(const usize m, const usize k,
                                       const f32 *__restrict a, const usize stride_a,
                                       f32 *__restrict packed_a) noexcept {
    for (usize i = 0; i < m; i += MR) {
        const usize ib = YAMI_MIN(m - i, MR);

        for (usize p = 0; p < k; ++p) {
            for (usize ii = 0; ii < ib; ++ii) *packed_a++ = a[(i + ii) * stride_a + p];
            for (usize ii = ib; ii < MR; ++ii) *packed_a++ = 0.f;
        }
    }
}

static YAMI_INLINE void yami_packB_f32(const usize k, const usize n,
                                       const f32 *__restrict b, const usize stride_b,
                                       f32 *__restrict packed_b) noexcept {
    for (usize j = 0; j < n; j += NR) {
        const usize jb = YAMI_MIN(n - j, NR);

        for (usize p = 0; p < k; ++p) {
            for (usize jj = 0; jj < jb; ++jj) *packed_b++ = b[p * stride_b + (j + jj)];
            for (usize jj = jb; jj < NR; ++jj) *packed_b++ = 0.f;
        }
    }
}
// =====================================================================================================

void yami_gemm_f32(const usize m, const usize n, const usize k,
                   const f32 *__restrict a, const usize stride_a,
                   const f32 *__restrict b, const usize stride_b,
                   f32 *__restrict c, const usize stride_c, void *work) noexcept {
    f32 *packed_a = (f32 *) work;
    f32 *packed_b = (f32 *) (packed_a + (MC * KC));
    alignas(32) f32 packed_c[MR * NR];

    // 5th loop
    for (usize j = 0; j < n; j += NC) {
        const usize jb = YAMI_MIN(n - j, NC);

        // 4th loop
        for (usize p = 0; p < k; p += KC) {
            const usize pb = YAMI_MIN(k - p, KC);

            // Pack B
            yami_packB_f32(pb, jb, &b[p * stride_b + j], stride_b, packed_b);

            // 3rd loop
            for (usize i = 0; i < m; i+= MC) {
                const usize ib = YAMI_MIN(m - i, MC);

                // Pack A
                yami_packA_f32(ib, pb, &a[i * stride_a + p], stride_a, packed_a);

                // 2nd loop
                for (usize ii = 0; ii < ib; ii += MR) {
                    const usize iib = YAMI_MIN(ib - ii, MR);

                    // Prefetch alpha
                    _mm_prefetch(&packed_a[ii * pb], _MM_HINT_T0);

                    // 1st loop
                    for (usize jj = 0; jj < jb; jj += NR) {
                        const usize jjb = YAMI_MIN(jb - jj, NR);

                        // Prefetch the current micro-panel of C
                        _mm_prefetch(&c[(i + ii + 0) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 1) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 2) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 3) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 4) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 5) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 6) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 7) * stride_c + (j + jj)], _MM_HINT_T0);

                        if (iib == MR && jjb == NR) {
                            yami_ukernel_8x8_f32(pb,
                                                  &packed_a[ii * pb],
                                                  &packed_b[jj * pb],
                                                  &c[((i + ii) * stride_c) + (j + jj)],
                                                  stride_c
                            );
                        } else {
                            // Use the optimized ukernel to compute A * B then add the result to C
                            memset(packed_c, 0, MR * NR * sizeof(f32));
                            yami_ukernel_8x8_f32(pb,
                                                  &packed_a[ii * pb],
                                                  &packed_b[jj * pb],
                                                  packed_c,
                                                  NR
                            );
                            for (usize iii = 0; iii < iib; ++iii) {
                                for (usize jjj = 0; jjj < jjb; ++jjj) c[(i + ii + iii) * stride_c + (j + jj + jjj)] += packed_c[iii * NR + jjj];
                            }
                        }
                    }
                }
            }
        }
    }
}

void yami_gevm_f32(const usize n, const usize k,
                   const f32 *__restrict a,
                   const f32 *__restrict b, const usize stride_b,
                   f32 *__restrict c, void *work) noexcept {
//    f32 *packed_b = (f32 *) work;
//
//    for (usize j = 0; j < n; j += NC) {
//        const usize jb = YAMI_MIN(n - j, NC);
//
//        for (usize p = 0; p < k; p += KC) {
//            const usize pb = YAMI_MIN(k - p, KC);
//
//            // Pack B
//            yami_packB_f32(pb, jb, &b[p * stride_b + j], stride_b, packed_b);
//
////            #pragma omp parallel for if (jb >= 6*NR)
//            for (usize jj = 0; jj < jb; jj += NR) {
//                const usize jjb = YAMI_MIN(jb - jj, NR);
//
//                if (jjb == NR) {
//                    f32x8 gamma_j = _mm256_loadu_ps(&c[j + jj]);
//
//                    for (usize pp = 0; pp < pb; ++pp) {
//                        const f32x8 beta_jp = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR)]);
//                        const f32x8 alpha_p = _mm256_broadcast_ss(&a[p + pp]);
//                        gamma_j = _mm256_fmadd_ps(alpha_p, beta_jp, gamma_j);
//                    }
//
//                    _mm256_storeu_ps(&c[j + jj], gamma_j);
//                } else {
//                    for (usize pp = 0; pp < pb; ++pp) {
//                        for (usize jjj = 0; jjj < jjb; ++jjj) {
//                            c[j + jj + jjj] += a[p + pp] * packed_b[(jj * pb) + (pp * NR) + jjj];
//                        }
//                    }
//                }
//            }
//        }
//    }
    YAMI_UNUSED(work);
    const usize jb = (n / 8) * 8;

    for (usize p = 0; p < k; ++p) {
        const f32x8 alpha_p = _mm256_broadcast_ss(&a[p]);

        for (usize j = 0; j < jb; j += 8) {
            f32x8 gamma_j = _mm256_loadu_ps(&c[j]);
            const f32x8 beta_pj = _mm256_loadu_ps(&b[p * stride_b + j]);
            gamma_j = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_j);

            _mm256_storeu_ps(&c[j], gamma_j);
        }

        for (usize j = jb; j < n; ++j) {
            c[j] += a[p] * b[p * stride_b + j];
        }
    }
}