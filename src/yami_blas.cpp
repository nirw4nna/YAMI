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
            for (usize jj = 0; jj < jb; ++jj) *packed_b++ = b[(j + jj) * stride_b + p];
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
            yami_packB_f32(pb, jb, &b[j * stride_b + p], stride_b, packed_b);

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


void yami_gevm_f32_prefetch(const usize n, const usize k,
                            const f32 *__restrict a,
                            const f32 *__restrict b, const usize stride_b,
                            f32 *__restrict c, void *work) noexcept {
    //    f32 *packed_b = (f32 *) work;
    //    alignas(32) f32 packed_b[KC * NC];
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
    //            for (usize jj = 0; jj < jb; jj += NR_v) {
    //                _mm_prefetch(&c[(j + jj) + 0 * 8], _MM_HINT_T0);
    //                _mm_prefetch(&c[(j + jj) + 2 * 8], _MM_HINT_T0);
    //                _mm_prefetch(&c[(j + jj) + 4 * 8], _MM_HINT_T0);
    //
    //                const usize jjb = YAMI_MIN(jb - jj, NR_v);
    //
    //                if (jjb == NR_v) {
    //                    f32x8 gamma_j_0 = _mm256_loadu_ps(&c[(j + jj) + 0 * 8]);
    //                    f32x8 gamma_j_1 = _mm256_loadu_ps(&c[(j + jj) + 1 * 8]);
    //                    f32x8 gamma_j_2 = _mm256_loadu_ps(&c[(j + jj) + 2 * 8]);
    //                    f32x8 gamma_j_3 = _mm256_loadu_ps(&c[(j + jj) + 3 * 8]);
    //                    f32x8 gamma_j_4 = _mm256_loadu_ps(&c[(j + jj) + 4 * 8]);
    //                    f32x8 gamma_j_5 = _mm256_loadu_ps(&c[(j + jj) + 5 * 8]);
    //
    //                    for (usize pp = 0; pp < pb; ++pp) {
    //                        _mm_prefetch(&packed_b[(jj * pb) + (pp * NR_v) + (8 * 8)], _MM_HINT_T0);
    //                        _mm_prefetch(&packed_b[(jj * pb) + (pp * NR_v) + (10 * 8)], _MM_HINT_T0);
    //                        _mm_prefetch(&packed_b[(jj * pb) + (pp * NR_v) + (12 * 8)], _MM_HINT_T0);
    //                        const f32x8 alpha_p = _mm256_broadcast_ss(&a[p + pp]);
    //
    //                        const f32x8 beta_jp_0 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (0 * 8)]);
    //                        gamma_j_0 = _mm256_fmadd_ps(alpha_p, beta_jp_0, gamma_j_0);
    //
    //                        const f32x8 beta_jp_1 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (1 * 8)]);
    //                        gamma_j_1 = _mm256_fmadd_ps(alpha_p, beta_jp_1, gamma_j_1);
    //
    //                        const f32x8 beta_jp_2 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (2 * 8)]);
    //                        gamma_j_2 = _mm256_fmadd_ps(alpha_p, beta_jp_2, gamma_j_2);
    //
    //                        const f32x8 beta_jp_3 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (3 * 8)]);
    //                        gamma_j_3 = _mm256_fmadd_ps(alpha_p, beta_jp_3, gamma_j_3);
    //
    //                        const f32x8 beta_jp_4 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (4 * 8)]);
    //                        gamma_j_4 = _mm256_fmadd_ps(alpha_p, beta_jp_4, gamma_j_4);
    //
    //                        const f32x8 beta_jp_5 = _mm256_loadu_ps(&packed_b[(jj * pb) + (pp * NR_v) + (5 * 8)]);
    //                        gamma_j_5 = _mm256_fmadd_ps(alpha_p, beta_jp_5, gamma_j_5);
    //                    }
    //
    //                    _mm256_storeu_ps(&c[(j + jj) + (0 * 8)], gamma_j_0);
    //                    _mm256_storeu_ps(&c[(j + jj) + (1 * 8)], gamma_j_1);
    //                    _mm256_storeu_ps(&c[(j + jj) + (2 * 8)], gamma_j_2);
    //                    _mm256_storeu_ps(&c[(j + jj) + (3 * 8)], gamma_j_3);
    //                    _mm256_storeu_ps(&c[(j + jj) + (4 * 8)], gamma_j_4);
    //                    _mm256_storeu_ps(&c[(j + jj) + (5 * 8)], gamma_j_5);
    //                } else {
    //                    for (usize pp = 0; pp < pb; ++pp) {
    //                        for (usize jjj = 0; jjj < jjb; ++jjj) {
    //                            c[j + jj + jjj] += a[p + pp] * packed_b[(jj * pb) + (pp * NR_v) + jjj];
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //    }
    YAMI_UNUSED(work);
    const usize jb = (n / 64) * 64;

    for (usize p = 0; p < k; ++p) {
        const f32x8 alpha_p = _mm256_broadcast_ss(&a[p]);

        for (usize j = 0; j < jb; j += 64) {
            _mm_prefetch(&c[(j + 1) + (0 * 8)], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 1) + (2 * 8)], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 1) + (4 * 8)], _MM_HINT_T0);
            _mm_prefetch(&c[(j + 1) + (6 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[(p * stride_b) + (j + 8 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[(p * stride_b) + (j + 8 * 10)], _MM_HINT_T0);
            _mm_prefetch(&b[(p * stride_b) + (j + 8 * 12)], _MM_HINT_T0);
            _mm_prefetch(&b[(p * stride_b) + (j + 8 * 14)], _MM_HINT_T0);

            f32x8 gamma_j_0 = _mm256_loadu_ps(&c[j + (0 * 8)]);
            f32x8 gamma_j_1 = _mm256_loadu_ps(&c[j + (1 * 8)]);
            f32x8 gamma_j_2 = _mm256_loadu_ps(&c[j + (2 * 8)]);
            f32x8 gamma_j_3 = _mm256_loadu_ps(&c[j + (3 * 8)]);
            f32x8 gamma_j_4 = _mm256_loadu_ps(&c[j + (4 * 8)]);
            f32x8 gamma_j_5 = _mm256_loadu_ps(&c[j + (5 * 8)]);
            f32x8 gamma_j_6 = _mm256_loadu_ps(&c[j + (6 * 8)]);
            f32x8 gamma_j_7 = _mm256_loadu_ps(&c[j + (7 * 8)]);


            const f32x8 beta_pj_0 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 0 * 8)]);
            const f32x8 beta_pj_1 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 1 * 8)]);
            const f32x8 beta_pj_2 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 2 * 8)]);
            const f32x8 beta_pj_3 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 3 * 8)]);
            const f32x8 beta_pj_4 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 4 * 8)]);
            const f32x8 beta_pj_5 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 5 * 8)]);
            const f32x8 beta_pj_6 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 6 * 8)]);
            const f32x8 beta_pj_7 = _mm256_loadu_ps(&b[(p * stride_b) + (j + 7 * 8)]);

            gamma_j_0 = _mm256_fmadd_ps(alpha_p, beta_pj_0, gamma_j_0);
            gamma_j_1 = _mm256_fmadd_ps(alpha_p, beta_pj_1, gamma_j_1);
            gamma_j_2 = _mm256_fmadd_ps(alpha_p, beta_pj_2, gamma_j_2);
            gamma_j_3 = _mm256_fmadd_ps(alpha_p, beta_pj_3, gamma_j_3);
            gamma_j_4 = _mm256_fmadd_ps(alpha_p, beta_pj_4, gamma_j_4);
            gamma_j_5 = _mm256_fmadd_ps(alpha_p, beta_pj_5, gamma_j_5);
            gamma_j_6 = _mm256_fmadd_ps(alpha_p, beta_pj_6, gamma_j_6);
            gamma_j_7 = _mm256_fmadd_ps(alpha_p, beta_pj_7, gamma_j_7);


            _mm256_storeu_ps(&c[j + (0 * 8)], gamma_j_0);
            _mm256_storeu_ps(&c[j + (1 * 8)], gamma_j_1);
            _mm256_storeu_ps(&c[j + (2 * 8)], gamma_j_2);
            _mm256_storeu_ps(&c[j + (3 * 8)], gamma_j_3);
            _mm256_storeu_ps(&c[j + (4 * 8)], gamma_j_4);
            _mm256_storeu_ps(&c[j + (5 * 8)], gamma_j_5);
            _mm256_storeu_ps(&c[j + (6 * 8)], gamma_j_6);
            _mm256_storeu_ps(&c[j + (7 * 8)], gamma_j_7);
        }

        for (usize j = jb; j < n; ++j) {
            c[j] += a[p] * b[p * stride_b + j];
        }
    }
    //
    //    YAMI_UNUSED(work);
    //    const usize jb = (n / 8) * 8;
    //
    //    for (usize p = 0; p < k; ++p) {
    //        const f32x8 alpha_p = _mm256_broadcast_ss(&a[p]);
    //
    //        for (usize j = 0; j < jb; j += 8) {
    //            f32x8 gamma_j = _mm256_loadu_ps(&c[j]);
    //            const f32x8 beta_pj = _mm256_loadu_ps(&b[p * stride_b + j]);
    //            gamma_j = _mm256_fmadd_ps(alpha_p, beta_pj, gamma_j);
    //
    //            _mm256_storeu_ps(&c[j], gamma_j);
    //        }
    //
    //        for (usize j = jb; j < n; ++j) {
    //            c[j] += a[p] * b[p * stride_b + j];
    //        }
    //    }
}


void yami_gevm_f32(usize n, usize k,
                   const f32 *__restrict a,
                   const f32 *__restrict b, usize stride_b,
                   f32 *__restrict c, void *work) noexcept {
    YAMI_UNUSED(work);
    for (usize j = 0; j < n; ++j) {
        for (usize p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}

void yami_gevm_f32_T(const usize n, const usize k,
                     const f32 *__restrict a,
                     const f32 *__restrict b, const usize stride_b,
                     f32 *__restrict c, void *work) noexcept {
    YAMI_UNUSED(work);
//    const usize pb = (k / 2) * 2;
//
//    for (usize j = 0; j < n; ++j) {
//        f32 acc_0 = 0.f, acc_1 = 0.f;
//
//        for (usize p = 0; p < pb; p += 2) {
//            acc_0 += a[p + 0] * b[(j * stride_b) + (p + 0)];
//            acc_1 += a[p + 1] * b[(j * stride_b) + (p + 1)];
//        }
//
//        c[j] += (acc_0 + acc_1);
//
//        for (usize p = pb; p < k; ++p) {
//            c[j] += a[p] * b[j * stride_b + p];
//        }
//    }
    for (usize j = 0; j < n; ++j) {
        for (usize p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}

// The objective here is to compute the dot product of two vectors A, B of K elements.

// rC = load32(rCPtr)
// rIdx = 0
//
// loop:
//  rA = load32(rAPtr)
//  rB = load32(rBPtr)
//  rAB = rA * rB
//  rC = rC + rAB
//  rAPtr = rAPtr + 4
//  rBPtr = rBPtr + 4
//  rIdx = rIdx + 1
//  if rIdx < K goto loop

// Idea of reduce using AVX2:
//
//  const __m128 left  = _mm256_extractf128_ps(acc, 1);
//  const __m128 right = _mm256_castps256_ps128(acc);
//  const __m128 x128  = _mm_add_ps(left, right);
//  const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
//  const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
//  return  _mm_cvtss_f32(x32);


static YAMI_INLINE f32 reduce(const f32 *__restrict a,
                              const f32 *__restrict b) noexcept {
    // Load alpha and beta (Latency=7, TH~0.5)
    const f32x8 alpha_p = _mm256_loadu_ps(a);
    const f32x8 beta_pj = _mm256_loadu_ps(b);

    // Multiply alpha and beta element-wise acc = [a0*b0, .., a7*b7] (Latency=4, TH=0.5)
    const __m256 acc = _mm256_mul_ps(alpha_p, beta_pj);

    // Accumulate:

    // Extract the upper 4 floats [a4*b4, a5*b5, a6*b6, a7*b7] (Latency=4, TH=1)
    const __m128 upper = _mm256_extractf128_ps(acc, 1);
    // Cast the acc to __m128 to extract the lower 4 floats [a0*b0, a1*b1, a2*b2, a3*b3] (0 latency)
    const __m128 lower = _mm256_castps256_ps128(acc);

    // Sum [a4*b4 + a0*b0, a5*b5 + a1*b1, a6*b6 + a2*b2, a7*b7 + a3*b3] (Latency=4, TH=0.5)
    const __m128 partial_2 = _mm_add_ps(upper, lower);

    // Now that we have `partial_2` to get to the next step we just need to add the first two elements of partial_2
    // with its last two elements. To do so we can use `_mm_movehl_ps(partial_sum_128, partial_sum_128)` this will give us [a6*b6 + a2*b2, a7*b7 + a3*b3, a6*b6 + a2*b2, a7*b7 + a3*b3]
    // (Latency=1, TH=1).

    // Sum [a4*b4 + a0*b0, a5*b5 + a1*b1, a6*b6 + a2*b2, a7*b7 + a3*b3] + [a6*b6 + a2*b2, a7*b7 + a3*b3, a6*b6 + a2*b2, a7*b7 + a3*b3] =
    // [(a4*b4+a0*b0) + (a6*b6+a2*b2), (a5*b5+a1*b1) + (a7*b7+a3*b3), (a6*b6+a2*b2) + (a6*b6+a2*b2), (a7*b7+a3*b3) + (a7*b7+a3*b3)]
    // (Latency=4, TH=0.5)
    const __m128 partial_4 = _mm_add_ps(partial_2, _mm_movehl_ps(partial_2, partial_2));

    // 0x01 should be enough to shuffle, we just care about the first element (Latency=1, TH=0.5)
    // (Latency=4, TH=0.5)
    return _mm_cvtss_f32(_mm_add_ss(partial_4, _mm_shuffle_ps(partial_4, partial_4, 0x01)));
}

//// Load alpha and beta (Latency=7, TH~0.5)
//const f32x8 alpha_p = _mm256_loadu_ps(&a[p]);
//const f32x8 beta_pj = _mm256_loadu_ps(&b[j * stride_b + p]);
//
//// Multiply alpha and beta element-wise acc = [a0*b0, .., a7*b7] (Latency=4, TH=0.5)
//const __m256 a_b = _mm256_mul_ps(alpha_p, beta_pj);
//
//// Accumulate:
//const __m256 partial_2 = _mm256_hadd_ps(a_b, a_b);
//const __m256 partial_4 = _mm256_hadd_ps(partial_2, partial_2);
//c[j] = (c[j] + _mm256_cvtss_f32(partial_4)) + _mm_cvtss_f32(_mm256_extractf128_ps(partial_4, 1));

void yami_gevm_f32_T_simd(usize n, usize k,
                          const f32 *__restrict a,
                          const f32 *__restrict b, usize stride_b,
                          f32 *__restrict c, void *work) noexcept {

    YAMI_UNUSED(work);

    const usize pb = (k / 64) * 64;

    for (usize j = 0; j < n; ++j) {

        f32 acc_0 = 0.f, acc_1 = 0.f, acc_2 = 0.f, acc_3 = 0.f,
            acc_4 = 0.f,acc_5 = 0.f, acc_6 = 0.f, acc_7 = 0.f;
        for (usize p = 0; p < pb; p += 64) {
            _mm_prefetch(&a[p + 4 * 8], _MM_HINT_T0);
            _mm_prefetch(&a[p + 6 * 8], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 4 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 6 * 8)], _MM_HINT_T0);

            acc_0 += reduce(&a[p + 0 * 8], &b[j * stride_b + (p + 0 * 8)]);
            acc_1 += reduce(&a[p + 1 * 8], &b[j * stride_b + (p + 1 * 8)]);
            acc_2 += reduce(&a[p + 2 * 8], &b[j * stride_b + (p + 2 * 8)]);
            acc_3 += reduce(&a[p + 3 * 8], &b[j * stride_b + (p + 3 * 8)]);

            _mm_prefetch(&a[p + 8 * 8], _MM_HINT_T0);
            _mm_prefetch(&a[p + 10 * 8], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 8 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 10 * 8)], _MM_HINT_T0);

            acc_4 += reduce(&a[p + 4 * 8], &b[j * stride_b + (p + 4 * 8)]);
            acc_5 += reduce(&a[p + 5 * 8], &b[j * stride_b + (p + 5 * 8)]);
            acc_6 += reduce(&a[p + 6 * 8], &b[j * stride_b + (p + 6 * 8)]);
            acc_7 += reduce(&a[p + 7 * 8], &b[j * stride_b + (p + 7 * 8)]);
        }

        c[j] += (acc_0 + acc_1) + (acc_2 + acc_3) + (acc_4 + acc_5) + (acc_6 + acc_7);

        for (usize p = pb; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}
