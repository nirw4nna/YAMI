#pragma once

#include "../platform.h"

// A GEMM always computes C += A x B
//
// M = rows of C
// N = columns of C
// K = columns of A
#if defined(__cplusplus)
extern "C" {
#endif

    extern f64 gemm_naive(usize m, usize n, usize k,
                        f32 *a, usize lda,
                        f32 *b, usize ldb,
                        f32 *c, usize ldc) noexcept;

#if defined(__cplusplus)
}
#endif
