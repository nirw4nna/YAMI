#pragma once

#include "../platform.h"

#if defined(__cplusplus)
extern "C" {
#endif

    extern f64 gemm_yami(const usize m, const usize n, const usize k,
                        const f32 *__restrict a, const usize lda,
                        const f32 *__restrict b, const usize ldb,
                        f32 *__restrict c, const usize ldc) noexcept;

#if defined(__cplusplus)
}
#endif
