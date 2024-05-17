#pragma once

// Collection of optimized BLAS-like routines for deep learning. All the non-trivial mathematical
// functions required by YAMI must be implemented here.
//
// TODO: define the supported data types

#include "yami.h"

#if defined(__AVX2__)
#   include <immintrin.h>
#else
#   error "AVX2 support is required"
#endif

// Register blocking parameters, these are completely architecture-dependant and so, they should be defined
// based on the highest vector extension available. Also, the micro-kernel must be selected based on the
// vector extension.
#define MR 8
#define NR 8

// Cache blocking parameters obtained via analytical methods.
#if !defined(MC)
#   define MC   128
#endif

#if !defined(NC)
#   define NC   2820
#endif

#if !defined(KC)
#   define KC   384
#endif


extern void yami_gemm_f32(usize m, usize n, usize k,
                          const f32 *__restrict a, usize stride_a,
                          const f32 *__restrict b, usize stride_b,
                          f32 *__restrict c, usize stride_c,
                          void *work) noexcept;

extern void yami_gevm_f32(usize n, usize k,
                          const f32 *__restrict a,
                          const f32 *__restrict b, usize stride_b,
                          f32 *__restrict c, void *work) noexcept;

extern void yami_gevm_f32_prefetch(usize n, usize k,
                                   const f32 *__restrict a,
                                   const f32 *__restrict b, usize stride_b,
                                   f32 *__restrict c, void *work) noexcept;

extern void yami_gevm_f32_T(usize n, usize k,
                            const f32 *__restrict a,
                            const f32 *__restrict b, usize stride_b,
                            f32 *__restrict c, void *work) noexcept;

extern void yami_gevm_f32_T_simd(usize n, usize k,
                                 const f32 *__restrict a,
                                 const f32 *__restrict b, usize stride_b,
                                 f32 *__restrict c, void *work) noexcept;