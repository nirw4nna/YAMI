#pragma once

// Collection of optimized BLAS-like routines for deep learning. All the non-trivial mathematical
// functions required by YAMI must be implemented here.

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
//#   define NC   2824
#   define NC   96
#endif

#if !defined(KC)
#   define KC   384
#endif


struct yami_blas_ctx;


extern yami_blas_ctx *yami_blas_init(int n_workers) noexcept;
extern void yami_blas_free(yami_blas_ctx *ctx) noexcept;

extern int yami_blas_num_workers(const yami_blas_ctx *ctx) noexcept;

extern void yami_gemm_f32(yami_blas_ctx *ctx,
                          usize m, usize n, usize k,
                          const f32 *__restrict a, usize stride_a,
                          const f32 *__restrict b, usize stride_b,
                          f32 *__restrict c, usize stride_c) noexcept;

extern void yami_gevm_f32(yami_blas_ctx *ctx,
                          usize n, usize k,
                          const f32 *__restrict a,
                          const f32 *__restrict b, usize stride_b,
                          f32 *__restrict c) noexcept;
