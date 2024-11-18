// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <ctime>

#define YAMI_LOG_FATAL(format, ...) \
    do {                            \
        YAMI_LOG_ERR(format, ##__VA_ARGS__);    \
        exit(EXIT_FAILURE);                     \
    } while(0)

#define YAMI_LOG_ERR(format, ...)   fprintf(stderr, "%s: " format"\n",__func__, ##__VA_ARGS__)
#define YAMI_LOG_INFO(format, ...)  fprintf(stdout, "%s: " format"\n",__func__, ##__VA_ARGS__)

#define YAMI_ASSERT(x)  \
    do{                 \
        if (!(x)) {     \
            fprintf(stderr, "YAMI_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x); \
            exit(EXIT_FAILURE);                                                 \
        } \
    } while(0)

#if defined(YAMI_DEBUG)
#   define YAMI_LOG_DEBUG(format, ...)  YAMI_LOG_INFO(format, ##__VA_ARGS__)
#else
#   define YAMI_LOG_DEBUG(format, ...)  ((void) 0)
#endif

#define YAMI_UNUSED(x) ((void) (x))

#if defined(__GNUC__)
// A 'strictly pure' function is a function whose return value doesn't depend on the global state of the program,
// this means that it must not access global variables subject to change or access parameters passed by pointer
// unless the actual value of the pointer does not change after the first invocation.
// A 'pure' function is basically the same thing without the restriction on global state change, this means
// that a 'pure' function can take in and read the value of parameters passed by pointer even if that value
// changes between subsequent invocations.
#   define YAMI_STRICTLY_PURE   __attribute_const__
#   define YAMI_PURE            __attribute_pure__
#   define YAMI_INLINE          inline __attribute__((always_inline))
#   define YAMI_NOINLINE        __attribute__((__noinline__))
#else
#   define YAMI_STRICTLY_PURE
#   define YAMI_PURE
#   define YAMI_INLINE
#endif

#define YAMI_MAX(x, y)      ((x) > (y) ? (x) : (y))
#define YAMI_MIN(x, y)      ((x) < (y) ? (x) : (y))
#define YAMI_B_TO_KB(b)     ((f64)(b) / 1024.)
#define YAMI_B_TO_MB(b)     ((f64)(b) / (1024. * 1024.))
// Compute the next value of X aligned to Y
#define YAMI_ALIGN(x, y)    (((x) + (y) - 1) & ~((y) - 1))

#define YAMI_MAX_DIMS   ((int) 4)
#define YAMI_LABEL_SIZE ((int) 64)
#define YAMI_MINUS_INF  (-std::numeric_limits<f32>::infinity())
#define YAMI_SCOPES     ((int) 3)

#if !defined(YAMI_PAGE_SIZE)
#   define YAMI_PAGE_SIZE ((usize) 4096)
#endif

static_assert(YAMI_MAX_DIMS == 4, "YAMI_MAX_DIMS != 4: update");
static_assert(YAMI_SCOPES == 3, "YAMI_SCOPES != 3: update");


// Get the index of dimension 'dim' of tensor 'PTR'.
// dim is usize[YAMI_MAX_DIMS] = [1 x x x] so the actual first valid index is always at YAMI_MAX_DIMS - n_dim while the last
// dimension index is always at YAMI_MAX_DIMS - 1.
// TODO: it probably makes sense to distinguish between `get_dim_index` and `get_dim`, both are useful and used to some extent
#define yami_tensor_dim(PTR, dim) (((dim) < 0) ? (YAMI_MAX_DIMS + (dim)) : (YAMI_MAX_DIMS - (PTR)->n_dim + (dim)))


#if defined(__cplusplus)
extern "C" {
#endif
    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using size = ptrdiff_t;
    using usize = size_t;
    using byte = char;
    using f32 = float;
    using f64 = double;

    enum yami_scope : u8 {
        GLOBAL,
        LOCAL,
        PRIVATE,
    };

    static constexpr const char* YAMI_SCOPE[YAMI_SCOPES] = {
            "GLOBAL",
            "LOCAL",
            "PRIVATE"
    };

    struct yami_ctx;
    struct yami_obj;

    struct yami_init_params {
        int n_workers;
        usize nb;
        usize scratch_nb;
    };

    struct yami_tensor {
        usize ne;
        // The shape of this tensor, right-aligned. For example a 1D tensor T of 4 elements
        // will have dim = [1, 1, 1, 4].
        usize dim[YAMI_MAX_DIMS];
        // Stride for a given dimension expressed in number of f32.
        usize stride[YAMI_MAX_DIMS];
        char label[YAMI_LABEL_SIZE];

        f32 *data;

        int n_dim;
        bool contiguous;
    };

    enum yami_mask : u8 {
        LOWER,
        EQUAL,
        GREATER,
    };

    static YAMI_INLINE f64 yami_timer() noexcept {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
    }

    // ============================ Initialization ============================
    extern yami_ctx *yami_init(yami_init_params params) noexcept;
    extern void yami_free(yami_ctx *ctx) noexcept;
    extern void yami_clear_ctx(yami_ctx *ctx) noexcept;
    // ========================================================================

    // ================================= Misc =================================
    extern void yami_set_scope(yami_ctx *ctx, yami_scope scope) noexcept;
    extern YAMI_PURE usize yami_used_mem(const yami_ctx *ctx) noexcept;
    extern void yami_mem_usage(const yami_ctx *ctx) noexcept;
    extern void yami_print_traces(yami_ctx *ctx) noexcept;
    extern void yami_clear_traces(yami_ctx *ctx) noexcept;
    // ========================================================================

    // ========================== Tensor Manipulation =========================
    extern yami_tensor *yami_new_tensor(yami_ctx *ctx, int n_dim,
                                        const usize *dim,
                                        const char *label = "",
                                        void *data = nullptr) noexcept;
    extern yami_tensor *yami_tensor_1d(yami_ctx *ctx, const char *label,
                                       usize dim1) noexcept;
    extern yami_tensor *yami_tensor_2d(yami_ctx *ctx, const char *label,
                                       usize dim1, usize dim2) noexcept;
    extern yami_tensor *yami_tensor_3d(yami_ctx *ctx, const char *label,
                                       usize dim1, usize dim2,
                                       usize dim3) noexcept;
    extern yami_tensor *yami_tensor_4d(yami_ctx *ctx, const char *label,
                                       usize dim1, usize dim2,
                                       usize dim3, usize dim4) noexcept;
    // View requires the tensor to be contiguous in memory
    extern yami_tensor *yami_view_1d(yami_ctx *ctx, yami_tensor *x,
                                     usize dim1, usize offset = 0) noexcept;
    extern yami_tensor *yami_view_2d(yami_ctx *ctx, yami_tensor *x,
                                     usize dim1, usize dim2,
                                     usize offset = 0) noexcept;
    extern yami_tensor *yami_view_3d(yami_ctx *ctx, yami_tensor *x,
                                     usize dim1, usize dim2,
                                     usize dim3, usize offset = 0) noexcept;
    extern yami_tensor *yami_view_4d(yami_ctx *ctx, yami_tensor *x,
                                     usize dim1, usize dim2,
                                     usize dim3, usize dim4,
                                     usize offset = 0) noexcept;
    // Reshape tensor x to the new dimensions.
    // If the new dimensions are compatible with the old ones x will be updated,
    // otherwise this is a NOP.
    // The dimensions must be all explicit and must result in the same number of elements
    // of x (e.g. if you have a 3x4 matrix you can reshape it as a 6x2, but you cannot write -1x2,
    // nor you can reshape it as a 2x3).
    extern yami_tensor *yami_reshape(yami_tensor *x, int n_dims...) noexcept;
    // Transpose dimensions dim1 and dim2 of tensor x.
    // This function operates on the underlying data of x, it just swaps the given dimensions and strides
    // without allocating any new memory.
    // NOTE: after this the tensor will no longer be contiguous in memory so a reshuffling is required
    // before accessing the data field for, say, a memcpy!
    extern yami_tensor *yami_transpose(yami_ctx *,
                                       yami_tensor *x,
                                       int dim1 = -1,
                                       int dim2 = -2) noexcept;
    extern yami_tensor *yami_contiguous(yami_ctx *ctx, yami_tensor *x) noexcept;
    // Returns a tensors which is the lower triangular part of x with the other elements
    // set to mask. x must be at least a 2D tensor.
    extern yami_tensor *yami_lt_mask(yami_ctx *,
                                     yami_tensor *x,
                                     f32 mask,
                                     usize start_idx = 0) noexcept;
    extern yami_tensor *yami_mask_if(yami_ctx *,
                                     yami_tensor *x,
                                     yami_mask flag,
                                     f32 val,
                                     f32 mask) noexcept;
    extern yami_tensor *yami_embed(yami_ctx *ctx,
                                   const yami_tensor *x,
                                   const int *indexes,
                                   usize n) noexcept;
    extern yami_tensor *yami_rope(yami_ctx *,
                                  yami_tensor *x,
                                  usize n,
                                  bool k_mode,
                                  usize start_idx = 0) noexcept;
    extern yami_tensor *yami_split(yami_ctx *ctx,
                                   const yami_tensor *x,
                                   usize n,
                                   int offset,
                                   int dim = -1) noexcept;
    // Concatenate xa with xb, both xa and xb must have the same shape except for the dimension along which
    // the concat will happen.
    // Note: xa and xb are assumed to be contiguous.
    // Todo: have the possibility to append to the first tensor, this way copy is not needed
    extern yami_tensor *yami_concat(yami_ctx *ctx,
                                    const yami_tensor *xa,
                                    const yami_tensor *xb,
                                    int dim = 0) noexcept;
    // Copy x into res. This is a very dangerous function!
    // Currently, there are no checks in place to verify whether the underlying
    // buffer of res is big enough for x. Also, after this call all the information
    // regarding the original size and shape of res will be lost.
    extern void yami_copy(const yami_tensor *x,
                          yami_tensor *res) noexcept;
    // ========================================================================

    // =========================== Tensor Operations ==========================
    extern yami_tensor *yami_matmul(yami_ctx *ctx,
                                    const yami_tensor *__restrict xa,
                                    const yami_tensor *__restrict xb) noexcept;
    extern yami_tensor *yami_add(yami_ctx *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_addc(yami_ctx *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_sub(yami_ctx *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_subc(yami_ctx *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_mul(yami_ctx *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_mulc(yami_ctx *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_div(yami_ctx *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_divc(yami_ctx *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    // ========================================================================

    // ============================ Math Functions ============================
    extern yami_tensor *yami_gelu(yami_ctx *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_swiglu(yami_ctx *ctx,
                                    yami_tensor *x,
                                    bool in_place = true) noexcept;
    extern yami_tensor *yami_sum(yami_ctx *ctx,
                                     const yami_tensor *x,
                                     int dim) noexcept;
    extern yami_tensor *yami_mean(yami_ctx *ctx,
                                  const yami_tensor *x,
                                  int dim) noexcept;
    extern yami_tensor *yami_var(yami_ctx *ctx,
                                 yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_exp(yami_ctx *ctx,
                                 yami_tensor *x,
                                 bool in_place = true) noexcept;
    extern yami_tensor *yami_sqrt(yami_ctx *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_rsqrt(yami_ctx *ctx,
                                   yami_tensor *x,
                                   bool in_place = true) noexcept;
    extern yami_tensor *yami_square(yami_ctx *ctx,
                                    yami_tensor *x,
                                    bool in_place = true) noexcept;
    extern yami_tensor *yami_max(yami_ctx *ctx,
                                 const yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_softmax(yami_ctx *ctx,
                                     yami_tensor *x,
                                     int dim = -1) noexcept;
    extern yami_tensor *yami_layer_norm(yami_ctx *ctx,
                                        const yami_tensor *w,
                                        const yami_tensor *b,
                                        yami_tensor *x,
                                        f32 eps = 1e-5f) noexcept;
    extern yami_tensor *yami_rms_norm(yami_ctx *ctx,
                                      const yami_tensor *w,
                                      yami_tensor *x,
                                      f32 eps = 1e-5f) noexcept;
    // ========================================================================

#if defined(__cplusplus)
}
#endif
