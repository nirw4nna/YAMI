#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define YAMI_LOG_INFO(format, ...)  fprintf(stdout, "%s: " format"\n",__func__, ##__VA_ARGS__)
#define YAMI_LOG_ERR(format, ...)   fprintf(stderr, "%s: " format"\n",__func__, ##__VA_ARGS__)
#define YAMI_ASSERT(x)  \
    do{                 \
        if (!(x)) {     \
            fprintf(stderr, "YAMI_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x); \
            exit(EXIT_FAILURE);                                                 \
        } \
    } while(0)

#ifdef YAMI_DEBUG
#   define YAMI_LOG_DEBUG(format, ...) YAMI_LOG_INFO(format, ##__VA_ARGS__)
#else
#   define YAMI_LOG_DEBUG(format, ...)  ((void) 0)
#endif

#define YAMI_MAX(x, y)  ((x) > (y) ? (x) : (y))

extern "C" {
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

    constexpr static int yami_max_dims = 4;
    constexpr static int yami_max_label = 64;

    struct yami_context;
    struct yami_obj;

    struct yami_context_init_params {
        size mem_size;
        size scratch_mem_size;
        void *mem_buffer;
        void *scratch_mem_buffer;
    };

    struct yami_tensor {
        int n_dim;
        size ne;
        // The shape of this tensor.
        size dimensions[yami_max_dims];
        // The shape of this tensor in its complete form, for example a 1D tensor T of 4 elements
        // will have T.dimensions = [4, 0, 0, 0] and T.extended_dim = [1, 1, 1, 4].
        // This information is used when broadcasting the tensor.
        size extended_dim[yami_max_dims];
        // Stride for a given dimension expressed in number of f32.
        // For example: given a tensor T with dimensions [2, 3, 2, 2] T.stride[3] = 1, T.stride[1] = 2*2
        size stride[yami_max_dims];
        char label[yami_max_label];
        f32 *data;
    };

    // ============================ Initialization ============================
    extern yami_context *yami_init(yami_context_init_params params) noexcept;
    extern void yami_free(yami_context *ctx) noexcept;
    extern void yami_clear_ctx(yami_context *ctx) noexcept;
    // ========================================================================

    // ============================== Misc Stats ==============================
    extern void yami_mem_usage(const yami_context *ctx) noexcept;
    // ========================================================================

    // ========================== Tensor Manipulation =========================
    extern yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
                                       size dim1) noexcept;
    extern yami_tensor *yami_tensor_2d(yami_context *ctx, const char *label,
                                       size dim1, size dim2) noexcept;
    extern yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
                                       size dim1, size dim2,
                                       size dim3) noexcept;
    extern yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
                                       size dim1, size dim2,
                                       size dim3, size dim4) noexcept;
    extern yami_tensor *yami_clone(yami_context *ctx,
                                   const yami_tensor *x) noexcept;
    // Reshape tensor x to the new dimensions.
    // If the new dimensions are compatible with the old ones x will be updated,
    // otherwise this is a NOP.
    // The dimensions must be all explicit and must result in the same number of elements
    // of x (e.g. if you have a 3x4 matrix you can reshape it as a 6x2, but you cannot write -1x2,
    // nor you can reshape it as a 2x3).
    extern void yami_reshape(yami_tensor *x, int n_dims...) noexcept;
    // Transpose dimensions dim1 and dim2 of tensor x.
    // By default, this function will transpose the last two dimensions of x
    // in a newly allocated tensor.
    // Todo: implement the in-place version...
    extern yami_tensor *yami_transpose(yami_context *ctx,
                                       yami_tensor *x,
                                       int dim1 = -1,
                                       int dim2 = -2,
                                       bool in_place = false) noexcept;
    // ========================================================================

    // =========================== Tensor Operations ==========================
    extern yami_tensor *yami_matmul(yami_context *ctx,
                                    const yami_tensor *xa,
                                    const yami_tensor *xb) noexcept;
    extern yami_tensor *yami_add(yami_context *ctx,
                                 const yami_tensor *xa,
                                 const yami_tensor *xb) noexcept;
    extern yami_tensor *yami_sub(yami_context *ctx,
                                 const yami_tensor *xa,
                                 const yami_tensor *xb) noexcept;
    extern yami_tensor *yami_mul(yami_context *ctx,
                                 const yami_tensor *xa,
                                 const yami_tensor *xb) noexcept;
    extern yami_tensor *yami_div(yami_context *ctx,
                                 const yami_tensor *xa,
                                 const yami_tensor *xb) noexcept;
    // ========================================================================

    // ============================ Math Functions ============================
    extern yami_tensor *yami_tanh(yami_context *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_gelu(yami_context *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_sum(yami_context *ctx,
                                 const yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_exp(yami_context *ctx,
                                 yami_tensor *x,
                                 bool in_place = true) noexcept;
    extern yami_tensor *yami_max(yami_context *ctx,
                                 const yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_softmax(yami_context *ctx,
                                     const yami_tensor *x,
                                     int dim) noexcept;
    // ========================================================================
}
