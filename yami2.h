#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define YAMI_LOG_INFO(format, ...)  fprintf(stdout, "%s: " format"\n",__func__, ##__VA_ARGS__)
#define YAMI_LOG_ERR(format, ...)   fprintf(stderr, "%s: " format"\n",__func__, ##__VA_ARGS__)

#ifdef YAMI_DEBUG
#   define YAMI_LOG_DEBUG(format, ...) YAMI_LOG_INFO(format, ##__VA_ARGS__)
#   define YAMI_ASSERT(x) \
       do{                \
            if (!(x)) {      \
                fprintf(stderr, "YAMI_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x); \
                exit(EXIT_FAILURE);                                                 \
            }                   \
       } while(0)
#else
#   define YAMI_ASSERT(x)               ((void) 0)
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
        size dimensions[yami_max_dims];
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
    extern yami_tensor *yami_clone(yami_context *ctx, const yami_tensor *x) noexcept;
    // ========================================================================

    // =========================== Tensor Operations ==========================
    extern yami_tensor *yami_matmul(yami_context *ctx,
                                    const yami_tensor *xa,
                                    const yami_tensor *xb) noexcept;
    // ========================================================================
}
