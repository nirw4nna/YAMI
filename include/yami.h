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

#ifdef YAMI_DEBUG
#   define YAMI_LOG_DEBUG(format, ...)  YAMI_LOG_INFO(format, ##__VA_ARGS__)
#else
#   define YAMI_LOG_DEBUG(format, ...)  ((void) 0)
#endif

#define YAMI_MAX(x, y)  ((x) > (y) ? (x) : (y))
#define YAMI_MIN(x, y)  ((x) < (y) ? (x) : (y))
#define YAMI_B_TO_KB(b) ((f64)(b) / 1024.)
#define YAMI_B_TO_MB(b) ((f64)(b) / (1024. * 1024.))

#define YAMI_MAX_DIMS   ((int) 4)
#define YAMI_LABEL_SIZE ((int) 64)
#define YAMI_MINUS_INF  (-std::numeric_limits<f32>::infinity())

#ifdef __cplusplus
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

    struct yami_context;
    struct yami_obj;

    struct yami_context_init_params {
        int n_workers;
        usize mem_size;
        usize scratch_mem_size;
    };

    struct yami_tensor {
        usize ne;
        // The shape of this tensor.
        usize dimensions[YAMI_MAX_DIMS];
        // The shape of this tensor in its complete form, for example a 1D tensor T of 4 elements
        // will have T.dimensions = [4, 0, 0, 0] and T.extended_dim = [1, 1, 1, 4].
        // This information is used when broadcasting the tensor.
        usize extended_dim[YAMI_MAX_DIMS];
        // Stride for a given dimension expressed in number of f32.
        // For example: given a tensor T with dimensions [2, 3, 2, 2] T.stride[3] = 1, T.stride[1] = 2*2
        usize stride[YAMI_MAX_DIMS];
        char label[YAMI_LABEL_SIZE];
        f32 *data;
        int n_dim;
    };

    enum yami_mask_flag : u8 {
        LOWER,
        EQUAL,
        GREATER,
    };

    static inline f64 yami_timer() noexcept {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
    }

    // ============================ Initialization ============================
    extern yami_context *yami_init(yami_context_init_params params) noexcept;
    extern void yami_free(yami_context *ctx) noexcept;
    extern void yami_clear_ctx(yami_context *ctx) noexcept;
    // ========================================================================

    // ================================= Misc =================================
    extern yami_context *yami_ctx_scratch(yami_context *ctx) noexcept;
    extern usize yami_used_mem(const yami_context *ctx) noexcept;
    extern void yami_mem_usage(const yami_context *ctx) noexcept;
    // ========================================================================

    // ========================== Tensor Manipulation =========================
    extern yami_tensor *yami_new_tensor(yami_context *ctx, int n_dim,
                                        const usize *dimensions,
                                        const char *label = "",
                                        void *data = nullptr) noexcept;
    extern yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
                                       usize dim1) noexcept;
    extern yami_tensor *yami_tensor_2d(yami_context *ctx, const char *label,
                                       usize dim1, usize dim2) noexcept;
    extern yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
                                       usize dim1, usize dim2,
                                       usize dim3) noexcept;
    extern yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
                                       usize dim1, usize dim2,
                                       usize dim3, usize dim4) noexcept;
    extern yami_tensor *yami_view_1d(yami_context *ctx, yami_tensor *x,
                                     usize dim1, usize offset = 0) noexcept;
    extern yami_tensor *yami_view_2d(yami_context *ctx, yami_tensor *x,
                                     usize dim1, usize dim2,
                                     usize offset = 0) noexcept;
    extern yami_tensor *yami_view_3d(yami_context *ctx, yami_tensor *x,
                                     usize dim1, usize dim2,
                                     usize dim3, usize offset = 0) noexcept;
    extern yami_tensor *yami_view_4d(yami_context *ctx, yami_tensor *x,
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
    extern yami_tensor *yami_transpose(yami_context *,
                                       yami_tensor *x,
                                       int dim1 = -1,
                                       int dim2 = -2) noexcept;
    extern yami_tensor *yami_contiguous(yami_context *ctx, yami_tensor *x) noexcept;
    // Returns a tensors which is the lower triangular part of x with the other elements
    // set to mask. x must be at least a 2D tensor.
    extern yami_tensor *yami_lt_mask(yami_context *,
                                     yami_tensor *x,
                                     f32 mask,
                                     usize start_idx = 0) noexcept;
    extern yami_tensor *yami_mask_if(yami_context *,
                                     yami_tensor *x,
                                     yami_mask_flag flag,
                                     f32 val,
                                     f32 mask) noexcept;
    extern yami_tensor *yami_embed(yami_context *ctx,
                                   const yami_tensor *x,
                                   const int *indexes,
                                   usize n) noexcept;
    extern yami_tensor *yami_split(yami_context *ctx,
                                   const yami_tensor *x,
                                   usize n,
                                   int offset,
                                   int dim = -1) noexcept;
    // Concatenate xa with xb, both xa and xb must have the same shape except for the dimension along which
    // the concat will happen.
    // Note: xa and xb are assumed to be contiguous.
    extern yami_tensor *yami_concat(yami_context *ctx,
                                    const yami_tensor *xa,
                                    const yami_tensor *xb,
                                    int dim = 0) noexcept;
    // Copy x into res. This is a very dangerous function!
    // Currently, there are no checks in place to verify whether the underlying
    // buffer of res is big enough for x. Also, after this call all the information
    // regarding the original size and shape of res will be lost.
    // Fixme: we need to add an object that describes the "backend" of a tensor,
    //  there we could store information such as the actual size of the underlying buffer
    //  as well as where it's actually stored (RAM, GPU, ...)
    extern void yami_copy(const yami_tensor *x,
                          yami_tensor *res) noexcept;
    // ========================================================================

    // =========================== Tensor Operations ==========================
    extern yami_tensor *yami_matmul(yami_context *ctx,
                                    const yami_tensor *xa,
                                    const yami_tensor *xb,
                                    yami_tensor *res = nullptr) noexcept;
    extern yami_tensor *yami_add(yami_context *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_addc(yami_context *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_sub(yami_context *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_subc(yami_context *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_mul(yami_context *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_mulc(yami_context *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_div(yami_context *ctx,
                                 yami_tensor *xa,
                                 const yami_tensor *xb,
                                 bool in_place = false) noexcept;
    extern yami_tensor *yami_divc(yami_context *ctx,
                                  yami_tensor *x, f32 c,
                                  bool in_place = true) noexcept;
    // ========================================================================

    // ============================ Math Functions ============================
    extern yami_tensor *yami_gelu(yami_context *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_sum(yami_context *ctx,
                                 const yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_mean(yami_context *ctx,
                                  const yami_tensor *x,
                                  int dim) noexcept;
    extern yami_tensor *yami_var(yami_context *ctx,
                                 yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_exp(yami_context *ctx,
                                 yami_tensor *x,
                                 bool in_place = true) noexcept;
    extern yami_tensor *yami_sqrt(yami_context *ctx,
                                  yami_tensor *x,
                                  bool in_place = true) noexcept;
    extern yami_tensor *yami_square(yami_context *ctx,
                                    yami_tensor *x,
                                    bool in_place = true) noexcept;
    extern yami_tensor *yami_max(yami_context *ctx,
                                 const yami_tensor *x,
                                 int dim) noexcept;
    extern yami_tensor *yami_softmax(yami_context *ctx,
                                     yami_tensor *x,
                                     int dim = -1) noexcept;
    extern yami_tensor *yami_layer_norm(yami_context *ctx,
                                        const yami_tensor *w,
                                        const yami_tensor *b,
                                        yami_tensor *x,
                                        bool in_place = true,
                                        f32 eps = 1e-5f) noexcept;
    // ========================================================================

#ifdef __cplusplus
}
#endif
