#include "yami2.h"
#include <cstring>
#include <cmath>
#include <cstdarg>


struct yami_context {
    size mem_size;
    void *mem_buffer;

    yami_obj *first, *last;
    // Each context has his own internal context which can be used as a scratch buffer.
    yami_context *scratch;

    int n_objs;
    bool is_own_memory;
    u8 pad[3];
};

struct yami_obj {
    // add kind
    size offset;
    size obj_size;
    yami_obj *next;
};

constexpr static size yami_obj_size = sizeof(yami_obj);
constexpr static size yami_tensor_size = sizeof(yami_tensor);

// ============================================== Helper functions ==============================================
static inline void yami_set_tensor_dim(yami_tensor *x, const int n_dim, const size *dimensions) noexcept {
    x->n_dim = n_dim;

    memcpy(x->dimensions, dimensions, x->n_dim * sizeof(size));

    // Compute the extended dimension
    for (int i = 0; i < yami_max_dims; ++i) {
        x->extended_dim[yami_max_dims - i - 1] = i >= x->n_dim ? 1 : x->dimensions[x->n_dim - i - 1];
    }

    // Compute the stride
    memset(x->stride, 0, yami_max_dims * sizeof(size));
    for (int i = yami_max_dims - 1; i >= 0; --i) {
        x->stride[i] = i == yami_max_dims - 1 ? 1 : x->stride[i + 1] * x->extended_dim[i + 1];
    }
}

static inline size yami_tensor_offset(const yami_tensor *x, const size d0,
                                      const size d1 = 0, const size d2 = 0,
                                      const size d3 = 0) noexcept {
    return (d0 % x->extended_dim[0]) * x->stride[0] + (d1 % x->extended_dim[1]) * x->stride[1] +
           (d2 % x->extended_dim[2]) * x->stride[2] + (d3 % x->extended_dim[3]) * x->stride[3];
}

// Returns the 'real' dimension given a number which can be either positive or negative,
// dim=-1 means 'last dimension', dim=-2 'last but one' ecc...
static inline int yami_tensor_get_dim(const yami_tensor *x, const int dim) noexcept {
    // dim=-1 means 'last dimension', dim=-2 'last but one' ecc...
    const int real_dim = dim < 0 ? x->n_dim + dim : dim;

    YAMI_ASSERT(real_dim >= 0 && real_dim < x->n_dim);
    return real_dim;
}

static inline bool shape_matches(const yami_tensor *x, const int n_dim, const size *dims) noexcept {
    bool match = x->n_dim == n_dim;
    for (int i = 0; i < n_dim && match; ++i) {
        match &= x->dimensions[i] == dims[i];
    }
    return match;
}

static inline bool yami_can_broadcast(const yami_tensor *xa, const yami_tensor *xb, const int dims_to_check = yami_max_dims) noexcept {
    bool can_broadcast = true;
    for (int i = 0; i < dims_to_check && can_broadcast; ++i) {
        if (xa->extended_dim[i] == xb->extended_dim[i] || xa->extended_dim[i] == 1 || xb->extended_dim[i] == 1)
            continue;

        can_broadcast = false;
    }

    if (!can_broadcast) {
        YAMI_LOG_ERR("can't broadcast tensors \"%s\" and \"%s\"", xa->label, xb->label);
    }

    return can_broadcast;
}

static inline yami_tensor *yami_alloc_result(yami_context *ctx, const yami_tensor *xa,
                                             const yami_tensor *xb, const char *label = "") noexcept {

    YAMI_ASSERT(yami_can_broadcast(xa, xb));

    size extend_res[yami_max_dims];
    for (int i = 0; i < yami_max_dims; ++i) {
        extend_res[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);
    }

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);

    yami_tensor *res = yami_new_tensor(ctx, res_n_dim, &extend_res[yami_max_dims - res_n_dim], label);
    return res;
}
// ==============================================================================================================

static yami_context *yami_internal_ctx_init(size mem_size, void *mem_buffer) noexcept {
    constexpr static usize ctx_size = sizeof(yami_context);

    yami_context *ctx = (yami_context *) malloc(ctx_size);
    if (ctx == nullptr) {
        YAMI_LOG_ERR("error allocating context");
        return nullptr;
    }
    ctx->mem_size = mem_size;
    ctx->first = nullptr;
    ctx->last = nullptr;
    ctx->is_own_memory = mem_buffer == nullptr;
    ctx->n_objs = 0;

    if (!ctx->is_own_memory) {
        ctx->mem_buffer = mem_buffer;
    } else {
        ctx->mem_buffer = malloc(ctx->mem_size);
        if (ctx->mem_buffer == nullptr) {
            YAMI_LOG_ERR("error allocating context memory buffer");
            free(ctx);
            return nullptr;
        }
    }

    return ctx;
}

yami_context *yami_init(yami_context_init_params params) noexcept {
    if (params.mem_size <= 0) {
        YAMI_LOG_ERR("invalid memory size %ld", params.mem_size);
        return nullptr;
    }

    yami_context *ctx = yami_internal_ctx_init(params.mem_size, params.mem_buffer);
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->scratch = nullptr;

    if (params.scratch_mem_size > 0) {
        yami_context *scratch = yami_internal_ctx_init(params.scratch_mem_size, params.scratch_mem_buffer);
        if (scratch == nullptr) {
            free(ctx);
            return nullptr;
        }

        ctx->scratch = scratch;
    }

    YAMI_LOG_DEBUG("created new context %p size=%ld MB alloc=%d has_scratch=%d",
                   (void *) ctx, (size) YAMI_B_TO_MB(params.mem_size), params.mem_buffer == nullptr,
                   params.scratch_mem_size > 0);

    return ctx;
}

void yami_free(yami_context *ctx) noexcept {
    YAMI_LOG_DEBUG("freeing context %p size=%.2f MB is_owner=%d has_scratch=%d",
                   (void *) ctx, YAMI_B_TO_MB(ctx->mem_size), ctx->is_own_memory, ctx->scratch != nullptr);

    if (ctx->scratch != nullptr) {
        if (ctx->scratch->is_own_memory)
            free(ctx->scratch->mem_buffer);

        free(ctx->scratch);
    }

    if (ctx->is_own_memory)
        free(ctx->mem_buffer);

    free(ctx);
}

void yami_clear_ctx(yami_context *ctx) noexcept {
    YAMI_LOG_INFO("clearing context %p mem_size=%ld MB n_objs=%d",
                  (void *)ctx, (size) YAMI_B_TO_MB(ctx->mem_size), ctx->n_objs);

    ctx->first = nullptr;
    ctx->last = nullptr;
    ctx->n_objs = 0;
}

yami_context *yami_ctx_scratch(yami_context *ctx) noexcept {
    YAMI_ASSERT(ctx->scratch != nullptr);

    return ctx->scratch;
}

void yami_mem_usage(const yami_context *ctx) noexcept {
    size used = ctx->last->offset + ctx->last->obj_size;
    YAMI_LOG_INFO("n_objs=%d used memory %.2f MB out of %ld MB (%.2f%%)",
                  ctx->n_objs,
                  YAMI_B_TO_MB(used),
                  (size) YAMI_B_TO_MB(ctx->mem_size),
                  ((f64) (used) / (f64) (ctx->mem_size)) * 100.
    );
}

yami_tensor *yami_new_tensor(yami_context *ctx, const int n_dim,
                             const size *dimensions, const char *label) noexcept {

    if (n_dim > yami_max_dims || n_dim <= 0) {
        YAMI_LOG_ERR("%d is not a valid number of dimensions", n_dim);
        return nullptr;
    }

    if (strlen(label) > yami_max_label) {
        YAMI_LOG_ERR("label \"%s\" is too long", label);
        return nullptr;
    }

    size ne = 1;
    for (int i = 0; i < n_dim; ++i)
        ne *= dimensions[i];

    if (ne <= 0) {
        YAMI_LOG_ERR("can't allocate %ld elements", ne);
        return nullptr;
    }

    size mem_needed = ne * sizeof(f32) + yami_tensor_size;

    size last_offset = ctx->last == nullptr ? 0 : ctx->last->offset;
    size last_size = ctx->last == nullptr ? 0 : ctx->last->obj_size;
    size last_end = last_offset + last_size;

    if (mem_needed + yami_obj_size + last_end > ctx->mem_size) {
        YAMI_LOG_ERR("can't allocate %.2f MB", YAMI_B_TO_MB(mem_needed));
        return nullptr;
    }

    yami_obj *new_obj = (yami_obj *) ((byte *)ctx->mem_buffer + last_end);
    // The offset refers to the actual offset of the "contained" object which comes after
    // the yami_object "header".
    new_obj->offset = last_end + yami_obj_size;
    new_obj->obj_size = mem_needed;
    new_obj->next = nullptr;

    if (ctx->last == nullptr) {
        ctx->first = new_obj;
    } else {
        ctx->last->next = new_obj;
    }

    ctx->last = new_obj;
    ctx->n_objs++;
    // Allocate the actual tensor
    yami_tensor *new_tensor = (yami_tensor *) ((byte *)ctx->mem_buffer + new_obj->offset);

    new_tensor->ne = ne;
    new_tensor->data = (f32 *) (new_tensor + 1);
    memset(new_tensor->data, 0, ne * sizeof(f32));

    strncpy(new_tensor->label, label, yami_max_label);

    yami_set_tensor_dim(new_tensor, n_dim, dimensions),

            YAMI_LOG_DEBUG("label=\"%s\" n_dim=%d extended_dim=[%ld, %ld, %ld, %ld] stride=[%ld, %ld, %ld, %ld]",
                           label, n_dim, new_tensor->extended_dim[0], new_tensor->extended_dim[1], new_tensor->extended_dim[2],
                           new_tensor->extended_dim[3], new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2],
                           new_tensor->stride[3]);

    return new_tensor;
}

yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
                            size dim1) noexcept{
    size dims[yami_max_dims] = {dim1};
    return yami_new_tensor(ctx, 1, dims, label);
}

yami_tensor *yami_tensor_2d(yami_context *ctx, const char *label,
                            size dim1, size dim2) noexcept{
    size dims[yami_max_dims] = {dim1, dim2};
    return yami_new_tensor(ctx, 2, dims, label);
}

yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
                            size dim1, size dim2,
                            size dim3) noexcept{
    size dims[yami_max_dims] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, 3, dims, label);
}

yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
                            size dim1, size dim2,
                            size dim3, size dim4) noexcept{
    size dims[yami_max_dims] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, 4, dims, label);
}

yami_tensor *yami_clone(yami_context *ctx, const yami_tensor *x) noexcept{
    yami_tensor *clone = yami_new_tensor(ctx, x->n_dim, x->dimensions, x->label);
    if (clone == nullptr) {
        YAMI_LOG_ERR("error cloning %s", x->label);
        return nullptr;
    }

    memcpy(clone->data, x->data, x->ne * sizeof(f32));
    return clone;
}

void yami_reshape(yami_tensor *x, const int n_dims...) noexcept {
    YAMI_ASSERT(n_dims > 0 && n_dims < yami_max_dims);

    size new_dimensions[yami_max_dims];

    std::va_list args;
    va_start(args, n_dims);
    for (int i = 0; i < n_dims; ++i) {
        const size d = va_arg(args, size);
        YAMI_ASSERT(d > 0);
        new_dimensions[i] = d;
    }
    va_end(args);


    size new_ne = 1;
    for (int i = 0; i < n_dims; ++i)
        new_ne *= new_dimensions[i];

    YAMI_ASSERT(new_ne == x->ne);

    yami_set_tensor_dim(x, n_dims, new_dimensions);
}

yami_tensor *yami_transpose(yami_context *, yami_tensor *x,
                            const int dim1, const int dim2) noexcept {
    YAMI_ASSERT(x->n_dim > 1);

    const size real_d1 = yami_tensor_get_dim(x, dim1);
    const size real_d2 = yami_tensor_get_dim(x, dim2);

    YAMI_ASSERT(real_d1 >= 0 && real_d1 < yami_max_dims);
    YAMI_ASSERT(real_d2 >= 0 && real_d2 < yami_max_dims);

    const size ext_d1 = x->n_dim == yami_max_dims ? real_d1 : (real_d1 + (yami_max_dims - x->n_dim));
    const size ext_d2 = x->n_dim == yami_max_dims ? real_d2 : (real_d2 + (yami_max_dims - x->n_dim));

    const size tmp_d1 = x->dimensions[real_d1];
    x->dimensions[real_d1] = x->dimensions[real_d2];
    x->dimensions[real_d2] = tmp_d1;

    const size tmp_ed1 = x->extended_dim[ext_d1];
    x->extended_dim[ext_d1] = x->extended_dim[ext_d2];
    x->extended_dim[ext_d2] = tmp_ed1;

    const size tmp_s1 = x->stride[ext_d1];
    x->stride[ext_d1] = x->stride[ext_d2];
    x->stride[ext_d2] = tmp_s1;

    return x;
}

yami_tensor *yami_lt_mask(yami_context *ctx, yami_tensor *x,
                          const f32 mask, const bool in_place) noexcept {
    YAMI_ASSERT(x->n_dim >= 2);

    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (size d0 = 0; d0 < x->extended_dim[0]; ++d0){
        for (size d1 = 0; d1 < x->extended_dim[1]; ++d1){
            for (size d2 = 0; d2 < x->extended_dim[2]; ++d2){
                for (size d3 = 0; d3 < x->extended_dim[3]; ++d3) {
                    const size idx = yami_tensor_offset(x, d0, d1, d2, d3);
                    res->data[idx] = d3 <= d2 ? x->data[idx] : mask;
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_embed(yami_context *ctx, const yami_tensor *x,
                        const int *indexes, size n) noexcept {
    YAMI_ASSERT(x->n_dim == 2);

    const size max_idx = x->dimensions[yami_tensor_get_dim(x, -2)];
    const size emb_size = x->dimensions[yami_tensor_get_dim(x, -1)];
    size res_dim[2] = {n, emb_size};

    yami_tensor *res = yami_new_tensor(ctx, 2, res_dim);
    for (size i = 0; i < n; ++i) {
        const int idx = indexes[i];
        YAMI_ASSERT(idx < max_idx);

        // todo: probably offset should be relative to the real index not the extended one
        const size x_offset = yami_tensor_offset(x, 0, 0, idx);
        const size res_offset = yami_tensor_offset(res,0, 0, i);
        memcpy(&res->data[res_offset], &x->data[x_offset], emb_size * sizeof(f32));
    }

    return res;
}

static inline void yami_internal_matmul(f32 *__restrict out, const f32 *__restrict xa,
                                        const f32 *__restrict xb,
                                        const size n_rows_a, const size n_cols_a, const size n_cols_b) noexcept {
    for (int i = 0; i < n_rows_a; ++i) {
        for (int k = 0; k < n_cols_a; ++k) {
            for (int j = 0; j < n_cols_b; ++j) {
                out[i*n_cols_b + j] += xa[i*n_cols_a + k] * xb[k*n_cols_b + j];
            }
        }
    }
}

yami_tensor *yami_matmul(yami_context *ctx, const yami_tensor *xa,
                         const yami_tensor *xb, yami_tensor *res) noexcept {
    // Verify that the two matrices are at least 2-dimensional
    if (xa->n_dim < 2 || xb->n_dim < 2) {
        YAMI_LOG_ERR("too few dimensions, use yami_mul for 1D tensor multiply");
        YAMI_ASSERT(false);
    }

    const size xa_n_rows = xa->dimensions[yami_tensor_get_dim(xa, -2)];
    const size xa_n_cols = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const size xb_n_rows = xb->dimensions[yami_tensor_get_dim(xb, -2)];
    const size xb_n_cols = xb->dimensions[yami_tensor_get_dim(xb, -1)];

    if (xa_n_cols != xb_n_rows) {
        YAMI_LOG_ERR("can't multiply (%ld, %ld) by (%ld, %ld)",
                     xa_n_rows, xa_n_cols,
                     xb_n_rows, xb_n_cols);
        YAMI_ASSERT(false);
    }

    YAMI_ASSERT(yami_can_broadcast(xa, xb, yami_max_dims - 2));

    size res_dim[yami_max_dims];
    res_dim[yami_max_dims - 2] = xa_n_rows;
    res_dim[yami_max_dims - 1] = xb_n_cols;
    for (int i = 0; i < yami_max_dims - 2; ++i)
        res_dim[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);
    res = res == nullptr ? yami_new_tensor(ctx, res_n_dim, &res_dim[yami_max_dims - res_n_dim]) : res;
    YAMI_ASSERT(shape_matches(res, res_n_dim, &res_dim[yami_max_dims - res_n_dim]));

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            const size res_offset = yami_tensor_offset(res, d0, d1);
            const size xa_offset = yami_tensor_offset(xa, d0, d1);
            const size xb_offset = yami_tensor_offset(xb, d0, d1);
            yami_internal_matmul(&res->data[res_offset],
                                 &xa->data[xa_offset],
                                 &xb->data[xb_offset],
                                 xa_n_rows, xa_n_cols, xb_n_cols);
        }
    }

    return res;
}

static inline void yami_internal_vec_add(f32 *__restrict out, const f32 *xa,
                                         const f32 *xb, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] + xb[i];
}

static inline void yami_internal_vec_addc(f32 *out, const f32 *x,
                                          const f32 c, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = x[i] + c;
}

// We have to possibilities here:
//  - the last dim is equal                                 --> sum over two equal tensors with size N
//  - the last dim of one tensor is 1 and the other is > 1  --> sum a constant to a tensor
yami_tensor *yami_add(yami_context *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const size xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const size xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < res->extended_dim[2]; ++d2) {
                const size res_offset = yami_tensor_offset(res, d0, d1, d2);
                const size xa_offset = yami_tensor_offset(xa, d0, d1, d2);
                const size xb_offset = yami_tensor_offset(xb, d0, d1, d2);

                if (scalar) {
                    const f32 c = xa_1_ne == 1 ? xa->data[xa_offset] : xb->data[xb_offset];
                    const f32 *data = xa_1_ne != 1 ? &xa->data[xa_offset] : &xb->data[xb_offset];

                    yami_internal_vec_addc(&res->data[res_offset],
                                           data, c, n);
                } else {
                    yami_internal_vec_add(&res->data[res_offset],
                                          &xa->data[xa_offset],
                                          &xb->data[xb_offset],
                                          n);
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_addc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_addc(res->data, x->data, c, x->ne);
    return res;
}

static inline void yami_internal_vec_sub(f32 *out, const f32 *xa,
                                         const f32 *xb, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] - xb[i];
}

static inline void yami_internal_vec_subc(f32 *out, const f32 *x,
                                          const f32 c, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = x[i] - c;
}

yami_tensor *yami_sub(yami_context *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place){
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const size xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const size xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < res->extended_dim[2]; ++d2) {
                const size res_offset = yami_tensor_offset(res, d0, d1, d2);
                const size xa_offset = yami_tensor_offset(xa, d0, d1, d2);
                const size xb_offset = yami_tensor_offset(xb, d0, d1, d2);

                if (scalar) {
                    const f32 c = xa_1_ne == 1 ? xa->data[xa_offset] : xb->data[xb_offset];
                    const f32 *data = xa_1_ne != 1 ? &xa->data[xa_offset] : &xb->data[xb_offset];

                    yami_internal_vec_subc(&res->data[res_offset],
                                           data, c, n);
                } else {
                    yami_internal_vec_sub(&res->data[res_offset],
                                          &xa->data[xa_offset],
                                          &xb->data[xb_offset],
                                          n);
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_subc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_subc(res->data, x->data, c, x->ne);
    return res;
}

static inline void yami_internal_vec_mul(f32 *out, const f32 *xa,
                                         const f32 *xb, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] * xb[i];
}

static inline void yami_internal_vec_mulc(f32 *out, const f32 *x,
                                          const f32 c, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = x[i] * c;
}

extern yami_tensor *yami_mul(yami_context *ctx, yami_tensor *xa,
                             const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const size xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const size xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < res->extended_dim[2]; ++d2) {
                const size res_offset = yami_tensor_offset(res, d0, d1, d2);
                const size xa_offset = yami_tensor_offset(xa, d0, d1, d2);
                const size xb_offset = yami_tensor_offset(xb, d0, d1, d2);

                if (scalar) {
                    const f32 c = xa_1_ne == 1 ? xa->data[xa_offset] : xb->data[xb_offset];
                    const f32 *data = xa_1_ne != 1 ? &xa->data[xa_offset] : &xb->data[xb_offset];

                    yami_internal_vec_mulc(&res->data[res_offset],
                                           data, c, n);
                } else {
                    yami_internal_vec_mul(&res->data[res_offset],
                                          &xa->data[xa_offset],
                                          &xb->data[xb_offset],
                                          n);
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_mulc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_mulc(res->data, x->data, c, res->ne);
    return res;
}

static inline void yami_internal_vec_div(f32 *out, const f32 *xa,
                                         const f32 *xb, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] / xb[i];
}

static inline void yami_internal_vec_divc(f32 *out, const f32 *x,
                                          const f32 c, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = x[i] / c;
}

static inline void yami_internal_c_vec_div(f32 *out, const f32 c,
                                           const f32 *x, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = c / x[i];
}

extern yami_tensor *yami_div(yami_context *ctx, yami_tensor *xa,
                             const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const size xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const size xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < res->extended_dim[2]; ++d2) {
                const size res_offset = yami_tensor_offset(res, d0, d1, d2);
                const size xa_offset = yami_tensor_offset(xa, d0, d1, d2);
                const size xb_offset = yami_tensor_offset(xb, d0, d1, d2);

                if (scalar) {
                    // fixme: this is a very very very ugly workaround...
                    if (xa_1_ne == 1)
                        yami_internal_c_vec_div(&res->data[res_offset],
                                                xa->data[xa_offset],
                                                &xb->data[xb_offset], n);
                    else
                        yami_internal_vec_divc(&res->data[res_offset],
                                               &xa->data[xa_offset],
                                               xb->data[xb_offset], n);

                } else {
                    yami_internal_vec_div(&res->data[res_offset],
                                          &xa->data[xa_offset],
                                          &xb->data[xb_offset],
                                          n);
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_divc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_divc(res->data, x->data, c, res->ne);
    return res;
}

static inline f32 yami_internal_tanh_f32(const f32 x) noexcept {
    const f32 e_p = std::exp(x);
    const f32 e_n = std::exp(-x);
    return (e_p - e_n) / (e_p + e_n);
}

static inline void yami_internal_tanh(f32 *out, const f32 *x, const size n) noexcept {
    for (size i = 0; i < n; ++i) {
        out[i] = yami_internal_tanh_f32(x[i]);
    }
}

yami_tensor *yami_tanh(yami_context *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_tanh(res->data, x->data, x->ne);
    return res;
}

static inline void yami_internal_gelu(f32 *out, const f32 *x, const size n) noexcept {
    const static f32 c1 = std::sqrt(M_2_PIf);
    constexpr static f32 c2 = 0.044715f;

    for (size i = 0; i < n; ++i) {
        const f32 x_i = x[i];
        out[i] = (0.5f * x_i) * (1.f + yami_internal_tanh_f32(c1 * (x_i + c2 * x_i * x_i * x_i)));
    }
}

yami_tensor *yami_gelu(yami_context *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_gelu(res->data, x->data, x->ne);
    return res;
}

yami_tensor *yami_sum(yami_context *ctx, const yami_tensor *x, const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    size res_dim[yami_max_dims];
    memcpy(res_dim, x->dimensions, x->n_dim * sizeof(size));
    res_dim[real_dim] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, res_dim);

    const size dim_ext_idx = yami_max_dims - x->n_dim + real_dim;

    size out_idx;
    for (size d0 = 0; d0 < x->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < x->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < x->extended_dim[2]; ++d2) {
                for (size d3 = 0; d3 < x->extended_dim[3]; ++d3) {
                    const size in_idx = yami_tensor_offset(x, d0, d1, d2, d3);

                    if (dim_ext_idx == 0) out_idx = yami_tensor_offset(res, 0, d1, d2, d3);
                    else if (dim_ext_idx == 1) out_idx = yami_tensor_offset(res, d0, 0, d2, d3);
                    else if (dim_ext_idx == 2) out_idx = yami_tensor_offset(res, d0, d1, 0, d3);
                    else out_idx = yami_tensor_offset(res, d0, d1, d2, 0);

                    res->data[out_idx] += x->data[in_idx];
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_mean(yami_context *ctx, const yami_tensor *x, const int dim) noexcept {
    yami_tensor *res = yami_sum(ctx, x, dim);

    yami_mulc(ctx, res, 1.f / (f32) x->dimensions[yami_tensor_get_dim(x, dim)]);

    return res;
}

yami_tensor *yami_var(yami_context *ctx, const yami_tensor *x, const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    yami_tensor *res = yami_mean(ctx, x, dim);

    const size dim_ext_idx = yami_max_dims - x->n_dim + real_dim;

    size out_idx;
    for (size d0 = 0; d0 < x->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < x->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < x->extended_dim[2]; ++d2) {
                for (size d3 = 0; d3 < x->extended_dim[3]; ++d3) {
                    const size in_idx = yami_tensor_offset(x, d0, d1, d2, d3);
                    if (dim_ext_idx == 0) out_idx = yami_tensor_offset(res, 0, d1, d2, d3);
                    else if (dim_ext_idx == 1) out_idx = yami_tensor_offset(res, d0, 0, d2, d3);
                    else if (dim_ext_idx == 2) out_idx = yami_tensor_offset(res, d0, d1, 0, d3);
                    else out_idx = yami_tensor_offset(res, d0, d1, d2, 0);

                    res->data[out_idx] += ((x->data[in_idx] - res->data[out_idx]) *
                                           (x->data[in_idx] - res->data[out_idx]));
                }
            }
        }
    }

    yami_mulc(ctx, res, 1.f / (f32) x->dimensions[real_dim]);

    return res;
}

yami_tensor *yami_exp(yami_context *ctx, yami_tensor *x, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (size i = 0; i < x->ne; ++i)
        res->data[i] = std::exp(x->data[i]);

    return res;
}

yami_tensor *yami_sqrt(yami_context *ctx, yami_tensor *x, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (size i = 0; i < x->ne; ++i)
        res->data[i] = std::sqrt(x->data[i]);

    return res;
}

yami_tensor *yami_max(yami_context *ctx, const yami_tensor *x, const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    size res_dim[yami_max_dims];
    memcpy(res_dim, x->dimensions, x->n_dim * sizeof(size));
    res_dim[real_dim] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, res_dim);

    for (size i = 0; i < res->ne; ++i)
        res->data[i] = yami_neg_inf;

    const size dim_ext_idx = yami_max_dims - x->n_dim + real_dim;

    size out_idx;
    for (size d0 = 0; d0 < x->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < x->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < x->extended_dim[2]; ++d2) {
                for (size d3 = 0; d3 < x->extended_dim[3]; ++d3) {
                    const size in_idx = yami_tensor_offset(x, d0, d1, d2, d3);
                    if (dim_ext_idx == 0) out_idx = yami_tensor_offset(res, 0, d1, d2, d3);
                    else if (dim_ext_idx == 1) out_idx = yami_tensor_offset(res, d0, 0, d2, d3);
                    else if (dim_ext_idx == 2) out_idx = yami_tensor_offset(res, d0, d1, 0, d3);
                    else out_idx = yami_tensor_offset(res, d0, d1, d2, 0);

                    res->data[out_idx] = YAMI_MAX(res->data[out_idx], x->data[in_idx]);
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_softmax(yami_context *ctx, yami_tensor *x,
                          const int dim, const bool in_place) noexcept {
    // fixme: there are two allocations that should be freed after the function return
    yami_tensor *e_x = yami_exp(ctx, yami_sub(ctx, x, yami_max(ctx, x, dim), in_place));
    return yami_div(ctx, e_x, yami_sum(ctx, e_x, dim), in_place);
}

yami_tensor *yami_layer_norm(yami_context *ctx, const yami_tensor *w, const yami_tensor *b,
                             yami_tensor *x, const bool in_place, const f32 eps) noexcept {
    // fixme: same as before, we have two allocations that should be 'temporary'
    yami_tensor *x_mean = yami_mean(ctx, x, -1);
    yami_tensor *x_var = yami_var(ctx, x, -1);

    yami_addc(ctx, x_var, eps);

    yami_tensor *out = yami_div(ctx,
                                yami_sub(ctx, x, x_mean, in_place),
                                yami_sqrt(ctx, x_var),
                                in_place);

    return yami_add(ctx, yami_mul(ctx, out, w, in_place), b, in_place);
}
