#include "yami2.h"
#include <cstring>

#define YAMI_B_TO_KB(b) ((f64)(b) / 1024.)
#define YAMI_B_TO_MB(b) ((f64)(b) / (1024. * 1024.))

struct yami_context {
    size mem_size;
    void *mem_buffer;

    int n_objs;

    yami_obj *first, *last;
    // Each context has his own internal context which can be used as a scratch buffer.
    yami_context *scratch;
    bool is_own_memory;
};

struct yami_obj {
    // add kind
    size offset;
    size obj_size;
    yami_obj *next;
};

constexpr static size yami_obj_size = sizeof(yami_obj);
constexpr static size yami_tensor_size = sizeof(yami_tensor);


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

    YAMI_LOG_DEBUG("created new context %p size=%.2f alloc=%d has_scratch=%d",
                   (void *) ctx, YAMI_B_TO_MB(params.mem_size), params.mem_buffer == nullptr,
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
    YAMI_LOG_INFO("clearing context %p mem_size=%.2f MB n_objs=%d",
                  (void *)ctx, YAMI_B_TO_MB(ctx->mem_size), ctx->n_objs);

    ctx->first = nullptr;
    ctx->last = nullptr;
    ctx->n_objs = 0;
}

void yami_mem_usage(const yami_context *ctx) noexcept {
    size used = ctx->last->offset + ctx->last->obj_size;
    YAMI_LOG_INFO("n_objs=%d used memory %.2f MB out of %.2f MB (%.2f%%)",
                  ctx->n_objs,
                  YAMI_B_TO_MB(used),
                  YAMI_B_TO_MB(ctx->mem_size),
                  ((f64) (used) / (f64) (ctx->mem_size)) * 100.
    );
}

static yami_tensor *yami_new_tensor(yami_context *ctx, const char *label,
                                    int n_dim, const size *dimensions) noexcept {

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
    new_tensor->n_dim = n_dim;
    new_tensor->data = (f32 *) (new_tensor + 1);
    memset(new_tensor->data, 0, ne * sizeof(f32));

    memcpy(new_tensor->dimensions, dimensions, n_dim * sizeof(size));
    strncpy(new_tensor->label, label, yami_max_label);

    // Compute the extended dimension
    for (int i = 0; i < yami_max_dims; ++i) {
        new_tensor->extended_dim[yami_max_dims - i - 1] = i >= n_dim ? 1 : new_tensor->dimensions[n_dim - i - 1];
    }

    // Compute the stride
    memset(new_tensor->stride, 0, yami_max_dims * sizeof(size));
    for (int i = yami_max_dims - 1; i >= 0; --i) {
        new_tensor->stride[i] = i == yami_max_dims - 1 ? 1 : new_tensor->stride[i + 1] * new_tensor->extended_dim[i + 1];
    }

    YAMI_LOG_DEBUG("label=\"%s\" n_dim=%d extended_dim=[%ld, %ld, %ld, %ld] stride=[%ld, %ld, %ld, %ld]",
                   label, n_dim, new_tensor->extended_dim[0], new_tensor->extended_dim[1], new_tensor->extended_dim[2],
                   new_tensor->extended_dim[3], new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2],
                   new_tensor->stride[3]);

    return new_tensor;
}

yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
                            size dim1) noexcept{
    size dims[yami_max_dims] = {dim1};
    return yami_new_tensor(ctx, label, 1, dims);
}

yami_tensor *yami_tensor_2d(yami_context *ctx, const char *label,
                            size dim1, size dim2) noexcept{
    size dims[yami_max_dims] = {dim1, dim2};
    return yami_new_tensor(ctx, label, 2, dims);
}

yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
                            size dim1, size dim2,
                            size dim3) noexcept{
    size dims[yami_max_dims] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, label, 3, dims);
}

yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
                            size dim1, size dim2,
                            size dim3, size dim4) noexcept{
    size dims[yami_max_dims] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, label, 4, dims);
}

yami_tensor *yami_clone(yami_context *ctx, const yami_tensor *x) noexcept{
    yami_tensor *clone = yami_new_tensor(ctx, x->label, x->n_dim, x->dimensions);
    if (clone == nullptr) {
        YAMI_LOG_ERR("error cloning %s", x->label);
        return nullptr;
    }

    memcpy(clone->data, x->data, x->ne * sizeof(f32));
    return clone;
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

static inline bool yami_can_broadcast(const yami_tensor *xa, const yami_tensor *xb, const int dims_to_check) noexcept {
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

yami_tensor *yami_matmul(yami_context *ctx,
                         const yami_tensor *xa,
                         const yami_tensor *xb) noexcept {
    // Verify that the two matrices are at least 2-dimensional
    if (xa->n_dim < 2 || xb->n_dim < 2) {
        YAMI_LOG_ERR("too few dimensions, use yami_mul for 1D tensor multiply");
        return nullptr;
    }

    const size xa_n_rows = xa->dimensions[xa->n_dim - 2];
    const size xa_n_cols = xa->dimensions[xa->n_dim - 1];
    const size xb_n_rows = xb->dimensions[xb->n_dim - 2];
    const size xb_n_cols = xb->dimensions[xb->n_dim - 1];

    if (xa_n_cols != xb_n_rows) {
        YAMI_LOG_ERR("can't multiply (%ld, %ld) by (%ld, %ld)",
                     xa_n_rows, xa_n_cols,
                     xb_n_rows, xb_n_cols);

        return nullptr;
    }

    // In a matrix multiplication checking if two extended shapes can be broadcast
    // is quite easy: just check whether dimensions[0] and dimensions[1] are either equal or 1
    if (!yami_can_broadcast(xa, xb, yami_max_dims - 2)) {
        return nullptr;
    }

    size res_dim[yami_max_dims];
    memset(res_dim, 0, yami_max_dims * sizeof(size));
    res_dim[yami_max_dims - 2] = xa_n_rows;
    res_dim[yami_max_dims - 1] = xb_n_cols;
    for (int i = 0; i < yami_max_dims - 2; ++i)
        res_dim[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);
    yami_tensor *res = yami_new_tensor(ctx, "", res_n_dim, &res_dim[yami_max_dims - res_n_dim]);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            const size res_offset = d1 * res->stride[1] + d0 * res->stride[0];
            const size xa_offset = (d1 % xa->extended_dim[1]) * xa->stride[1] + (d0 % xa->extended_dim[0]) * xa->stride[0];
            const size xb_offset = (d1 % xb->extended_dim[1]) * xb->stride[1] + (d0 % xb->extended_dim[0]) * xb->stride[0];
            yami_internal_matmul(&res->data[res_offset],
                                 &xa->data[xa_offset],
                                 &xb->data[xb_offset],
                                 xa_n_rows, xa_n_cols, xb_n_cols);
        }
    }

    return res;
}

static inline void yami_internal_vec_add(f32 *__restrict out, const f32 *__restrict xa,
                                         const f32 *__restrict xb, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] + xb[i];
}

static inline void yami_internal_vec_addc(f32 *__restrict out, const f32 *__restrict xa,
                                          const f32 c, const size n) noexcept {
    for (size i = 0; i < n; ++i)
        out[i] = xa[i] + c;
}

// We have to possibilities here:
//  - the last dim is equal                                 --> sum over two equal tensors with size N
//  - the last dim of one tensor is 1 and the other is > 1  --> sum a constant to a tensor
yami_tensor *yami_add(yami_context *ctx,
                      const yami_tensor *xa,
                      const yami_tensor *xb) noexcept {
    if (!yami_can_broadcast(xa, xb, yami_max_dims)) {
        return nullptr;
    }

    size extend_res[yami_max_dims];
    for (int i = 0; i < yami_max_dims; ++i) {
        extend_res[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);
    }

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);

    yami_tensor *res = yami_new_tensor(ctx, "", res_n_dim, &extend_res[yami_max_dims - res_n_dim]);

    const size xa_1_ne = xa->dimensions[xa->n_dim - 1];
    const size xb_1_ne = xb->dimensions[xb->n_dim - 1];
    const bool addc = xa_1_ne == 1 || xb_1_ne == 1;
    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    for (size d0 = 0; d0 < res->extended_dim[0]; ++d0) {
        for (size d1 = 0; d1 < res->extended_dim[1]; ++d1) {
            for (size d2 = 0; d2 < res->extended_dim[2]; ++d2) {
                const size res_offset = d2 * res->stride[2] + d1 * res->stride[1] + d0 * res->stride[0];
                const size xa_offset = (d2 % xa->extended_dim[2]) * xa->stride[2] + (d1 % xa->extended_dim[1]) * xa->stride[1]
                                       + (d0 % xa->extended_dim[0]) * xa->stride[0];
                const size xb_offset = (d2 % xb->extended_dim[2]) * xb->stride[2] + (d1 % xb->extended_dim[1]) * xb->stride[1]
                                       + (d0 % xb->extended_dim[0]) * xb->stride[0];

                if (addc) {
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