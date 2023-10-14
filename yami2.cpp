#include "yami2.h"
#include <cstring>

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

static yami_context *internal_ctx_init(size mem_size, void *mem_buffer) noexcept {
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

    yami_context *ctx = internal_ctx_init(params.mem_size, params.mem_buffer);
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->scratch = nullptr;

    if (params.scratch_mem_size > 0) {
        yami_context *scratch = internal_ctx_init(params.scratch_mem_size, params.scratch_mem_buffer);
        if (scratch == nullptr) {
            free(ctx);
            return nullptr;
        }

        ctx->scratch = scratch;
    }

    YAMI_LOG_DEBUG("created new context %p size=%ld alloc=%d scratch=%d",
                   (void *) ctx, params.mem_size, params.mem_buffer == nullptr,
                   params.scratch_mem_size > 0);

    return ctx;
}

void yami_free(yami_context *ctx) noexcept {
    YAMI_LOG_DEBUG("freeing context %p size=%ld is_owner=%d scratch=%d",
                   (void *) ctx, ctx->mem_size, ctx->is_own_memory, ctx->scratch != nullptr);

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
    YAMI_LOG_DEBUG("clearing context %p", (void *)ctx);

    YAMI_LOG_INFO("context mem_size=%.2lf KB, n_objs=%d", ctx->mem_size / 1024., ctx->n_objs);

    ctx->first = nullptr;
    ctx->last = nullptr;
    ctx->n_objs = 0;
}

void yami_mem_usage(const yami_context *ctx) noexcept {
    size used = ctx->last->offset + ctx->last->obj_size;
    YAMI_LOG_INFO("n_objs=%d used memory %.2lf KB out of %.2lf KB",
                  ctx->n_objs,
                  used / 1024.,
                  ctx->mem_size / 1024.
    );
}

static yami_tensor *yami_new_tensor(yami_context *ctx, const char *label,
                                    int n_dim, const size *dimensions) noexcept {
    YAMI_LOG_DEBUG("label=\"%s\" n_dim=%d [%ld, %ld, %ld, %ld]",
                   label, n_dim, dimensions[0], dimensions[1],
                   dimensions[2], dimensions[3]);

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
        YAMI_LOG_ERR("can't allocate %ld B", mem_needed);
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

static inline void tensor_get_extended_dim(const yami_tensor *xa, size *ext) noexcept {
    for (int i = 0; i < yami_max_dims; ++i) {
        ext[yami_max_dims - i - 1] = i >= xa->n_dim ? 1 : xa->dimensions[xa->n_dim - i - 1];
    }
}

static inline void internal_matmul(f32 *__restrict out, const f32 *__restrict xa,
                                   const f32 *__restrict xb,
                                   size n_rows_a, size n_cols_a, size n_cols_b) noexcept {
    for (int i = 0; i < n_rows_a; ++i) {
        for (int k = 0; k < n_cols_a; ++k) {
            for (int j = 0; j < n_cols_b; ++j) {
                out[i*n_cols_b + j] += xa[i*n_cols_a + k] * xb[k*n_cols_b + j];
            }
        }
    }
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

    // Number of elements of the 2D matrices
    const size xa_2_ne = xa_n_rows * xa_n_cols;
    const size xb_2_ne = xb_n_rows * xb_n_cols;
    const size res_2_ne = xa_n_rows * xb_n_cols;
    if (xa_n_cols != xb_n_rows) {
        YAMI_LOG_ERR("can't multiply (%ld, %ld) by (%ld, %ld)",
                     xa_n_rows, xa_n_cols,
                     xb_n_rows, xb_n_cols);

        return nullptr;
    }

    // Broadcast the other dimensions and allocate the result
    size extend_a[yami_max_dims], extend_b[yami_max_dims];

    tensor_get_extended_dim(xa, extend_a);
    tensor_get_extended_dim(xb, extend_b);

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);

    YAMI_LOG_DEBUG("extend_a = [%ld %ld %ld %ld]", extend_a[0], extend_a[1],
                   extend_a[2], extend_a[3]);
    YAMI_LOG_DEBUG("extend_b = [%ld %ld %ld %ld]", extend_b[0], extend_b[1],
                   extend_b[2], extend_b[3]);

    // In a matrix multiplication checking if two extended shapes can be broadcasted
    // is quite easy: just check whether dimensions[0] and dimensions[1] are either equal or 1
    bool can_broadcast = true;
    for (int i = 0; i < yami_max_dims - 2; ++i) {
        if (extend_a[i] == 1 || extend_b[i] == 1)
            continue;

        if (extend_a[i] != extend_b[i]) {
            can_broadcast = false;
            break;
        }
    }
    if (!can_broadcast) {
        YAMI_LOG_ERR("can't broadcast tensors \"%s\" and \"%s\"", xa->label, xb->label);
        return nullptr;
    }

    size res_dim[yami_max_dims];
    memset(res_dim, 0, yami_max_dims * sizeof(size));
    res_dim[res_n_dim - 2] = xa_n_rows;
    res_dim[res_n_dim - 1] = xb_n_cols;
    switch (res_n_dim) {
        case 2:
            break;
        case 3:
            res_dim[0] = YAMI_MAX(extend_a[1], extend_b[1]);
            break;
        case 4:
            res_dim[0] = YAMI_MAX(extend_a[0], extend_b[0]);
            res_dim[1] = YAMI_MAX(extend_a[1], extend_b[1]);
            break;
        default:
            YAMI_ASSERT(false);
    }

    yami_tensor *res = yami_new_tensor(ctx, "", res_n_dim, res_dim);
    size extend_res[yami_max_dims];
    tensor_get_extended_dim(res, extend_res);

    for (size d0 = 0; d0 < extend_res[0]; ++d0) {
        for (size d1 = 0; d1 < extend_res[1]; ++d1) {
            const size res_offset = d0 * extend_res[1] + d1;
            const size xa_offset = d1 % extend_a[1] + ((d0 % extend_a[0]) * extend_a[1]);
            const size xb_offset = d1 % extend_b[1] + ((d0 % extend_b[0]) * extend_b[1]);

            internal_matmul(&res->data[res_offset*res_2_ne],
                        &xa->data[xa_offset*xa_2_ne],
                        &xb->data[xb_offset*xb_2_ne],
                        xa_n_rows, xa_n_cols, xb_n_cols);
        }
    }

    return res;
}