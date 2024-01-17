#include "yami.h"
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <atomic>

#ifdef __AVX2__
#   include <immintrin.h>
#endif


#define YAMI_TENSOR_DIMS_0(PTR) const usize d0_##PTR = PTR->extended_dim[0]

#define YAMI_TENSOR_DIMS_1(PTR) YAMI_TENSOR_DIMS_0(PTR); \
                                const usize d1_##PTR = PTR->extended_dim[1]

#define YAMI_TENSOR_DIMS_2(PTR) YAMI_TENSOR_DIMS_1(PTR); \
                                const usize d2_##PTR = PTR->extended_dim[2]

#define YAMI_TENSOR_DIMS_3(PTR) YAMI_TENSOR_DIMS_2(PTR); \
                                const usize d3_##PTR = PTR->extended_dim[3]

#define YAMI_TENSOR_STRIDES_0(PTR)  const usize d0_stride_##PTR = PTR->stride[0]

#define YAMI_TENSOR_STRIDES_1(PTR)  YAMI_TENSOR_STRIDES_0(PTR); \
                                    const usize d1_stride_##PTR = PTR->stride[1]

#define YAMI_TENSOR_STRIDES_2(PTR)  YAMI_TENSOR_STRIDES_1(PTR); \
                                    const usize d2_stride_##PTR = PTR->stride[2]

#define YAMI_TENSOR_STRIDES_3(PTR)  YAMI_TENSOR_STRIDES_2(PTR); \
                                    const usize d3_stride_##PTR = PTR->stride[3]

#define YAMI_TENSOR_DIMS(PTR, n)    YAMI_TENSOR_DIMS_##n(PTR)
#define YAMI_TENSOR_STRIDES(PTR, n) YAMI_TENSOR_STRIDES_##n(PTR)
#define YAMI_TENSOR_FIELDS(PTR, n)  YAMI_TENSOR_DIMS(PTR, n); YAMI_TENSOR_STRIDES(PTR, n)

#define yami_for_0(PTR) for (usize d0 = 0; d0 < (d0_##PTR); ++d0)

#define yami_for_1(PTR) yami_for_0(PTR) \
                        for (usize d1 = 0; d1 < (d1_##PTR); ++d1)

#define yami_for_2(PTR) yami_for_1(PTR) \
                        for (usize d2 = 0; d2 < (d2_##PTR); ++d2)

#define yami_for_3(PTR) yami_for_2(PTR) \
                        for (usize d3 = 0; d3 < (d3_##PTR); ++d3)

#define yami_for(PTR, n) yami_for_##n(PTR)

#define yami_offset_0(PTR) ((d0) * (d0_stride_##PTR))
#define yami_offset_1(PTR) ((yami_offset_0(PTR)) + ((d1) * (d1_stride_##PTR)))
#define yami_offset_2(PTR) ((yami_offset_1(PTR)) + ((d2) * (d2_stride_##PTR)))
#define yami_offset_3(PTR) ((yami_offset_2(PTR)) + ((d3) * (d3_stride_##PTR)))
#define yami_offset(PTR, n) yami_offset_##n(PTR)

#define yami_broadcast_offset_0(PTR) (((d0) % (d0_##PTR)) * (d0_stride_##PTR))
#define yami_broadcast_offset_1(PTR) ((yami_broadcast_offset_0(PTR)) + (((d1) % (d1_##PTR)) * (d1_stride_##PTR)))
#define yami_broadcast_offset_2(PTR) ((yami_broadcast_offset_1(PTR)) + (((d2) % (d2_##PTR)) * (d2_stride_##PTR)))
#define yami_broadcast_offset_3(PTR) ((yami_broadcast_offset_2(PTR)) + (((d3) % (d3_##PTR)) * (d3_stride_##PTR)))
#define yami_broadcast_offset(PTR, n) yami_broadcast_offset_##n(PTR)

// Returns the 'real' dimension given a number which can be either positive or negative,
// dim=-1 means 'last dimension', dim=-2 'last but one' ecc...
#define yami_tensor_get_dim(PTR, dim) ((dim) < 0) ? ((PTR->n_dim) + (dim)) : (dim)

#define yami_tensor_get_ext_dim(PTR, dim) (dim) + (YAMI_MAX_DIMS - (PTR->n_dim))

#define YAMI_RANGE_START    ((int) 0)
#define YAMI_RANGE_STOP     ((int) 1)
#define YAMI_BLOCK_SIZE     ((usize) 32)


using yami_dim_range = usize[YAMI_MAX_DIMS][2];

enum yami_op : u8 {
    YAMI_OP_MATMUL,
    YAMI_OP_DONE,
};

struct yami_task {
    const yami_tensor *xa;
    const yami_tensor *xb;
    yami_tensor *res;
    yami_dim_range ranges;
    // Counter shared between all the workers, used to determine whether they have finished or not.
    std::atomic_int *progress_counter;
    yami_op op;
};

struct yami_worker {
    pthread_t id;
    // The current implementation uses a spinlock in the "father" thread (0th thread) to wait for its "children" completion
    // and a condition variable in said children to wait for work.
    // Performance-wise this is a suboptimal choice as there will inevitably be some OS-related delay between when the father signals
    // and the child actually receives that signal. This is not the case with a spinlock however the spinlock will cause
    // higher utilization (all the cores will be maxed-out until the inference is done) since the children are created only once
    // and are utilized only (at least for now) to speed-up matmuls which means they are quite often idle.
    bool has_work;
    pthread_cond_t cond;
    pthread_mutex_t mtx;
    yami_task task;
};

struct yami_context {
    usize mem_size;
    void *mem_buffer;

    yami_obj *last;
    // Each context has his own internal context which can be used as a scratch buffer.
    yami_context *scratch;
    int n_workers;
    yami_worker *workers;

    int n_objs;
};

struct yami_obj {
    usize offset;
    usize obj_size;
};

// ============================================== Helper functions ==============================================
static void yami_set_tensor_dim(yami_tensor *x, const int n_dim,
                                const usize *dimensions) noexcept {
    x->n_dim = n_dim;
    memcpy(x->dimensions, dimensions, x->n_dim * sizeof(size));
    // Compute the extended dimension
    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        x->extended_dim[YAMI_MAX_DIMS - i - 1] = i >= x->n_dim ? 1 : x->dimensions[x->n_dim - i - 1];
    }
    // Compute the stride
    memset(x->stride, 0, YAMI_MAX_DIMS * sizeof(size));
    for (int i = YAMI_MAX_DIMS - 1; i >= 0; --i) {
        x->stride[i] = i == YAMI_MAX_DIMS - 1 ? 1 : x->stride[i + 1] * x->extended_dim[i + 1];
    }
}

static bool yami_can_broadcast(const yami_tensor *xa, const yami_tensor *xb,
                               const int dims_to_check = YAMI_MAX_DIMS) noexcept {
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

static yami_tensor *yami_alloc_result(yami_context *ctx, const yami_tensor *xa,
                                      const yami_tensor *xb, const char *label = "") noexcept {

    YAMI_ASSERT(yami_can_broadcast(xa, xb));

    usize extend_res[YAMI_MAX_DIMS];
    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        extend_res[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);
    }

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);

    yami_tensor *res = yami_new_tensor(ctx, res_n_dim, &extend_res[YAMI_MAX_DIMS - res_n_dim], label);
    return res;
}

// ==============================================================================================================

static inline void yami_internal_matmul_naive(f32 *__restrict out, const f32 *__restrict xa,
                                              const f32 *__restrict xb, const usize n_rows_a,
                                              const usize n_cols_a, const usize n_cols_b,
                                              const usize a_row_start = 0, usize a_row_stop = 0,
                                              const usize b_col_start = 0, usize b_col_stop = 0) noexcept {
    b_col_stop = b_col_stop == 0 ? n_cols_b : YAMI_MIN(b_col_stop, n_cols_b);
    a_row_stop = a_row_stop == 0 ? n_rows_a : YAMI_MIN(a_row_stop, n_rows_a);
    for (usize i = a_row_start; i < a_row_stop; ++i) {
        for (usize k = 0; k < n_cols_a; ++k) {
            const f32 c_a = xa[i * n_cols_a + k];
            for (usize j = b_col_start; j < b_col_stop; ++j) {
                out[i * n_cols_b + j] += c_a * xb[k * n_cols_b + j];
            }
        }
    }
}

// Some things to keep in mind:
// - L1 cache is 32KB
// - L2 cache is 256KB
// - cache line is 64B
// - TLB lv1 is 64 entries for 4KB/entry
// - AVX2
static inline void yami_internal_matmul_blocked(f32 *__restrict res, const f32 *__restrict xa,
                                                const f32 *__restrict xb, const usize n_rows_a,
                                                const usize n_cols_a, const usize n_cols_b,
                                                const usize a_row_start = 0, usize a_row_stop = 0,
                                                const usize b_col_start = 0, usize b_col_stop = 0) noexcept {
    // This is the first implementation of what looks like a very efficient GEMM algorithm.
    // TODO: we have to investigate 2 things:
    //  - what's the best set of parameters for this algorithm
    //  - how to find the best set of parameters at runtime (e.g. during init) for a given hardware

    const usize b_size = YAMI_BLOCK_SIZE * sizeof(f32);
    alignas(32) f32 *packed_a = (f32 *) alloca(b_size);
    alignas(32) f32 *packed_b = (f32 *) alloca(b_size * YAMI_BLOCK_SIZE);
    alignas(32) f32 *packed_c = (f32 *) alloca(b_size);

    b_col_stop = b_col_stop == 0 ? n_cols_b : YAMI_MIN(b_col_stop, n_cols_b);
    a_row_stop = a_row_stop == 0 ? n_rows_a : YAMI_MIN(a_row_stop, n_rows_a);
    for (usize bj = b_col_start; bj < b_col_stop; bj += YAMI_BLOCK_SIZE) {

        const usize max_bj = YAMI_MIN(bj + YAMI_BLOCK_SIZE, b_col_stop);

        for (usize bi = 0; bi < n_cols_a; bi += YAMI_BLOCK_SIZE) {

            const usize max_bi = YAMI_MIN(bi + YAMI_BLOCK_SIZE, n_cols_a);

            const usize block_rows = max_bi - bi;
            const usize block_cols = max_bj - bj;
            // Fill the buffer with 0s and copy only those values that are part of the YAMI_BLOCK_SIZE x YAMI_BLOCK_SIZE matrix
            memset(packed_b, 0, b_size * YAMI_BLOCK_SIZE);
            for (usize ib = 0; ib < block_cols; ++ib) {
                for (usize jb = 0; jb < block_rows; ++jb) {
                    // pack B in column-major order
                    // todo: optimize!
                    packed_b[ib * YAMI_BLOCK_SIZE + jb] = xb[(bi + jb) * n_cols_b + (bj + ib)];
                }
            }

            // Take the given subset of rows of A from b_r_start to b_r_stop and multiply by pack_b
            for (usize i = a_row_start; i < a_row_stop; ++i) {
                memset(packed_a, 0, b_size);
                for (usize tmp = bi; tmp < max_bi; ++tmp) packed_a[tmp - bi] = xa[i*n_cols_a + tmp];

                // Block multiply
                for (usize kk = 0; kk < YAMI_BLOCK_SIZE; ++kk) {
                    f32 acc = 0.f;
                    for (usize jj = 0; jj < YAMI_BLOCK_SIZE; ++jj) {
                        // todo: optimize!
                        acc += packed_a[jj] * packed_b[kk * YAMI_BLOCK_SIZE + jj];
                    }
                    packed_c[kk] = acc;
                }

                for (usize tmp = bj; tmp < max_bj; ++tmp) {
                    res[i*n_cols_b + tmp] += packed_c[tmp - bj];
                }

            }
        }
    }
}

static void yami_internal_matmul(const yami_tensor *xa, const yami_tensor *xb,
                                 yami_tensor *res, const yami_dim_range ranges) noexcept {
    YAMI_TENSOR_FIELDS(xa, 1);
    YAMI_TENSOR_FIELDS(xb, 1);
    YAMI_TENSOR_STRIDES(res, 1);
    const usize d2_xa = xa->extended_dim[2];
    const usize d3_xa = xa->extended_dim[3];
    const usize d3_xb = xb->extended_dim[3];

    const usize d2_start = ranges[2][YAMI_RANGE_START], d2_stop = ranges[2][YAMI_RANGE_STOP];
    const usize d3_start = ranges[3][YAMI_RANGE_START], d3_stop = ranges[3][YAMI_RANGE_STOP];

    const bool use_naive = (d2_stop - d2_start) < YAMI_BLOCK_SIZE &&
                           (d3_stop - d3_start) < YAMI_BLOCK_SIZE &&
                           d2_xa < YAMI_BLOCK_SIZE;
    for (usize d0 = ranges[0][YAMI_RANGE_START]; d0 < ranges[0][YAMI_RANGE_STOP]; ++d0) {
        for (usize d1 = ranges[1][YAMI_RANGE_START]; d1 < ranges[1][YAMI_RANGE_STOP]; ++d1) {
            f32 *res_data = &res->data[yami_offset(res, 1)];
            const f32 *xa_data = &xa->data[yami_broadcast_offset(xa, 1)];
            const f32 *xb_data = &xb->data[yami_broadcast_offset(xb, 1)];

            if (use_naive) yami_internal_matmul_naive(res_data, xa_data, xb_data, d2_xa, d3_xa,
                                                      d3_xb, d2_start, d2_stop,
                                                      d3_start, d3_stop);
            else yami_internal_matmul_blocked(res_data, xa_data,
                                              xb_data, d2_xa, d3_xa,
                                              d3_xb, d2_start, d2_stop,
                                              d3_start, d3_stop);
        }
    }
}

static yami_context *yami_internal_ctx_init(const usize mem_size) noexcept {
    yami_context *ctx = (yami_context *) malloc(sizeof(yami_context));
    if (ctx == nullptr) {
        YAMI_LOG_ERR("error allocating context");
        return nullptr;
    }
    ctx->mem_size = mem_size;
    ctx->last = nullptr;
    ctx->n_objs = 0;

    ctx->mem_buffer = malloc(ctx->mem_size);

    if (ctx->mem_buffer == nullptr) {
        YAMI_LOG_ERR("error allocating context memory buffer");
        free(ctx);
        return nullptr;
    }

    return ctx;
}

static void *yami_worker_thread(void *arg) noexcept {
    yami_worker *self = (yami_worker *) arg;

    yami_task *work;
    bool exit = false;
    while (true) {
        pthread_mutex_lock(&self->mtx);
        while (!self->has_work)
            pthread_cond_wait(&self->cond, &self->mtx);

        self->has_work = false;
        pthread_mutex_unlock(&self->mtx);

        // We don't need to hold the lock as the master thread will wait for completion before setting another task
        work = &self->task;

        switch (work->op) {
            case YAMI_OP_MATMUL: {
                const yami_tensor *xa = work->xa;
                const yami_tensor *xb = work->xb;
                yami_tensor *res = work->res;
                yami_internal_matmul(xa, xb, res, work->ranges);
                work->progress_counter->fetch_add(1);
                break;
            }
            case YAMI_OP_DONE: {
                exit = true;
                break;
            }
            default: {
                YAMI_LOG_ERR("unknown op=%d", work->op);
                break;
            }
        }

        if (exit)
            break;
    }

    pthread_exit(nullptr);
}

yami_context *yami_init(const yami_context_init_params params) noexcept {
    if (params.mem_size <= 0) {
        YAMI_LOG_ERR("invalid memory size %ld", params.mem_size);
        return nullptr;
    }

    yami_context *ctx = yami_internal_ctx_init(params.mem_size);
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->scratch = nullptr;

    if (params.scratch_mem_size > 0) {
        yami_context *scratch = yami_internal_ctx_init(params.scratch_mem_size);
        if (scratch == nullptr) {
            free(ctx);
            return nullptr;
        }

        ctx->scratch = scratch;
    }

    const int n_cpus = get_nprocs();
    int n_workers = params.n_workers;
    if (n_workers > n_cpus) {
        YAMI_LOG_INFO("n_workers=%d > n_cpus=%d, the actual number of workers will be limited to n_cpus", n_workers, n_cpus);
        n_workers = n_cpus;
    }

    // Create the Yami workforce!
    ctx->n_workers = n_workers;
    if (ctx->scratch != nullptr)
        ctx->scratch->n_workers = n_workers;

    if (n_workers > 1) {
        ctx->workers = (yami_worker *) malloc((n_workers - 1) * sizeof(yami_worker));
        for (int i = 1; i < n_workers; ++i) {
            yami_worker *worker = &ctx->workers[i-1];
            pthread_mutex_init(&worker->mtx, nullptr);
            pthread_cond_init(&worker->cond, nullptr);
            worker->has_work = false;
            pthread_create(&worker->id, nullptr, yami_worker_thread, worker);

            // Set affinity
            cpu_set_t cpu_set{};
            CPU_ZERO(&cpu_set);
            CPU_SET(i, &cpu_set);
            pthread_setaffinity_np(worker->id, sizeof(cpu_set_t), &cpu_set);
        }

        // Set the same workers also in the scratch buffer
        // Todo: keep this in check, maybe in the future it won't be such a good idea
        if (ctx->scratch != nullptr) {
            ctx->scratch->workers = ctx->workers;
        }
    }

    // Set affinity for the main worker
    cpu_set_t cpu_set{};
    CPU_ZERO(&cpu_set);
    CPU_SET(0, &cpu_set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);

    YAMI_LOG_INFO("created new context %p size=%ldMB scratch=%ldMB workers=%d",
                  (void *) ctx, (usize) YAMI_B_TO_MB(params.mem_size),
                  (usize) YAMI_B_TO_MB(params.scratch_mem_size), ctx->n_workers);

    return ctx;
}

void yami_free(yami_context *ctx) noexcept {
    YAMI_LOG_DEBUG("freeing context %p size=%.2fMB has_scratch=%d",
                   (void *) ctx, YAMI_B_TO_MB(ctx->mem_size), ctx->scratch != nullptr);

    if (ctx->scratch != nullptr) {
        free(ctx->scratch->mem_buffer);
        free(ctx->scratch);
    }

    free(ctx->mem_buffer);

    if (ctx->n_workers > 1) {
        for (int i = 1; i < ctx->n_workers; ++i) {
            yami_worker *worker = &ctx->workers[i - 1];
            pthread_mutex_lock(&worker->mtx);

            worker->task.op = YAMI_OP_DONE;
            worker->has_work = true;

            pthread_cond_signal(&worker->cond);
            pthread_mutex_unlock(&worker->mtx);
        }

        for (int i = 1; i < ctx->n_workers; ++i) {
            yami_worker *worker = &ctx->workers[i - 1];
            pthread_join(worker->id, nullptr);
            pthread_mutex_destroy(&worker->mtx);
            pthread_cond_destroy(&worker->cond);
        }

        free(ctx->workers);
    }

    free(ctx);
}

void yami_clear_ctx(yami_context *ctx) noexcept {
    YAMI_LOG_DEBUG("clearing context %p mem_size=%ldMB n_objs=%d",
                   (void *)ctx, (usize) YAMI_B_TO_MB(ctx->mem_size), ctx->n_objs);

    ctx->last = nullptr;
    ctx->n_objs = 0;
}

yami_context *yami_ctx_scratch(yami_context *ctx) noexcept {
    YAMI_ASSERT(ctx->scratch != nullptr);

    return ctx->scratch;
}

usize yami_used_mem(const yami_context *ctx) noexcept {
    return ctx->last == nullptr ? 0 : ctx->last->offset + ctx->last->obj_size;
}

void yami_mem_usage(const yami_context *ctx) noexcept {
    const usize used = yami_used_mem(ctx);
    YAMI_LOG_INFO("n_objs=%d used memory %.2fMB out of %ldMB (%.2f%%)",
                  ctx->n_objs,
                  YAMI_B_TO_MB(used),
                  (usize) YAMI_B_TO_MB(ctx->mem_size),
                  ((f64) (used) / (f64) (ctx->mem_size)) * 100.
    );
}

yami_tensor *yami_new_tensor(yami_context *ctx, const int n_dim,
                             const usize *dimensions, const char *label,
                             void *data) noexcept {

    if ((unsigned) n_dim > YAMI_MAX_DIMS) {
        YAMI_LOG_ERR("%d is not a valid number of dimensions", n_dim);
        return nullptr;
    }

    if (strlen(label) > YAMI_LABEL_SIZE) {
        YAMI_LOG_ERR("label \"%s\" is too long", label);
        return nullptr;
    }

    usize ne = 1;
    for (int i = 0; i < n_dim; ++i)
        ne *= dimensions[i];

    usize mem_needed = sizeof(yami_tensor);
    if (data == nullptr) {
        mem_needed += ne * sizeof(f32);
    }

    const usize last_offset = ctx->last == nullptr ? 0 : ctx->last->offset;
    const usize last_size = ctx->last == nullptr ? 0 : ctx->last->obj_size;
    const usize last_end = last_offset + last_size;

    if (mem_needed + sizeof(yami_obj) + last_end > ctx->mem_size) {
        YAMI_LOG_ERR("can't allocate %.2fMB", YAMI_B_TO_MB(mem_needed));
        return nullptr;
    }

    yami_obj *new_obj = (yami_obj *) ((byte *)ctx->mem_buffer + last_end);
    // The offset refers to the actual offset of the "contained" object which comes after
    // the yami_object "header".
    new_obj->offset = last_end + sizeof(yami_obj);
    new_obj->obj_size = mem_needed;

    ctx->last = new_obj;
    ctx->n_objs++;
    // Allocate the actual tensor
    yami_tensor *new_tensor = (yami_tensor *) ((byte *)ctx->mem_buffer + new_obj->offset);

    new_tensor->ne = ne;
    if (data == nullptr) {
        new_tensor->data = (f32 *) (new_tensor + 1);
        memset(new_tensor->data, 0, ne * sizeof(f32));
    } else {
        new_tensor->data = (f32 *) data;
    }

    strncpy(new_tensor->label, label, YAMI_LABEL_SIZE);

    yami_set_tensor_dim(new_tensor, n_dim, dimensions);

    YAMI_LOG_DEBUG("label=\"%s\" n_dim=%d extended_dim=[%ld, %ld, %ld, %ld] stride=[%ld, %ld, %ld, %ld] data=%p",
                   label, n_dim, new_tensor->extended_dim[0], new_tensor->extended_dim[1], new_tensor->extended_dim[2],
                   new_tensor->extended_dim[3], new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2],
                   new_tensor->stride[3], data);

    return new_tensor;
}

yami_tensor *yami_tensor_1d(yami_context *ctx, const char *label,
                            const usize dim1) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1};
    return yami_new_tensor(ctx, 1, dims, label);
}

yami_tensor *yami_tensor_2d(yami_context *ctx, const char *label,
                            const usize dim1, const usize dim2) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2};
    return yami_new_tensor(ctx, 2, dims, label);
}

yami_tensor *yami_tensor_3d(yami_context *ctx, const char *label,
                            const usize dim1, const usize dim2,
                            const usize dim3) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, 3, dims, label);
}

yami_tensor *yami_tensor_4d(yami_context *ctx, const char *label,
                            const usize dim1, const usize dim2,
                            const usize dim3, const usize dim4) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, 4, dims, label);
}

yami_tensor *yami_view_1d(yami_context *ctx, yami_tensor *x,
                          const usize dim1, const usize offset) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1};
    return yami_new_tensor(ctx, 1, dims, x->label, &x->data[offset]);
}

yami_tensor *yami_view_2d(yami_context *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize offset) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2};
    return yami_new_tensor(ctx, 2, dims, x->label, &x->data[offset]);
}

yami_tensor *yami_view_3d(yami_context *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize dim3, const usize offset) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, 3, dims, x->label, &x->data[offset]);
}

yami_tensor *yami_view_4d(yami_context *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize dim3, const usize dim4,
                          const usize offset) noexcept{
    const usize dims[YAMI_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, 4, dims, x->label, &x->data[offset]);
}

yami_tensor *yami_reshape(yami_tensor *x, const int n_dims...) noexcept {
    YAMI_ASSERT(n_dims > 0 && n_dims <= YAMI_MAX_DIMS);

    usize new_dimensions[YAMI_MAX_DIMS];

    std::va_list args;
    va_start(args, n_dims);
    for (int i = 0; i < n_dims; ++i) {
        const usize d = va_arg(args, usize);
        YAMI_ASSERT(d > 0);
        new_dimensions[i] = d;
    }
    va_end(args);


    usize new_ne = 1;
    for (int i = 0; i < n_dims; ++i)
        new_ne *= new_dimensions[i];

    YAMI_ASSERT(new_ne == x->ne);

    yami_set_tensor_dim(x, n_dims, new_dimensions);

    return x;
}

yami_tensor *yami_transpose(yami_context *, yami_tensor *x,
                            const int dim1, const int dim2) noexcept {
    YAMI_ASSERT(x->n_dim > 1);

    const usize real_d1 = yami_tensor_get_dim(x, dim1);
    const usize real_d2 = yami_tensor_get_dim(x, dim2);

    YAMI_ASSERT(real_d1 < YAMI_MAX_DIMS);
    YAMI_ASSERT(real_d2 < YAMI_MAX_DIMS);

    const usize ext_d1 = yami_tensor_get_ext_dim(x, real_d1);
    const usize ext_d2 = yami_tensor_get_ext_dim(x, real_d2);

    const usize tmp_d1 = x->dimensions[real_d1];
    x->dimensions[real_d1] = x->dimensions[real_d2];
    x->dimensions[real_d2] = tmp_d1;

    const usize tmp_ed1 = x->extended_dim[ext_d1];
    x->extended_dim[ext_d1] = x->extended_dim[ext_d2];
    x->extended_dim[ext_d2] = tmp_ed1;

    const usize tmp_s1 = x->stride[ext_d1];
    x->stride[ext_d1] = x->stride[ext_d2];
    x->stride[ext_d2] = tmp_s1;

    return x;
}

yami_tensor *yami_contiguous(yami_context *ctx, yami_tensor *x) noexcept {
    // Checking whether x is contiguous is straightforward: check if the last dimension has stride 1
    // and all the others are in non-decreasing order.
    // todo: add contiguous flag to yami_tensor, some ops right now assume that the underlying tensors are contiguous (e.g. matmul)
    bool ordered = x->stride[YAMI_MAX_DIMS - 1] == 1;
    for (size i = 0; i < YAMI_MAX_DIMS - 1 && ordered; ++i)
        ordered &= x->stride[i] >= x->stride[i + 1];

    if (ordered)
        return x;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, x->dimensions);

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        res->data[yami_offset(res, 3)] = x->data[yami_offset(x, 3)];
    }

    return res;
}

yami_tensor *yami_lt_mask(yami_context *, yami_tensor *x,
                          const f32 mask, const usize start_idx) noexcept {
    YAMI_ASSERT(x->n_dim >= 2);

    YAMI_TENSOR_FIELDS(x, 3);

    YAMI_ASSERT(start_idx < d3_x);

    yami_for(x, 2) {
        for (usize d3 = start_idx; d3 < d3_x; ++d3) {
            if (d3 > (d2 + start_idx))
                x->data[yami_offset(x, 3)] = mask;
        }
    }

    return x;
}

yami_tensor *yami_mask_if(yami_context *, yami_tensor *x,
                          const yami_mask_flag flag, const f32 val,
                          const f32 mask) noexcept {
    YAMI_TENSOR_FIELDS(x, 3);
    yami_for(x, 3) {
        const usize offset = yami_offset(x, 3);
        const f32 x_val = x->data[offset];
        switch (flag) {
            case LOWER:
                if (x_val < val) x->data[offset] = mask;
                break;
            case EQUAL:
                if (x_val == val) x->data[offset] = mask;
                break;
            case GREATER:
                if (x_val > val) x->data[offset] = mask;
                break;
            default:
                break;
        }
    }
    return x;
}

yami_tensor *yami_embed(yami_context *ctx, const yami_tensor *x,
                        const int *indexes, const usize n) noexcept {
    YAMI_ASSERT(x->n_dim == 2);

    const usize max_idx = x->dimensions[yami_tensor_get_dim(x, -2)];
    const usize emb_size = x->dimensions[yami_tensor_get_dim(x, -1)];
    usize res_dim[2] = {n, emb_size};

    yami_tensor *res = yami_new_tensor(ctx, 2, res_dim);

    const usize d2_stride_x = x->stride[2];
    const usize d2_stride_res = res->stride[2];
    for (usize i = 0; i < n; ++i) {
        const usize idx = (usize) indexes[i];
        YAMI_ASSERT(idx < max_idx);

        const usize x_offset = idx * d2_stride_x;
        const usize res_offset = i * d2_stride_res;
        memcpy(&res->data[res_offset], &x->data[x_offset], emb_size * sizeof(f32));
    }

    return res;
}

yami_tensor *yami_split(yami_context *ctx, const yami_tensor *x,
                        const usize n, const int offset,
                        const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    const usize x_dim_ne = x->dimensions[real_dim];

    YAMI_ASSERT((x_dim_ne % n == 0) && (offset * n + n <= x_dim_ne));

    usize res_dims[YAMI_MAX_DIMS];
    memcpy(res_dims, x->dimensions, x->n_dim * sizeof(usize));
    res_dims[real_dim] = n;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, res_dims);

    yami_dim_range ranges;
    const int dim_ext_idx = yami_tensor_get_ext_dim(x, real_dim);

    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        const usize di = x->extended_dim[i];
        usize start = 0;
        usize stop = di;
        if (dim_ext_idx == i) {
            start = n * offset;
            stop = n * offset + n;
        }

        ranges[i][YAMI_RANGE_START] = start;
        ranges[i][YAMI_RANGE_STOP] = stop;
    }

    YAMI_TENSOR_STRIDES(x, 3);
    YAMI_TENSOR_FIELDS(res, 3);

    for (usize d0 = ranges[0][YAMI_RANGE_START]; d0 < ranges[0][YAMI_RANGE_STOP]; ++d0) {
        for (usize d1 = ranges[1][YAMI_RANGE_START]; d1 < ranges[1][YAMI_RANGE_STOP]; ++d1) {
            for (usize d2 = ranges[2][YAMI_RANGE_START]; d2 < ranges[2][YAMI_RANGE_STOP]; ++d2) {
                for (usize d3 = ranges[3][YAMI_RANGE_START]; d3 < ranges[3][YAMI_RANGE_STOP]; ++d3) {
                    res->data[yami_broadcast_offset(res, 3)] = x->data[yami_offset(x, 3)];
                }
            }
        }
    }

    return res;
}

yami_tensor *yami_concat(yami_context *ctx, const yami_tensor *xa,
                         const yami_tensor *xb, const int dim) noexcept {
    const int axis = yami_tensor_get_dim(xa, dim);
    YAMI_ASSERT(xa->n_dim == xb->n_dim && (unsigned) axis < YAMI_MAX_DIMS);

    for (int i = 0; i < xa->n_dim; ++i) {
        if (i != axis && xa->dimensions[i] != xb->dimensions[i]) {
            YAMI_LOG_ERR("cannot concatenate tensor of shape [%ld %ld %ld %ld] with tensor of shape [%ld %ld %ld %ld] along axis %d",
                         xb->extended_dim[0], xb->extended_dim[1], xb->extended_dim[2], xb->extended_dim[3],
                         xa->extended_dim[0], xa->extended_dim[1], xa->extended_dim[2], xa->extended_dim[3],
                         axis);
            return nullptr;
        }
    }

    usize new_dims[YAMI_MAX_DIMS];
    memcpy(new_dims, xa->dimensions, xa->n_dim * sizeof(usize));
    new_dims[axis] += xb->dimensions[axis];
    
    yami_tensor *res = yami_new_tensor(ctx, xa->n_dim, new_dims);

    switch (axis) {
        case 0:
            // Simple copy the content of the two tensors in order
            memcpy(res->data, xa->data, xa->ne * sizeof(f32));
            memcpy(&res->data[xa->ne], xb->data, xb->ne * sizeof(f32));
            break;
        case 1: {
            YAMI_TENSOR_FIELDS(res, 0);
            YAMI_TENSOR_STRIDES(xa, 0);
            YAMI_TENSOR_STRIDES(xb, 0);
            yami_for(res, 0) {
                // For each row we first copy the data from xa then from xb
                memcpy(&res->data[yami_offset(res, 0)],
                       &xa->data[yami_offset(xa, 0)],
                       d0_stride_xa * sizeof(f32)
                );
                memcpy(&res->data[yami_offset(res, 0) + d0_stride_xa],
                       &xb->data[yami_offset(xb, 0)],
                       d0_stride_xb * sizeof(f32)
                );
            }
            break;

        }
        case 2: {
            YAMI_TENSOR_FIELDS(res, 1);
            YAMI_TENSOR_STRIDES(xa, 1);
            YAMI_TENSOR_STRIDES(xb, 1);
            yami_for(res, 1) {
                memcpy(&res->data[yami_offset(res, 1)],
                       &xa->data[yami_offset(xa, 1)],
                       d1_stride_xa * sizeof(f32)
                );
                memcpy(&res->data[yami_offset(res, 1) + d1_stride_xa],
                       &xb->data[yami_offset(xb, 1)],
                       d1_stride_xb * sizeof(f32)
                );
            }
            break;
        }
        case 3: {
            YAMI_TENSOR_FIELDS(res, 2);
            YAMI_TENSOR_STRIDES(xa, 2);
            YAMI_TENSOR_STRIDES(xb, 2);
            yami_for(res, 2) {
                memcpy(&res->data[yami_offset(res, 2)],
                       &xa->data[yami_offset(xa, 2)],
                       d2_stride_xa * sizeof(f32)
                );
                memcpy(&res->data[yami_offset(res, 2) + d2_stride_xa],
                       &xb->data[yami_offset(xb, 2)],
                       d2_stride_xb * sizeof(f32)
                );
            }
            break;
        }
        default:
            break;
    }

    return res;
}

void yami_copy(const yami_tensor *x, yami_tensor *res) noexcept {
    yami_set_tensor_dim(res, x->n_dim, x->dimensions);
    memcpy(res->data, x->data, res->ne * sizeof(f32));
}

yami_tensor *yami_matmul(yami_context *ctx, const yami_tensor *xa,
                         const yami_tensor *xb, yami_tensor *res) noexcept {
    // Verify that the two matrices are at least 2-dimensional
    if (xa->n_dim < 2 || xb->n_dim < 2) {
        YAMI_LOG_FATAL("too few dimensions, use yami_mul for 1D tensor multiply");
    }

//    const f64 start__ = yami_timer();

    const usize xa_n_rows = xa->dimensions[yami_tensor_get_dim(xa, -2)];
    const usize xa_n_cols = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const usize xb_n_rows = xb->dimensions[yami_tensor_get_dim(xb, -2)];
    const usize xb_n_cols = xb->dimensions[yami_tensor_get_dim(xb, -1)];

    if (xa_n_cols != xb_n_rows) {
        YAMI_LOG_FATAL("can't multiply (%ld, %ld) by (%ld, %ld)",
                     xa_n_rows, xa_n_cols,
                     xb_n_rows, xb_n_cols);
    }

    YAMI_ASSERT(yami_can_broadcast(xa, xb, YAMI_MAX_DIMS - 2));

    usize res_dim[YAMI_MAX_DIMS];
    res_dim[YAMI_MAX_DIMS - 2] = xa_n_rows;
    res_dim[YAMI_MAX_DIMS - 1] = xb_n_cols;
    for (int i = 0; i < YAMI_MAX_DIMS - 2; ++i)
        res_dim[i] = YAMI_MAX(xa->extended_dim[i], xb->extended_dim[i]);

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);
    res = res == nullptr ? yami_new_tensor(ctx, res_n_dim, &res_dim[YAMI_MAX_DIMS - res_n_dim]) : res;
    bool match = res->n_dim == res_n_dim;
    for (int i = 0; i < res_n_dim && match; ++i) {
        match &= res->dimensions[i] == res_dim[YAMI_MAX_DIMS - res_n_dim + i];
    }
    YAMI_ASSERT(match);

    const int n_workers = ctx->n_workers;

    YAMI_TENSOR_DIMS(res, 3);

//    yami_dim_range ranges[12];
    yami_dim_range *ranges = (yami_dim_range *) alloca(n_workers * sizeof(yami_dim_range));
    // Initialize all the ranges
    for (int i = 0; i < n_workers; ++i) {
        ranges[i][0][YAMI_RANGE_START] = 0, ranges[i][0][YAMI_RANGE_STOP] = d0_res;
        ranges[i][1][YAMI_RANGE_START] = 0, ranges[i][1][YAMI_RANGE_STOP] = d1_res;
        ranges[i][2][YAMI_RANGE_START] = 0, ranges[i][2][YAMI_RANGE_STOP] = d2_res;
        ranges[i][3][YAMI_RANGE_START] = 0, ranges[i][3][YAMI_RANGE_STOP] = d3_res;
    }

    std::atomic_int progress(1);
    int w = 0;
    // If one of the batch dimensions is >= than the number of workers, split on that dimension.
    if (d0_res >= (usize) n_workers) {
        // Todo: collapse two branches
        const usize n = d0_res / n_workers;
        for (usize ni = 0; ni < n; ++ni) {
            const usize start = ni * n, stop = ni != n - 1 ? YAMI_MIN(ni * n + n, d0_res) : d0_res;
            ranges[w][0][YAMI_RANGE_START] = start, ranges[w][0][YAMI_RANGE_STOP] = stop;
            w++;
        }
    } else if (d1_res >= (usize) n_workers) {
        const usize n = d1_res / n_workers;
        for (usize ni = 0; ni < n; ++ni) {
            const usize start = ni * n, stop = ni != n - 1 ? YAMI_MIN(ni * n + n, d1_res) : d1_res;
            ranges[w][1][YAMI_RANGE_START] = start, ranges[w][1][YAMI_RANGE_STOP] = stop;
            w++;
        }
    } else {
        // 2D partitioning algorithm:
        // 1. define a grid of (tc x tr) threads
        // 2. compute mi = n_rows_a / tr and nj = n_cols_b / tc
        // 3. partition C in 2D such that Cij is a (mi x nj) matrix
        // 4. partition A along the rows such that Ai is a (mi x k) matrix
        // 5. partition B along the columns such that Bi is a (k x nj) matrix
        // 6. compute Cij += Ai * Bj on each thread

        // Todo: these must be defined in the context
        const int tc = 12, tr = 1;

        const usize mi = xa_n_rows >= tr ? (xa_n_rows / tr) : 1;
        const usize nj = xb_n_cols >= tc ? (xb_n_cols / tc) : 1;

        for (usize nr = 0; nr < tr; ++nr) {
            for (usize nc = 0; nc < tc; ++nc) {
                const usize start_r = nr * mi, stop_r = nr != (usize) tr - 1 ? YAMI_MIN(start_r + mi, d2_res) : d2_res;
                const usize start_c = nc * nj, stop_c = nc != (usize) tc - 1 ? YAMI_MIN(start_c + nj, d3_res) : d3_res;
                ranges[w][2][YAMI_RANGE_START] = start_r,  ranges[w][2][YAMI_RANGE_STOP] = stop_r;
                ranges[w][3][YAMI_RANGE_START] = start_c,  ranges[w][3][YAMI_RANGE_STOP] = stop_c;
                w++;
            }
        }
    }
    // Enqueue tasks for the workers
    for (int i = 1; i < w; ++i) {
        yami_worker *worker = &ctx->workers[i-1];
        pthread_mutex_lock(&worker->mtx);

        yami_task *t = &worker->task;
        t->xa = xa;
        t->xb = xb;
        t->res = res;
        memcpy(t->ranges, ranges[i], sizeof(yami_dim_range));
        t->progress_counter = &progress;
        t->op = YAMI_OP_MATMUL;
        worker->has_work = true;

        pthread_cond_signal(&worker->cond);
        pthread_mutex_unlock(&worker->mtx);
    }
    // Matrix C0,0 is always processed in the main thread
    yami_internal_matmul(xa, xb, res, ranges[0]);

    while (progress.load() != w)
        ;

//    const f64 stop__ = yami_timer();

//    printf("GEMM,[%ldx%ldx%ldx%ld],[%ldx%ldx%ldx%ld],%.3f\n",
//           xa->extended_dim[0], xa->extended_dim[1], xa->extended_dim[2], xa->extended_dim[3],
//           xb->extended_dim[0], xb->extended_dim[1], xb->extended_dim[2], xb->extended_dim[3],
//           (stop__ - start__) * 1000.);
//    YAMI_LOG_INFO("[%ld, %ld, %ld, %ld] x [%ld, %ld, %ld, %ld] took %fms",
//                  xa->extended_dim[0], xa->extended_dim[1], xa->extended_dim[2], xa->extended_dim[3],
//                  xb->extended_dim[0], xb->extended_dim[1], xb->extended_dim[2], xb->extended_dim[3],
//                  (stop__ - start__) * 1000.);

    return res;
}

static inline void yami_internal_vec_add(f32 *out, const f32 *xa,
                                         const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = xa[i] + xb[i];
}

static inline void yami_internal_vec_addc(f32 *out, const f32 *x,
                                          const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = x[i] + c;
}

// When performing vector operations on a tensor (such as add, multiply...) there are two possibilities:
//  - the last dims are equal                               --> sum over two equal tensors with size N
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

    const usize xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const usize xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const size n = YAMI_MAX(xa_1_ne, xb_1_ne);

    YAMI_TENSOR_FIELDS(xa, 2);
    YAMI_TENSOR_FIELDS(xb, 2);
    YAMI_TENSOR_FIELDS(res, 2);

    yami_for(res, 2) {
        const usize res_offset = yami_offset(res, 2);
        const usize xa_offset = yami_broadcast_offset(xa, 2);
        const usize xb_offset = yami_broadcast_offset(xb, 2);

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

    return res;
}

yami_tensor *yami_addc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_addc(res->data, x->data, c, x->ne);
    return res;
}

static inline void yami_internal_vec_sub(f32 *out, const f32 *xa,
                                         const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = xa[i] - xb[i];
}

static inline void yami_internal_vec_subc(f32 *out, const f32 *x,
                                          const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = x[i] - c;
}

static inline void yami_internal_c_vec_sub(f32 *out, const f32 c,
                                           const f32 *x, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = c - x[i];
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

    const usize xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const usize xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const usize n = YAMI_MAX(xa_1_ne, xb_1_ne);

    YAMI_TENSOR_FIELDS(xa, 2);
    YAMI_TENSOR_FIELDS(xb, 2);
    YAMI_TENSOR_FIELDS(res, 2);

    yami_for(res, 2) {
        const usize xa_offset = yami_broadcast_offset(xa, 2);
        const usize xb_offset = yami_broadcast_offset(xb, 2);
        const usize res_offset = yami_offset(res, 2);

        if (scalar) {
            // fixme: this is a very very very ugly implementation, I have to find something better
            //  for those operations that support broadcasting and are not commutative (sub, div for now)...
            if (xa_1_ne == 1)
                yami_internal_c_vec_sub(&res->data[res_offset],
                                        xa->data[xa_offset],
                                        &xb->data[xb_offset], n);
            else
                yami_internal_vec_subc(&res->data[res_offset],
                                       &xa->data[xa_offset],
                                       xb->data[xb_offset], n);
        } else {
            yami_internal_vec_sub(&res->data[res_offset],
                                  &xa->data[xa_offset],
                                  &xb->data[xb_offset],
                                  n);
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
                                         const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = xa[i] * xb[i];
}

static inline void yami_internal_vec_mulc(f32 *out, const f32 *x,
                                          const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = x[i] * c;
}

yami_tensor *yami_mul(yami_context *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const usize xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const usize n = YAMI_MAX(xa_1_ne, xb_1_ne);

    YAMI_TENSOR_FIELDS(xa, 2);
    YAMI_TENSOR_FIELDS(xb, 2);
    YAMI_TENSOR_FIELDS(res, 2);

    yami_for(res, 2) {
        const usize res_offset = yami_offset(res, 2);
        const usize xa_offset = yami_broadcast_offset(xa, 2);
        const usize xb_offset = yami_broadcast_offset(xb, 2);

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

    return res;
}

yami_tensor *yami_mulc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_mulc(res->data, x->data, c, res->ne);
    return res;
}

static inline void yami_internal_vec_div(f32 *out, const f32 *xa,
                                         const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = xa[i] / xb[i];
}

static inline void yami_internal_vec_divc(f32 *out, const f32 *x,
                                          const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = x[i] / c;
}

static inline void yami_internal_c_vec_div(f32 *out, const f32 c,
                                           const f32 *x, const usize n) noexcept {
    for (usize i = 0; i < n; ++i)
        out[i] = c / x[i];
}

yami_tensor *yami_div(yami_context *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dimensions[yami_tensor_get_dim(xa, -1)];
    const usize xb_1_ne = xb->dimensions[yami_tensor_get_dim(xb, -1)];
    const bool scalar = xa_1_ne == 1 || xb_1_ne == 1;

    const usize n = YAMI_MAX(xa_1_ne, xb_1_ne);

    YAMI_TENSOR_FIELDS(xa, 2);
    YAMI_TENSOR_FIELDS(xb, 2);
    YAMI_TENSOR_FIELDS(res, 2);

    yami_for(res, 2) {
        const usize res_offset = yami_offset(res, 2);
        const usize xa_offset = yami_broadcast_offset(xa, 2);
        const usize xb_offset = yami_broadcast_offset(xb, 2);

        if (scalar) {
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

    return res;
}

yami_tensor *yami_divc(yami_context *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_vec_divc(res->data, x->data, c, res->ne);
    return res;
}

static inline void yami_internal_gelu(f32 *out, const f32 *x,
                                      const usize n) noexcept {
    const f32 sqrt_2_pi = std::sqrt(M_2_PIf);
    const f32 c = 0.044715f;

    for (usize i = 0; i < n; ++i) {
        const f32 x_i = x[i];
        out[i] = (0.5f * x_i) * (1.f + tanhf(sqrt_2_pi * (x_i + (c * x_i) * (x_i * x_i))));
    }
}

yami_tensor *yami_gelu(yami_context *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);
    yami_internal_gelu(res->data, x->data, x->ne);
    return res;
}

yami_tensor *yami_sum(yami_context *ctx, const yami_tensor *x,
                      const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    usize res_dim[YAMI_MAX_DIMS];
    memcpy(res_dim, x->dimensions, x->n_dim * sizeof(usize));
    res_dim[real_dim] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, res_dim);

    const usize dim_ext_idx = yami_tensor_get_ext_dim(x, real_dim);

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        const usize in_idx = yami_offset(x, 3);

        usize out_idx;

        if (dim_ext_idx == 0) out_idx = ((d1 * d1_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_ext_idx == 1) out_idx = ((d0 * d0_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_ext_idx == 2) out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d3 * d3_stride_res);
        else out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d2 * d2_stride_res);

        res->data[out_idx] += x->data[in_idx];
    }

    return res;
}

yami_tensor *yami_mean(yami_context *ctx, const yami_tensor *x,
                       const int dim) noexcept {
    yami_tensor *res = yami_sum(ctx, x, dim);

    yami_mulc(ctx, res, 1.f / (f32) x->dimensions[yami_tensor_get_dim(x, dim)]);

    return res;
}

yami_tensor *yami_var(yami_context *ctx, yami_tensor *x,
                      const int dim) noexcept {
    // fixme: there are two extra allocations that can be prevented once we will have means to deallocate tensors
    yami_tensor *mean = yami_mean(ctx, x, dim);
    return yami_mean(ctx,
                     yami_square(ctx,
                                 yami_sub(ctx, x, mean),
                                 true),
                     dim
    );
}

yami_tensor *yami_exp(yami_context *ctx, yami_tensor *x,
                      const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (usize i = 0; i < x->ne; ++i)
        res->data[i] = std::exp(x->data[i]);

    return res;
}

yami_tensor *yami_sqrt(yami_context *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (usize i = 0; i < x->ne; ++i)
        res->data[i] = std::sqrt(x->data[i]);

    return res;
}

yami_tensor *yami_square(yami_context *ctx, yami_tensor *x,
                         const bool in_place) noexcept {

    yami_tensor *res = in_place ? x : yami_new_tensor(ctx, x->n_dim, x->dimensions);

    for (usize i = 0; i < x->ne; ++i) {
        const f32 val = x->data[i];
        res->data[i] = val * val;
    }

    return res;
}

yami_tensor *yami_max(yami_context *ctx, const yami_tensor *x,
                      const int dim) noexcept {
    const int real_dim = yami_tensor_get_dim(x, dim);

    usize res_dim[YAMI_MAX_DIMS];
    memcpy(res_dim, x->dimensions, x->n_dim * sizeof(usize));
    res_dim[real_dim] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, res_dim);

    for (usize i = 0; i < res->ne; ++i)
        res->data[i] = YAMI_MINUS_INF;

    const usize dim_ext_idx = yami_tensor_get_ext_dim(x, real_dim);

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        const usize in_idx = yami_offset(x, 3);

        usize out_idx;

        if (dim_ext_idx == 0) out_idx = ((d1 * d1_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_ext_idx == 1) out_idx = ((d0 * d0_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_ext_idx == 2) out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d3 * d3_stride_res);
        else out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d2 * d2_stride_res);

        res->data[out_idx] = YAMI_MAX(res->data[out_idx], x->data[in_idx]);
    }

    return res;
}

yami_tensor *yami_softmax(yami_context *ctx, yami_tensor *x,
                          const int dim) noexcept {
    yami_tensor *e_x = yami_exp(ctx,
                                yami_sub(ctx,
                                         x,
                                         yami_max(ctx, x, dim),
                                         true
                                )
    );
    return yami_div(ctx,
                    e_x,
                    yami_sum(ctx, e_x, dim),
                    true
    );
}

yami_tensor *yami_layer_norm(yami_context *ctx, const yami_tensor *w,
                             const yami_tensor *b, yami_tensor *x,
                             const bool in_place, const f32 eps) noexcept {
    // fixme: these two are tmp results that should be freed before returning
    yami_tensor *x_mean = yami_mean(ctx, x, -1);
    yami_tensor *x_var = yami_var(ctx, x, -1);

    yami_addc(ctx, x_var, eps);

    yami_tensor *out = yami_div(ctx,
                                yami_sub(ctx, x, x_mean, in_place),
                                yami_sqrt(ctx, x_var),
                                true
    );

    return yami_add(ctx,
                    yami_mul(ctx, out, w, true),
                    b,
                    true
    );
}
