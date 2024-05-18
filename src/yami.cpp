#include "yami.h"
#include "yami_blas.h"
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <pthread.h>
#include <sys/sysinfo.h>

#if defined(YAMI_TRACE)
#   include <vector>
#   include <algorithm>
#endif

#define YAMI_ENTER_PRIVATE_SCOPE(CTX)   const yami_scope __old_scope = (CTX)->scope; (CTX)->scope = PRIVATE; yami_clear_ctx((CTX))
#define YAMI_EXIT_PRIVATE_SCOPE(CTX)    (CTX)->scope = __old_scope

#define YAMI_TENSOR_DIMS_0(PTR) const usize d0_##PTR = (PTR)->dim[0]

#define YAMI_TENSOR_DIMS_1(PTR) YAMI_TENSOR_DIMS_0(PTR); \
                                const usize d1_##PTR = (PTR)->dim[1]

#define YAMI_TENSOR_DIMS_2(PTR) YAMI_TENSOR_DIMS_1(PTR); \
                                const usize d2_##PTR = (PTR)->dim[2]

#define YAMI_TENSOR_DIMS_3(PTR) YAMI_TENSOR_DIMS_2(PTR); \
                                const usize d3_##PTR = (PTR)->dim[3]

#define YAMI_TENSOR_STRIDES_0(PTR)  const usize d0_stride_##PTR = (PTR)->stride[0]

#define YAMI_TENSOR_STRIDES_1(PTR)  YAMI_TENSOR_STRIDES_0(PTR); \
                                    const usize d1_stride_##PTR = (PTR)->stride[1]

#define YAMI_TENSOR_STRIDES_2(PTR)  YAMI_TENSOR_STRIDES_1(PTR); \
                                    const usize d2_stride_##PTR = (PTR)->stride[2]

#define YAMI_TENSOR_STRIDES_3(PTR)  YAMI_TENSOR_STRIDES_2(PTR); \
                                    const usize d3_stride_##PTR = (PTR)->stride[3]

#define YAMI_TENSOR_DIMS(PTR, n)    YAMI_TENSOR_DIMS_##n(PTR)
#define YAMI_TENSOR_STRIDES(PTR, n) YAMI_TENSOR_STRIDES_##n(PTR)
#define YAMI_TENSOR_FIELDS(PTR, n)  YAMI_TENSOR_DIMS(PTR, n); YAMI_TENSOR_STRIDES(PTR, n)

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

#define yami_for_0(PTR) for (usize d0 = 0; d0 < (d0_##PTR); ++d0)

#define yami_for_1(PTR) yami_for_0(PTR) \
                        for (usize d1 = 0; d1 < (d1_##PTR); ++d1)

#define yami_for_2(PTR) yami_for_1(PTR) \
                        for (usize d2 = 0; d2 < (d2_##PTR); ++d2)

#define yami_for_3(PTR) yami_for_2(PTR) \
                        for (usize d3 = 0; d3 < (d3_##PTR); ++d3)

#define yami_for(PTR, n) yami_for_##n(PTR)


// Create a new 'shape' for the given tensor. A shape is always YAMI_MAX_DIMS elements otherwise the indexes returned by
// yami_tensor_dim will not work on the shape copy. To make the process of using this copy less clunky yami_shape can be used,
// this way is easy to pass a valid pointer to yami_new_tensor (which expects the first element to be the actual first valid shape).
#define yami_shape_of(PTR)  usize PTR##_shape[YAMI_MAX_DIMS]; memcpy(PTR##_shape, (PTR)->dim, YAMI_MAX_DIMS * sizeof(usize))
#define yami_shape(PTR)     (&(PTR##_shape)[yami_tensor_dim(PTR, 0)])

#define yami_new_like(PTR)  (yami_new_tensor(ctx, (PTR)->n_dim, &(PTR)->dim[yami_tensor_dim(PTR, 0)]))
#define yami_set_zero(PTR)  (memset((PTR)->data, 0, (PTR)->ne * sizeof(f32)))
#define yami_scope_buffer() (ctx->buffers[ctx->scope])

#define YAMI_RANGE_START    ((int) 0)
#define YAMI_RANGE_STOP     ((int) 1)

#define YAMI_DTYPES         ((int) 1)

#if defined(YAMI_TRACE)
#   define YAMI_TRACE_COLUMNS  ((int) 7)
#   define YAMI_TRACE_KERNELS  ((int) 2)
#   define YAMI_TRACE_NODES    ((int) 100)

enum yami_kernel : u8 {
    GEMM,
    GEVM,
};

constexpr static const char *const kernel_labels[YAMI_TRACE_KERNELS] = {
        "GEMM",
        "GEVM",
};

static_assert(YAMI_TRACE_COLUMNS == 7, "YAMI_TRACE_COLUMNS != 7: update");
static_assert(YAMI_TRACE_KERNELS == 2, "YAMI_TRACE_KERNELS != 2: update");
#endif

enum yami_dtype : u8 {
    F32,
};

constexpr static const char *const dtype_labels[YAMI_DTYPES] = {
        "F32",
};

static_assert(YAMI_DTYPES == 1, "YAMI_DTYPES != 1: update");

using yami_dim_range = usize[YAMI_MAX_DIMS][2];

enum yami_op : u8 {
    YAMI_OP_GEMM,
    YAMI_OP_GEVM,
    YAMI_OP_DONE,
};

struct yami_task {
    const f32 *__restrict xa;
    const f32 *__restrict xb;
    f32 *__restrict res;
    usize n, k, stride_b;
    // Counter shared between all the workers, used to determine whether they have finished or not.
    std::atomic_int *progress_counter;
    yami_op op;
};

// TODO: add an index for each worker
struct yami_worker {
    pthread_t id;
    // The current implementation uses a spinlock in the main thread (0th thread) to wait for its workers to complete
    // and a condition variable in said workers to wait for work.
    // Performance-wise this is a suboptimal choice as there will inevitably be some OS-related delay between when the main thread signals
    // and the worker actually receives that signal. This is not the case with a spinlock however the spinlock will cause
    // higher utilization (all the cores will be maxed-out until the inference is done) since the workers are created only once
    // and are utilized only (at least for now) to speed-up GEVM which means they are quite often idle.
    pthread_cond_t cond;
    pthread_mutex_t mtx;
    bool has_work;
    yami_task task;
};

struct yami_obj {
    usize offset;
    usize nb;
};

// Simple trick: instead of having something like this:
//      struct yami_mem_buffer {
//          usize nb;
//          void *mem;
//      };
// we can avoid the `void *mem` part and just allocate enough memory for the struct plus the actual memory buffer.
// The actual address of the memory buffer starts at `sizeof(yami_mem_buffer)`.
struct yami_mem_buffer {
    usize nb;
    yami_obj *last;
    int n_objs;
};

#if defined(YAMI_TRACE)
struct yami_trace {
    yami_trace(const usize in_dim_1[YAMI_MAX_DIMS],
               const usize in_dim_2[YAMI_MAX_DIMS],
               const usize out[YAMI_MAX_DIMS],
               const usize flop_,
               const yami_kernel kernel_,
               const yami_dtype dtype_) noexcept :  flop(flop_), kernel(kernel_), dtype(dtype_) {
        memcpy(in_dims, in_dim_1, YAMI_MAX_DIMS * sizeof(usize));
        memcpy(in_dims[1], in_dim_2, YAMI_MAX_DIMS * sizeof(usize));
        memcpy(out_dim, out, YAMI_MAX_DIMS * sizeof(usize));
    }
    // For now assume always 2 inputs and 1 output
    usize in_dims[2][YAMI_MAX_DIMS]{};
    usize out_dim[YAMI_MAX_DIMS]{};

    usize ref{};
    f64 time_ms{};
    // Number of floating point operations performed by this kernel
    usize flop;
    yami_kernel kernel;
    yami_dtype dtype;
};

struct yami_trace_node {
    yami_trace el;
    // Next element to handle multiple elements mapped to the same bucket
    yami_trace_node *next;
};
#endif

// Each context has YAMI_SCOPES(3) buffers. A buffer is a memory region handled using a linear (arena) allocator, holding objects with the same lifetime.
// Each new allocation happens in the buffer specified by `scope`, same goes for clearing.
// The size of the GLOBAL buffer must be specified when initializing the context. This can be used for objects that should be initialized once and never modified
// like the weights in a neural network. After loading the GLOBAL objects the user has to manually change the default scope to LOCAL.
// For the LOCAL arena the size can either be specified at initialization or it will be automatically set to be 10% of the GLOBAL buffer size, which will in turn be reduced.
// The PRIVATE buffer size can't be specified, it's either 10% of the LOCAL buffer (if specified) or 10% of the GLOBAL one. Again, this means that the
// actual size of the GLOBAL/LOCAL buffer will be reduced.
//
// The PRIVATE buffer is used internally as a scratch buffer to store intermediate results inside certain functions (see: `yami_layer_norm`) and so the user can't use it.
// The GLOBAL and LOCAL buffers instead are completely user-managed.
struct yami_ctx {
#if defined(YAMI_TRACE)
    yami_trace_node *traces_table[YAMI_TRACE_NODES];
    f64 trace_start_time;
#endif

    yami_mem_buffer *buffers[YAMI_SCOPES];
    yami_worker *workers;
    int n_workers;
    yami_scope scope;
};

// ============================================== Math primitives ==============================================
static YAMI_INLINE void yami_internal_vec_add(f32 *out, const f32 *xa,
                                              const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = xa[i] + xb[i];
}

static YAMI_INLINE void yami_internal_vec_addc(f32 *out, const f32 *x,
                                               const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = x[i] + c;
}

static YAMI_INLINE void yami_internal_vec_sub(f32 *out, const f32 *xa,
                                              const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = xa[i] - xb[i];
}

static YAMI_INLINE void yami_internal_vec_subc(f32 *out, const f32 *x,
                                               const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = x[i] - c;
}

static YAMI_INLINE void yami_internal_c_vec_sub(f32 *out, const f32 c,
                                                const f32 *x, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = c - x[i];
}

static YAMI_INLINE void yami_internal_vec_mul(f32 *out, const f32 *xa,
                                              const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = xa[i] * xb[i];
}

static YAMI_INLINE void yami_internal_vec_mulc(f32 *out, const f32 *x,
                                               const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = x[i] * c;
}

static YAMI_INLINE void yami_internal_vec_div(f32 *out, const f32 *xa,
                                              const f32 *xb, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = xa[i] / xb[i];
}

static YAMI_INLINE void yami_internal_vec_divc(f32 *out, const f32 *x,
                                               const f32 c, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = x[i] / c;
}

static YAMI_INLINE void yami_internal_c_vec_div(f32 *out, const f32 c,
                                                const f32 *x, const usize n) noexcept {
    for (usize i = 0; i < n; ++i) out[i] = c / x[i];
}
// =============================================================================================================

// ============================================== Trace functions ==============================================
#if defined(YAMI_TRACE)
static int yami_trace_hash(const yami_trace *trace) noexcept {
    // Very simple djb2-like hashing function
    int hash = 5381;

    hash = ((hash << 5) + hash) + trace->kernel;
    hash = ((hash << 5) + hash) + trace->dtype;
    hash = ((hash << 5) + hash) + (int) trace->in_dims[0][0];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[0][1];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[0][2];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[0][3];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[1][0];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[1][1];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[1][2];
    hash = ((hash << 5) + hash) + (int) trace->in_dims[1][3];
    hash = ((hash << 5) + hash) + (int) trace->out_dim[0];
    hash = ((hash << 5) + hash) + (int) trace->out_dim[1];
    hash = ((hash << 5) + hash) + (int) trace->out_dim[2];
    hash = ((hash << 5) + hash) + (int) trace->out_dim[3];

    return hash % YAMI_TRACE_NODES;
}

// If trace is already there, update the counters otherwise insert a new node.
static void yami_trace_insert(yami_trace_node *traces_table[YAMI_TRACE_NODES],
                              yami_trace *trace) noexcept {
    const int idx = yami_trace_hash(trace);
    yami_trace_node *node = traces_table[idx];
    while (node != nullptr) {
        const yami_trace node_trace = node->el;
        if (node_trace.kernel == trace->kernel && node_trace.dtype == trace->dtype &&
            memcmp(node_trace.in_dims[0], trace->in_dims[0], YAMI_MAX_DIMS * sizeof(usize)) == 0 &&
            memcmp(node_trace.in_dims[1], trace->in_dims[1], YAMI_MAX_DIMS * sizeof(usize)) == 0 &&
            memcmp(node_trace.out_dim, trace->out_dim, YAMI_MAX_DIMS * sizeof(usize)) == 0) {
            break;
        }
        node = node->next;
    }

    if (node == nullptr) {
        // Allocate a new node for this trace
        node = (yami_trace_node *) malloc(sizeof(yami_trace_node));
        node->next = traces_table[idx];
        memcpy(&node->el, trace, sizeof(yami_trace));
        traces_table[idx] = node;
    } else {
        // Update the node with its new trace
        // TODO: verify that the measurement are properly initialized
        node->el.ref++;
        node->el.time_ms += trace->time_ms;
    }
}
#endif
// =============================================================================================================

// ============================================== Helper functions ==============================================
static void yami_tensor_set_dim(yami_tensor *x, const int n_dim,
                                const usize *dim) noexcept {
    x->n_dim = n_dim;
    // If n_dim is lower than YAMI_MAX_DIM then we need to pre-fill the beginning of the array with 1
    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        x->dim[i] = i < (YAMI_MAX_DIMS - n_dim) ? 1 : dim[i - (YAMI_MAX_DIMS - n_dim)];
    }

    // Compute the stride
    memset(x->stride, 0, YAMI_MAX_DIMS * sizeof(size));
    x->stride[YAMI_MAX_DIMS - 1] = 1;
    for (int i = YAMI_MAX_DIMS - 2; i >= 0; --i) {
        x->stride[i] = x->stride[i + 1] * x->dim[i + 1];
    }
}

static bool YAMI_PURE yami_can_broadcast(const yami_tensor *xa, const yami_tensor *xb,
                                         const int dims_to_check = YAMI_MAX_DIMS) noexcept {
    bool can_broadcast = true;
    for (int i = 0; i < dims_to_check && can_broadcast; ++i) {
        can_broadcast = xa->dim[i] == xb->dim[i] || xa->dim[i] == 1 || xb->dim[i] == 1;
    }

    return can_broadcast;
}

static yami_tensor *yami_alloc_result(yami_ctx *ctx, const yami_tensor *xa,
                                      const yami_tensor *xb, const char *label = "") noexcept {

    YAMI_ASSERT(yami_can_broadcast(xa, xb));

    usize dim[YAMI_MAX_DIMS];
    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        dim[i] = YAMI_MAX(xa->dim[i], xb->dim[i]);
    }

    const int n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);

    yami_tensor *res = yami_new_tensor(ctx, n_dim, &dim[YAMI_MAX_DIMS - n_dim], label);
    return res;
}

static yami_mem_buffer *yami_buffer_alloc(const usize nb) noexcept {
    const usize buff_size = YAMI_ALIGN(nb + sizeof(yami_mem_buffer), YAMI_PAGE_SIZE);

    yami_mem_buffer *buff = (yami_mem_buffer *) aligned_alloc(YAMI_PAGE_SIZE, buff_size);
    YAMI_ASSERT(buff != nullptr);

    buff->nb = buff_size - sizeof(yami_mem_buffer);
    buff->n_objs = 0;
    buff->last = nullptr;

    return buff;
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
            case YAMI_OP_GEMM: {
                YAMI_LOG_ERR("YAMI_OP_GEMM not implemented!");
                break;
            }
            case YAMI_OP_GEVM: {
                yami_gevm_f32(work->n,
                              work->k,
                              work->xa,
                              work->xb,
                              work->stride_b,
                              work->res,
                              nullptr
                );
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
// ==============================================================================================================

yami_ctx *yami_init(const yami_init_params params) noexcept {
    YAMI_ASSERT(params.nb > 0);

    yami_ctx *ctx = (yami_ctx *) malloc(sizeof(yami_ctx));
    YAMI_ASSERT(ctx != nullptr);

    // Set the default scope to GLOBAL
    ctx->scope = GLOBAL;

    usize scope_nb[YAMI_SCOPES];
    if (params.scratch_nb > 0) {
        scope_nb[GLOBAL] = params.nb;
        scope_nb[LOCAL] = (usize) ((f64) params.scratch_nb * 0.9);
        scope_nb[PRIVATE] = (usize) ((f64) params.scratch_nb * 0.1);
    } else {
        scope_nb[GLOBAL] = (usize) ((f64) params.nb * 0.8);
        scope_nb[LOCAL] = (usize) ((f64) params.nb * 0.1);
        scope_nb[PRIVATE] = (usize) ((f64) params.nb * 0.1);
    }

    // TODO: make sure we allocated enough memory in the private buffer for packed_a and packed_b

    ctx->buffers[GLOBAL] = yami_buffer_alloc(scope_nb[GLOBAL]);
    ctx->buffers[LOCAL] = yami_buffer_alloc(scope_nb[LOCAL]);
    ctx->buffers[PRIVATE] = yami_buffer_alloc(scope_nb[PRIVATE]);

    const int n_cpus = get_nprocs();
    int n_workers = params.n_workers;
    if (n_workers > n_cpus) {
        YAMI_LOG_INFO("n_workers=%d > n_cpus=%d, the actual number of workers will be limited to n_cpus", n_workers, n_cpus);
        n_workers = n_cpus;
    } else if (n_workers < 0) {
        n_workers = (int) (n_cpus * 0.5);
    }

    ctx->n_workers = n_workers;

    if (n_workers > 1) {
        ctx->workers = (yami_worker *) malloc((n_workers - 1) * sizeof(yami_worker));
        // If we can, bind threads to even cores
        const int restrict_even = n_workers <= (int) (n_cpus * 0.5) ? 2 : 1;
        for (int i = 1; i < n_workers; ++i) {
            yami_worker *worker = &ctx->workers[i-1];
            pthread_mutex_init(&worker->mtx, nullptr);
            pthread_cond_init(&worker->cond, nullptr);
            worker->has_work = false;
            pthread_create(&worker->id, nullptr, yami_worker_thread, worker);

            // Set affinity
            cpu_set_t cpu_set{};
            CPU_ZERO(&cpu_set);
            CPU_SET(i * restrict_even, &cpu_set);
            pthread_setaffinity_np(worker->id, sizeof(cpu_set_t), &cpu_set);
        }
    }

    // Set affinity for the main worker
    cpu_set_t cpu_set{};
    CPU_ZERO(&cpu_set);
    CPU_SET(0, &cpu_set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);

    YAMI_LOG_INFO("created new context %p: workers=%d GLOBAL=%ldMB LOCAL=%ldMB PRIVATE=%ldMB",
                  (void *) ctx, ctx->n_workers,
                  (usize) YAMI_B_TO_MB(ctx->buffers[GLOBAL]->nb),
                  (usize) YAMI_B_TO_MB(ctx->buffers[LOCAL]->nb),
                  (usize) YAMI_B_TO_MB(ctx->buffers[PRIVATE]->nb)
    );

    return ctx;
}

void yami_free(yami_ctx *ctx) noexcept {
    YAMI_LOG_DEBUG("freeing context %p GLOBAL=%ldMB LOCAL=%ldMB PRIVATE=%ldMB",
                   (void *) ctx,
                   (usize) YAMI_B_TO_MB(ctx->buffers[GLOBAL]->nb),
                   (usize) YAMI_B_TO_MB(ctx->buffers[LOCAL]->nb),
                   (usize) YAMI_B_TO_MB(ctx->buffers[PRIVATE]->nb)
    );

    free(ctx->buffers[GLOBAL]);
    free(ctx->buffers[LOCAL]);
    free(ctx->buffers[PRIVATE]);

    yami_clear_traces(ctx);

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

void yami_clear_ctx(yami_ctx *ctx) noexcept {
    yami_mem_buffer *buff = yami_scope_buffer();

    YAMI_LOG_DEBUG("clearing %s buffer of %p mem_size=%ldMB n_objs=%d",
                   YAMI_SCOPE[ctx->scope],
                   (void *)ctx,
                   (usize) YAMI_B_TO_MB(buff->nb),
                   buff->n_objs
    );

    buff->last = nullptr;
    buff->n_objs = 0;
}

void yami_set_scope(yami_ctx *ctx, const yami_scope scope) noexcept {
    YAMI_ASSERT(scope != PRIVATE);
    ctx->scope = scope;
}

usize yami_used_mem(const yami_ctx *ctx) noexcept {
    yami_mem_buffer *buff = yami_scope_buffer();
    return buff->last == nullptr ? 0 : buff->last->offset + buff->last->nb;
}

void yami_mem_usage(const yami_ctx *ctx) noexcept {
    yami_mem_buffer *buff = yami_scope_buffer();

    const usize used = yami_used_mem(ctx);

    YAMI_LOG_INFO("scope=%s n_objs=%d used memory %.2fMB out of %ldMB (%.2f%%)",
                  YAMI_SCOPE[ctx->scope],
                  buff->n_objs,
                  YAMI_B_TO_MB(used),
                  (usize) YAMI_B_TO_MB(buff->nb),
                  ((f64) (used) / (f64) (buff->nb)) * 100.
    );
}

// List all the traces and print them in a neat ASCII table
void yami_print_traces(yami_ctx *ctx) noexcept {
#if defined(YAMI_TRACE)
    const f64 trace_time = yami_timer();
    std::vector<yami_trace *> traces;

    for (int i = 0; i < YAMI_TRACE_NODES; ++i) {
        yami_trace_node *node = ctx->traces_table[i];
        while(node != nullptr) {
            traces.push_back(&node->el);
            node = node->next;
        }
    }
    // Sort by number of references
    std::sort(traces.begin(), traces.end(), [](const yami_trace *a, const yami_trace *b) {
       return a->ref > b->ref;
    });

    static constexpr int col_widths[YAMI_TRACE_COLUMNS] = {
        6,
        33,
        16,
        5,
        8,
        16,
        10
    };

    static constexpr const char *const col_labels[YAMI_TRACE_COLUMNS] = {
            "Kernel",
            "In",
            "Out",
            "Dtype",
            "#ref",
            "Time (ms)",
            "GFLOPS"
    };

    // Clear the console
    printf("\033[2J\033[H");

    // Header
    for (int i = 0; i < YAMI_TRACE_COLUMNS; ++i) {
        printf("+");
        for (int j = 0; j < col_widths[i]; ++j) printf("-");
    }
    printf("+\n");
    for (int i = 0; i < YAMI_TRACE_COLUMNS; ++i) {
        printf("|");
        printf("%-*s", col_widths[i], col_labels[i]);
    }
    printf("|\n");
    for (int i = 0; i < YAMI_TRACE_COLUMNS; ++i) {
        printf("+");
        for (int j = 0; j < col_widths[i]; ++j) printf("-");
    }
    printf("+\n");


    // Content
    char format_buff[256];
    usize global_flop = 0;
    const f64 elapsed_time = trace_time - ctx->trace_start_time;
    for (const auto trace : traces) {
        printf("|%-*s", col_widths[0], kernel_labels[trace->kernel]);
        snprintf(format_buff, col_widths[1], "[%ld,%ld,%ld,%ld] x [%ld,%ld,%ld,%ld]",
                 trace->in_dims[0][0], trace->in_dims[0][1], trace->in_dims[0][2], trace->in_dims[0][3],
                 trace->in_dims[1][0], trace->in_dims[1][1], trace->in_dims[1][2], trace->in_dims[1][3]
        );

        printf("|%-*s", col_widths[1], format_buff);

        snprintf(format_buff, col_widths[2], "[%ld,%ld,%ld,%ld]",
                 trace->out_dim[0], trace->out_dim[1], trace->out_dim[2], trace->out_dim[3]
        );

        printf("|%-*s", col_widths[2], format_buff);

        printf("|%-*s", col_widths[3], dtype_labels[trace->dtype]);
        printf("|%-*ld", col_widths[4], trace->ref);

        const f64 kernel_time_ms = trace->time_ms / (f64) trace->ref;
        snprintf(format_buff, col_widths[5], "%.1f (%.1f%%)",
                 kernel_time_ms,
                 // How much of the elapsed time this particular kernel took
                 trace->time_ms / (elapsed_time * 1e3) * 100
        );

        printf("|%-*s", col_widths[5], format_buff);
        printf("|%-*.2f|\n", col_widths[6], (f64) trace->flop / (kernel_time_ms * 1e6));
        global_flop += (trace->flop * trace->ref);
    }

    // Footer
    for (int i = 0; i < YAMI_TRACE_COLUMNS; ++i) {
        printf("+");
        for (int j = 0; j < col_widths[i]; ++j) printf("-");
    }
    printf("+\n");
    printf("Elapsed time:\t\t\t\t%.1fs\n", elapsed_time);
    printf("Number of floating point operations:\t%.1e (%.2f GFLOPS)\n", (f64) global_flop, (f64) global_flop / (elapsed_time * 1e9));

    // Try to correct the start time to not take into account the latency introduced by this logging function
    ctx->trace_start_time += (yami_timer() - trace_time);
#else
    YAMI_UNUSED(ctx);
#endif
}

void yami_clear_traces(yami_ctx *ctx) noexcept {
#if defined(YAMI_TRACE)
    yami_trace_node *current, *tmp;
    for (int i = 0; i < YAMI_TRACE_NODES; ++i) {
        current = ctx->traces_table[i];
        while (current != nullptr) {
            tmp = current;
            current = current->next;
            free(tmp);
        }
        ctx->traces_table[i] = nullptr;
    }

    ctx->trace_start_time = yami_timer();
#else
    YAMI_UNUSED(ctx);
#endif
}

yami_tensor *yami_new_tensor(yami_ctx *ctx, const int n_dim,
                             const usize *dim, const char *label,
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
        ne *= dim[i];

    usize mem_needed = sizeof(yami_tensor);
    if (data == nullptr) {
        mem_needed += ne * sizeof(f32);
    }
    yami_mem_buffer *buff = yami_scope_buffer();

    const usize last_offset = buff->last == nullptr ? 0 : buff->last->offset;
    const usize last_size = buff->last == nullptr ? 0 : buff->last->nb;
    const usize last_end = last_offset + last_size;

    if (mem_needed + sizeof(yami_obj) + last_end > buff->nb) {
        YAMI_LOG_ERR("can't allocate %.2fMB", YAMI_B_TO_MB(mem_needed));
        return nullptr;
    }

    // The actual buffer starts after the 'header' of the arena struct.
    yami_obj *new_obj = (yami_obj *) ((byte *) buff + last_end + sizeof(yami_mem_buffer));
    // The offset refers to the actual offset of the "contained" object which comes after
    // the yami_object "header".
    new_obj->offset = last_end + sizeof(yami_obj);
    new_obj->nb = mem_needed;

    buff->last = new_obj;
    buff->n_objs++;
    // Allocate the actual tensor
    yami_tensor *new_tensor = (yami_tensor *) ((byte *) buff + sizeof(yami_mem_buffer) + new_obj->offset);

    new_tensor->ne = ne;
    new_tensor->contiguous = true;
    if (data == nullptr) {
        new_tensor->data = (f32 *) (new_tensor + 1);
    } else {
        new_tensor->data = (f32 *) data;
    }

    strncpy(new_tensor->label, label, YAMI_LABEL_SIZE);

    yami_tensor_set_dim(new_tensor, n_dim, dim);

    YAMI_LOG_DEBUG("label=\"%s\" n_dim=%d extended_dim=[%ld, %ld, %ld, %ld] stride=[%ld, %ld, %ld, %ld] data=%p",
                   label, n_dim, new_tensor->dim[0], new_tensor->dim[1], new_tensor->dim[2],
                   new_tensor->dim[3], new_tensor->stride[0], new_tensor->stride[1], new_tensor->stride[2],
                   new_tensor->stride[3], data);

    return new_tensor;
}

yami_tensor *yami_tensor_1d(yami_ctx *ctx, const char *label,
                            const usize dim1) noexcept{
    const usize dim[YAMI_MAX_DIMS] = {dim1};
    return yami_new_tensor(ctx, 1, dim, label);
}

yami_tensor *yami_tensor_2d(yami_ctx *ctx, const char *label,
                            const usize dim1, const usize dim2) noexcept{
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2};
    return yami_new_tensor(ctx, 2, dim, label);
}

yami_tensor *yami_tensor_3d(yami_ctx *ctx, const char *label,
                            const usize dim1, const usize dim2,
                            const usize dim3) noexcept{
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, 3, dim, label);
}

yami_tensor *yami_tensor_4d(yami_ctx *ctx, const char *label,
                            const usize dim1, const usize dim2,
                            const usize dim3, const usize dim4) noexcept{
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, 4, dim, label);
}

yami_tensor *yami_view_1d(yami_ctx *ctx, yami_tensor *x,
                          const usize dim1, const usize offset) noexcept{
    YAMI_ASSERT(x->contiguous);
    const usize dim[YAMI_MAX_DIMS] = {dim1};
    return yami_new_tensor(ctx, 1, dim, x->label, &x->data[offset]);
}

yami_tensor *yami_view_2d(yami_ctx *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize offset) noexcept{
    YAMI_ASSERT(x->contiguous);
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2};
    return yami_new_tensor(ctx, 2, dim, x->label, &x->data[offset]);
}

yami_tensor *yami_view_3d(yami_ctx *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize dim3, const usize offset) noexcept{
    YAMI_ASSERT(x->contiguous);
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2, dim3};
    return yami_new_tensor(ctx, 3, dim, x->label, &x->data[offset]);
}

yami_tensor *yami_view_4d(yami_ctx *ctx, yami_tensor *x,
                          const usize dim1, const usize dim2,
                          const usize dim3, const usize dim4,
                          const usize offset) noexcept{
    YAMI_ASSERT(x->contiguous);
    const usize dim[YAMI_MAX_DIMS] = {dim1, dim2, dim3, dim4};
    return yami_new_tensor(ctx, 4, dim, x->label, &x->data[offset]);
}

yami_tensor *yami_reshape(yami_tensor *x, const int n_dims...) noexcept {
    YAMI_ASSERT(n_dims > 0 && n_dims <= YAMI_MAX_DIMS && x->contiguous);

    usize new_dim[YAMI_MAX_DIMS];

    std::va_list args;
    va_start(args, n_dims);
    for (int i = 0; i < n_dims; ++i) {
        const usize d = va_arg(args, usize);
        YAMI_ASSERT(d > 0);
        new_dim[i] = d;
    }
    va_end(args);


    usize new_ne = 1;
    for (int i = 0; i < n_dims; ++i)
        new_ne *= new_dim[i];

    YAMI_ASSERT(new_ne == x->ne);

    yami_tensor_set_dim(x, n_dims, new_dim);

    return x;
}

yami_tensor *yami_transpose(yami_ctx *, yami_tensor *x,
                            const int dim1, const int dim2) noexcept {
    YAMI_ASSERT(x->n_dim > 1);

    const usize d1 = yami_tensor_dim(x, dim1);
    const usize d2 = yami_tensor_dim(x, dim2);
    if (d1 == d2)
        return x;

    YAMI_ASSERT(d1 < YAMI_MAX_DIMS && d2 < YAMI_MAX_DIMS);

    const usize tmp_d1 = x->dim[d1];
    x->dim[d1] = x->dim[d2];
    x->dim[d2] = tmp_d1;

    const usize tmp_s1 = x->stride[d1];
    x->stride[d1] = x->stride[d2];
    x->stride[d2] = tmp_s1;

    // Checking whether x is contiguous is straightforward: check if the last dimension has stride 1
    // and the rest of them are in non-decreasing order.
    x->contiguous = x->stride[YAMI_MAX_DIMS - 1] == 1;
    for (size i = 0; i < YAMI_MAX_DIMS - 1 && x->contiguous; ++i)
        x->contiguous &= x->stride[i] >= x->stride[i + 1];

    return x;
}

yami_tensor *yami_contiguous(yami_ctx *ctx, yami_tensor *x) noexcept {
    if (x->contiguous)
        return x;

    yami_tensor *res = yami_new_like(x);

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        // Todo: blocking?
        res->data[yami_offset(res, 3)] = x->data[yami_offset(x, 3)];
    }
    
    return res;
}

yami_tensor *yami_lt_mask(yami_ctx *, yami_tensor *x,
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

yami_tensor *yami_mask_if(yami_ctx *, yami_tensor *x,
                          const yami_mask flag, const f32 val,
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

yami_tensor *yami_embed(yami_ctx *ctx, const yami_tensor *x,
                        const int *indexes, const usize n) noexcept {
    YAMI_ASSERT(x->n_dim == 2);
    
    const usize max_idx = x->dim[yami_tensor_dim(x, -2)];
    const usize emb_size = x->dim[yami_tensor_dim(x, -1)];
    const usize res_dim[2] = {n, emb_size};

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

yami_tensor *yami_rope(yami_ctx *, yami_tensor *x,
                       const usize n, const bool k_mode,
                       const usize start_idx) noexcept {
    // Todo:
    //  - in_place flag, for now it's always inplace

    YAMI_TENSOR_FIELDS(x, 3);

    YAMI_ASSERT(n <= d3_x);

    for (usize d0 = 0; d0 < d0_x; ++d0) {
        for (usize d1 = (k_mode ? 0 : start_idx); d1 < d1_x; ++d1) {
            const usize p = k_mode ? (start_idx + d1) : d1;
            for (usize d2 = 0; d2 < d2_x; ++d2) {
                for (usize d3 = 0; d3 < n; d3 += 2) {
                    const f32 theta = std::pow(10'000.f, (-((f32) d3)) / ((f32) n));
                    const f32 cos_theta = std::cos((f32) p * theta);
                    const f32 sin_theta = std::sin((f32) p * theta);

                    const usize offset = yami_offset(x, 3);
                    const f32 x0 = x->data[offset];
                    const f32 x1 = x->data[offset + 1];
                    x->data[offset] = x0 * cos_theta - x1 * sin_theta;
                    x->data[offset + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
    return x;
}

yami_tensor *yami_split(yami_ctx *ctx, const yami_tensor *x,
                        const usize n, const int offset,
                        const int dim) noexcept {
    const int dim_idx = yami_tensor_dim(x, dim);

    const usize x_dim_ne = x->dim[dim_idx];

    YAMI_ASSERT((x_dim_ne % n == 0) && (offset * n + n <= x_dim_ne));

    yami_shape_of(x);
    x_shape[dim_idx] = n;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, yami_shape(x));

    yami_dim_range ranges;

    for (int i = 0; i < YAMI_MAX_DIMS; ++i) {
        const usize di = x->dim[i];
        usize start = 0;
        usize stop = di;
        if (dim_idx == i) {
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

yami_tensor *yami_concat(yami_ctx *ctx, const yami_tensor *xa,
                         const yami_tensor *xb, const int dim) noexcept {
    const int dim_idx = yami_tensor_dim(xa, dim);
    YAMI_ASSERT(xa->n_dim == xb->n_dim && (unsigned) dim_idx < YAMI_MAX_DIMS);

    for (int i = 0; i < xa->n_dim; ++i) {
        if (i != dim_idx && xa->dim[i] != xb->dim[i]) {
            YAMI_LOG_ERR("cannot concatenate tensor of shape [%ld %ld %ld %ld] with tensor of shape [%ld %ld %ld %ld] along axis %d",
                         xb->dim[0], xb->dim[1], xb->dim[2], xb->dim[3],
                         xa->dim[0], xa->dim[1], xa->dim[2], xa->dim[3],
                         dim_idx);
            return nullptr;
        }
    }

    yami_shape_of(xa);
    xa_shape[dim_idx] += xb->dim[dim_idx];

    yami_tensor *res = yami_new_tensor(ctx, xa->n_dim, yami_shape(xa));

    switch (dim_idx) {
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
    YAMI_ASSERT(x != nullptr && res != nullptr);
    yami_tensor_set_dim(res, x->n_dim, x->dim);
    memcpy(res->data, x->data, res->ne * sizeof(f32));
}

// Compute C = A x B^T
// A = [m x k], B^T = [n x k]
yami_tensor *yami_matmul(yami_ctx *ctx, const yami_tensor *__restrict xa,
                         const yami_tensor *__restrict xb) noexcept {
    // Verify that the two matrices are at least 2-dimensional
    if (xa->n_dim < 2 || xb->n_dim < 2) {
        YAMI_LOG_FATAL("too few dimensions, use yami_mul to multiply 1D tensors");
    }

    const usize xa_n_rows = xa->dim[yami_tensor_dim(xa, -2)];
    const usize xa_n_cols = xa->dim[yami_tensor_dim(xa, -1)];
    const usize xb_n_rows = xb->dim[yami_tensor_dim(xb, -1)];
    const usize xb_n_cols = xb->dim[yami_tensor_dim(xb, -2)];

    if (xa_n_cols != xb_n_rows) {
        YAMI_LOG_FATAL("can't multiply (%ld, %ld) by (%ld, %ld)",
                       xa_n_rows, xa_n_cols,
                       xb_n_rows, xb_n_cols
        );
    }

    YAMI_ASSERT(yami_can_broadcast(xa, xb, YAMI_MAX_DIMS - 2));

    usize res_dim[YAMI_MAX_DIMS];
    res_dim[YAMI_MAX_DIMS - 2] = xa_n_rows;
    res_dim[YAMI_MAX_DIMS - 1] = xb_n_cols;
    for (int i = 0; i < YAMI_MAX_DIMS - 2; ++i)
        res_dim[i] = YAMI_MAX(xa->dim[i], xb->dim[i]);

    const int res_n_dim = YAMI_MAX(xa->n_dim, xb->n_dim);
    yami_tensor *res = yami_new_tensor(ctx, res_n_dim, &res_dim[YAMI_MAX_DIMS - res_n_dim]);
    yami_set_zero(res);

    YAMI_ENTER_PRIVATE_SCOPE(ctx);
    yami_mem_buffer *work_buff = yami_scope_buffer();

    YAMI_TENSOR_FIELDS(xa, 3);
    YAMI_TENSOR_FIELDS(xb, 3);
    YAMI_TENSOR_FIELDS(res, 2);
    YAMI_UNUSED(d2_xb), YAMI_UNUSED(d2_res);
    YAMI_UNUSED(d3_stride_xa), YAMI_UNUSED(d3_stride_xb);

    std::atomic_int progress{};

#if defined(YAMI_TRACE)
    yami_trace trace{xa->dim, xb->dim, res->dim,
                     2 * d2_xa * d3_xa * d3_xb,
                     d2_xa == 1 ? yami_kernel::GEVM : yami_kernel::GEMM,
                     yami_dtype::F32
    };
    trace.time_ms = yami_timer() * 1e3;
#endif
    for (usize d0 = 0; d0 < d0_res; ++d0) {
        for (usize d1 = 0; d1 < d1_res; ++d1) {
            // TODO: handle all the supported dtypes
            f32 *__restrict res_data = &res->data[yami_offset(res, 1)];
            const f32 *__restrict xa_data = &xa->data[yami_broadcast_offset(xa, 1)];
            const f32 *__restrict xb_data = &xb->data[yami_broadcast_offset(xb, 1)];

            // M = 1 so this is vector times matrix
            if (d2_xa == 1) {
                progress.store(1);

                // Split N by n_workers
                const usize n_work = xb_n_cols / ctx->n_workers;
                const usize leftover_n = xb_n_cols - (n_work * ctx->n_workers);

                // Enqueue tasks for the workers
                for (int i = 1; i < ctx->n_workers; ++i) {
                    yami_worker *worker = &ctx->workers[i - 1];
                    pthread_mutex_lock(&worker->mtx);

                    yami_task *t = &worker->task;
                    t->xa = xa_data;
                    t->xb = &xb_data[(i - 1) * n_work * d2_stride_xb];
                    t->res = &res_data[(i - 1) * n_work];
                    t->k = xa_n_cols;
                    t->n = n_work;
                    t->stride_b = d2_stride_xb;
                    t->progress_counter = &progress;
                    t->op = YAMI_OP_GEVM;
                    worker->has_work = true;

                    pthread_cond_signal(&worker->cond);
                    pthread_mutex_unlock(&worker->mtx);
                }

                // Do the leftover work on the main thread
                yami_gevm_f32(leftover_n, xa_n_cols,
                              xa_data,
                              &xb_data[(ctx->n_workers - 1) * n_work * d2_stride_xb],
                              d2_stride_xb,
                              &res_data[(ctx->n_workers - 1) * n_work],
                              nullptr
                );
                // Wait for the other threads
                while (progress.load() != ctx->n_workers)
                    ;
            } else {
                yami_gemm_f32(xa_n_rows, xb_n_cols, xa_n_cols,
                              xa_data, d2_stride_xa,
                              xb_data, d2_stride_xb,
                              res_data, d2_stride_res,
                              (byte *) work_buff + sizeof(yami_mem_buffer)
                );
            }
        }
    }
#if defined(YAMI_TRACE)
    trace.time_ms = yami_timer() * 1e3 - trace.time_ms;
    yami_trace_insert(ctx->traces_table, &trace);
#endif

    YAMI_EXIT_PRIVATE_SCOPE(ctx);
    return res;
}
// When performing vector operations on a tensor (such as add, multiply...) there are two possibilities:
//  - the last dims are equal                               --> sum over two equal tensors with size N
//  - the last dim of one tensor is 1 and the other is > 1  --> sum a constant to a tensor
yami_tensor *yami_add(yami_ctx *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dim[yami_tensor_dim(xa, -1)];
    const usize xb_1_ne = xb->dim[yami_tensor_dim(xb, -1)];
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

yami_tensor *yami_addc(yami_ctx *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);
    yami_internal_vec_addc(res->data, x->data, c, x->ne);
    return res;
}

yami_tensor *yami_sub(yami_ctx *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dim[yami_tensor_dim(xa, -1)];
    const usize xb_1_ne = xb->dim[yami_tensor_dim(xb, -1)];
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

yami_tensor *yami_subc(yami_ctx *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);
    yami_internal_vec_subc(res->data, x->data, c, x->ne);
    return res;
}

yami_tensor *yami_mul(yami_ctx *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dim[yami_tensor_dim(xa, -1)];
    const usize xb_1_ne = xb->dim[yami_tensor_dim(xb, -1)];
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

yami_tensor *yami_mulc(yami_ctx *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);
    yami_internal_vec_mulc(res->data, x->data, c, res->ne);
    return res;
}

yami_tensor *yami_div(yami_ctx *ctx, yami_tensor *xa,
                      const yami_tensor *xb, const bool in_place) noexcept {
    yami_tensor *res;
    if (!in_place) {
        res = yami_alloc_result(ctx, xa, xb);
    } else {
        YAMI_ASSERT(yami_can_broadcast(xa, xb));
        res = xa;
    }

    const usize xa_1_ne = xa->dim[yami_tensor_dim(xa, -1)];
    const usize xb_1_ne = xb->dim[yami_tensor_dim(xb, -1)];
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

yami_tensor *yami_divc(yami_ctx *ctx, yami_tensor *x,
                       const f32 c, const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);
    yami_internal_vec_divc(res->data, x->data, c, res->ne);
    return res;
}

yami_tensor *yami_gelu(yami_ctx *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    static const f32 sqrt_2_pi = std::sqrt(M_2_PIf);
    const f32 c = 0.044715f;

    yami_tensor *res = in_place ? x : yami_new_like(x);

    for (usize i = 0; i < res->ne; ++i) {
        const f32 val = x->data[i];
        res->data[i] = (0.5f * val) * (1.f + tanhf(sqrt_2_pi * (val + (c * val) * (val * val))));
    }

    return res;
}

yami_tensor *yami_swiglu(yami_ctx *ctx, yami_tensor *x,
                         const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);

    // silu(x) = x * sigma(x) where sigma(x) = 1 / (1 + e^(-x))
    for (usize i = 0; i < res->ne; ++i) {
        const f32 val = x->data[i];
        res->data[i] = val * (1.f / (1.f + std::exp(-val)));
    }

    return res;
}

yami_tensor *yami_sum(yami_ctx *ctx, const yami_tensor *x,
                      const int dim) noexcept {
    const int dim_idx = yami_tensor_dim(x, dim);

    yami_shape_of(x);
    x_shape[dim_idx] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, yami_shape(x));
    yami_set_zero(res);

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        const usize in_idx = yami_offset(x, 3);

        usize out_idx;

        if (dim_idx == 0) out_idx = ((d1 * d1_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_idx == 1) out_idx = ((d0 * d0_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_idx == 2) out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d3 * d3_stride_res);
        else out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d2 * d2_stride_res);

        res->data[out_idx] += x->data[in_idx];
    }

    return res;
}

yami_tensor *yami_mean(yami_ctx *ctx, const yami_tensor *x,
                       const int dim) noexcept {
    yami_tensor *res = yami_sum(ctx, x, dim);

    yami_mulc(ctx, res, 1.f / (f32) x->dim[yami_tensor_dim(x, dim)]);

    return res;
}

yami_tensor *yami_var(yami_ctx *ctx, yami_tensor *x,
                      const int dim) noexcept {
    YAMI_ENTER_PRIVATE_SCOPE(ctx);

    yami_tensor *x_mean_square = yami_square(ctx,
                                             yami_sub(ctx,
                                                      x,
                                                      yami_mean(ctx,
                                                                x,
                                                                dim
                                                      )
                                             ),
                                             true
    );

    YAMI_EXIT_PRIVATE_SCOPE(ctx);

    return yami_mean(ctx, x_mean_square, dim);
}

yami_tensor *yami_exp(yami_ctx *ctx, yami_tensor *x,
                      const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);

    for (usize i = 0; i < x->ne; ++i)
        res->data[i] = std::exp(x->data[i]);

    return res;
}

yami_tensor *yami_sqrt(yami_ctx *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);

    for (usize i = 0; i < x->ne; ++i)
        res->data[i] = std::sqrt(x->data[i]);

    return res;
}

yami_tensor *yami_rsqrt(yami_ctx *ctx, yami_tensor *x,
                       const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);

    for (usize i = 0; i < x->ne; ++i)
        res->data[i] = 1.f / std::sqrt(x->data[i]);

    return res;
}

yami_tensor *yami_square(yami_ctx *ctx, yami_tensor *x,
                         const bool in_place) noexcept {
    yami_tensor *res = in_place ? x : yami_new_like(x);

    for (usize i = 0; i < x->ne; ++i) {
        const f32 val = x->data[i];
        res->data[i] = val * val;
    }

    return res;
}

yami_tensor *yami_max(yami_ctx *ctx, const yami_tensor *x,
                      const int dim) noexcept {
    const int dim_idx = yami_tensor_dim(x, dim);

    yami_shape_of(x);
    x_shape[dim_idx] = 1;

    yami_tensor *res = yami_new_tensor(ctx, x->n_dim, yami_shape(x));

    for (usize i = 0; i < res->ne; ++i)
        res->data[i] = YAMI_MINUS_INF;

    YAMI_TENSOR_FIELDS(x, 3);
    YAMI_TENSOR_STRIDES(res, 3);

    yami_for(x, 3) {
        const usize in_idx = yami_offset(x, 3);

        usize out_idx;

        if (dim_idx == 0) out_idx = ((d1 * d1_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_idx == 1) out_idx = ((d0 * d0_stride_res) + (d2 * d2_stride_res)) + (d3 * d3_stride_res);
        else if (dim_idx == 2) out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d3 * d3_stride_res);
        else out_idx = ((d0 * d0_stride_res) + (d1 * d1_stride_res)) + (d2 * d2_stride_res);

        res->data[out_idx] = YAMI_MAX(res->data[out_idx], x->data[in_idx]);
    }

    return res;
}

yami_tensor *yami_softmax(yami_ctx *ctx, yami_tensor *x,
                          const int dim) noexcept {
    YAMI_ENTER_PRIVATE_SCOPE(ctx);

    yami_tensor *e_x = yami_exp(ctx,
                                yami_sub(ctx,
                                         x,
                                         yami_max(ctx,
                                                  x,
                                                  dim
                                         )
                                )
    );

    yami_tensor *sum_e_x = yami_sum(ctx, e_x, dim);

    YAMI_EXIT_PRIVATE_SCOPE(ctx);

    return yami_div(ctx,
                    e_x,
                    sum_e_x
    );
}

yami_tensor *yami_layer_norm(yami_ctx *ctx, const yami_tensor *w,
                             const yami_tensor *b, yami_tensor *x,
                             const f32 eps) noexcept {
    YAMI_ENTER_PRIVATE_SCOPE(ctx);

    yami_tensor *x_mean = yami_mean(ctx, x, -1);
    yami_tensor *x_var = yami_addc(ctx,
                                   yami_var(ctx,
                                            x,
                                            -1
                                   ),
                                   eps
    );

    yami_tensor *out = yami_div(ctx,
                                yami_sub(ctx,
                                         x,
                                         x_mean
                                ),
                                yami_sqrt(ctx,
                                          x_var
                                )
    );

    YAMI_EXIT_PRIVATE_SCOPE(ctx);

    return yami_add(ctx,
                    yami_mul(ctx,
                             out,
                             w
                    ),
                    b,
                    true
    );
}

yami_tensor *yami_rms_norm(yami_ctx *ctx, const yami_tensor *w,
                           yami_tensor *x, const f32 eps) noexcept {
    YAMI_ENTER_PRIVATE_SCOPE(ctx);

    yami_tensor *x_mean = yami_mean(ctx,
                                    yami_square(ctx,
                                                x,
                                                false
                                    ),
                                    -1);
    yami_tensor *out = yami_mul(ctx,
                                x,
                                yami_rsqrt(ctx,
                                           yami_addc(ctx,
                                                     x_mean,
                                                     eps
                                           )
                                )
    );

    YAMI_EXIT_PRIVATE_SCOPE(ctx);

    return yami_mul(ctx,
                    out,
                    w
    );
}