#include "yami_blas.h"
#include <cstring>
#include <atomic>
#include <pthread.h>
#include <sys/sysinfo.h> // get_nprocs()


enum yami_op : u8 {
    YAMI_OP_GEMM,
    YAMI_OP_GEVM,
    YAMI_OP_DONE,
};

struct yami_task {
    const f32 *__restrict a;
    const f32 *__restrict b;
    f32 *__restrict c;
    usize m, n, k;
    usize stride_a, stride_b, stride_c;
    // Counter shared between all the workers, used to determine whether they have finished or not.
    std::atomic_int *progress_counter;
    yami_op op;
};

struct yami_worker {
    // The current implementation uses a spinlock in the main thread (0th thread) to wait for its workers to complete
    // and a condition variable in said workers to wait for work.
    // Performance-wise this is a suboptimal choice as there will inevitably be some OS-related delay between when the main thread signals
    // and the worker actually receives that signal. This is not the case with a spinlock however the spinlock will cause
    // higher utilization (all the cores will be maxed-out until the inference is done) since the workers are created only once
    // and are utilized only to speed-up GEVMs and GEMMs, which means they are quite often idle.
    pthread_cond_t cond;
    pthread_mutex_t mtx;
    yami_task task;
    // Internal buffers used for packing (worker thread)
    void *__restrict packed_a, *__restrict packed_b;
    pthread_t id;
    bool has_work;
};

struct yami_blas_ctx {
    // Internal buffers used for packing (main thread)
    void *__restrict packed_a, *__restrict packed_b;
    yami_worker *workers;
    int n_workers;
};


using f32x8 = __m256;

#define rank1_8x8(A, B, idx)                                    \
    do {                                                        \
        const f32x8 beta_p = _mm256_loadu_ps(&(B)[(idx) * NR]); \
\
        /* Broadcast alpha_0 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 0]);   \
        gamma_0 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_0);   \
\
        /* Broadcast alpha_1 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 1]);   \
        gamma_1 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_1);   \
\
        /* Broadcast alpha_2 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 2]);   \
        gamma_2 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_2);   \
\
        /* Broadcast alpha_3 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 3]);   \
        gamma_3 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_3);   \
\
        /* Broadcast alpha_4 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 4]);   \
        gamma_4 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_4);   \
\
        /* Broadcast alpha_5 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 5]);   \
        gamma_5 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_5);   \
\
        /* Broadcast alpha_6 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 6]);   \
        gamma_6 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_6);   \
\
        /* Broadcast alpha_7 */                                 \
        alpha_pj = _mm256_broadcast_ss(&(A)[(idx) * MR + 7]);   \
        gamma_7 = _mm256_fmadd_ps(alpha_pj, beta_p, gamma_7);   \
    } while(0)

// ============================================== Ukernel ==============================================
static YAMI_INLINE void yami_ukernel_8x8_f32(const usize k,
                                             const f32 *__restrict a,
                                             const f32 *__restrict b,
                                             f32 *__restrict c, const usize stride_c) noexcept {
    f32x8 gamma_0 = _mm256_loadu_ps(&c[0 * stride_c]);
    f32x8 gamma_1 = _mm256_loadu_ps(&c[1 * stride_c]);
    f32x8 gamma_2 = _mm256_loadu_ps(&c[2 * stride_c]);
    f32x8 gamma_3 = _mm256_loadu_ps(&c[3 * stride_c]);
    f32x8 gamma_4 = _mm256_loadu_ps(&c[4 * stride_c]);
    f32x8 gamma_5 = _mm256_loadu_ps(&c[5 * stride_c]);
    f32x8 gamma_6 = _mm256_loadu_ps(&c[6 * stride_c]);
    f32x8 gamma_7 = _mm256_loadu_ps(&c[7 * stride_c]);

    f32x8 alpha_pj;

    const usize pb = (k / 4) * 4;
    for (usize p = 0; p < pb; p += 4) {
        rank1_8x8(a, b, p + 0);
        rank1_8x8(a, b, p + 1);
        rank1_8x8(a, b, p + 2);
        rank1_8x8(a, b, p + 3);
    }

    for (usize p = pb; p < k; ++p) {
        rank1_8x8(a, b, p);
    }

    _mm256_storeu_ps(&c[0 * stride_c], gamma_0);
    _mm256_storeu_ps(&c[1 * stride_c], gamma_1);
    _mm256_storeu_ps(&c[2 * stride_c], gamma_2);
    _mm256_storeu_ps(&c[3 * stride_c], gamma_3);
    _mm256_storeu_ps(&c[4 * stride_c], gamma_4);
    _mm256_storeu_ps(&c[5 * stride_c], gamma_5);
    _mm256_storeu_ps(&c[6 * stride_c], gamma_6);
    _mm256_storeu_ps(&c[7 * stride_c], gamma_7);
}
// =====================================================================================================

// ============================================== Packing ==============================================
static YAMI_INLINE void yami_packA_f32(const usize m, const usize k,
                                       const f32 *__restrict a, const usize stride_a,
                                       f32 *__restrict packed_a) noexcept {
    for (usize i = 0; i < m; i += MR) {
        const usize ib = YAMI_MIN(m - i, MR);

        for (usize p = 0; p < k; ++p) {
            for (usize ii = 0; ii < ib; ++ii) *packed_a++ = a[(i + ii) * stride_a + p];
            for (usize ii = ib; ii < MR; ++ii) *packed_a++ = 0.f;
        }
    }
}

static YAMI_INLINE void yami_packB_f32(const usize k, const usize n,
                                       const f32 *__restrict b, const usize stride_b,
                                       f32 *__restrict packed_b) noexcept {
    for (usize j = 0; j < n; j += NR) {
        const usize jb = YAMI_MIN(n - j, NR);

        for (usize p = 0; p < k; ++p) {
            for (usize jj = 0; jj < jb; ++jj) *packed_b++ = b[(j + jj) * stride_b + p];
            for (usize jj = jb; jj < NR; ++jj) *packed_b++ = 0.f;
        }
    }
}
// =====================================================================================================

// ============================================== Helper functions ==============================================
static YAMI_INLINE void yami_internal_gemm_f32(const usize m, const usize n, const usize k,
                                               const f32 *__restrict a, const usize stride_a,
                                               const f32 *__restrict b, const usize stride_b,
                                               f32 *__restrict c, const usize stride_c,
                                               f32 *__restrict packed_a, f32 *__restrict packed_b) noexcept {
    alignas(64) f32 packed_c[MR * NR];

    // 5th loop
    for (usize j = 0; j < n; j += NC) {
        const usize jb = YAMI_MIN(n - j, NC);

        // 4th loop
        for (usize p = 0; p < k; p += KC) {
            const usize pb = YAMI_MIN(k - p, KC);

            // Pack B
            yami_packB_f32(pb, jb, &b[j * stride_b + p], stride_b, packed_b);

            // 3rd loop
            for (usize i = 0; i < m; i+= MC) {
                const usize ib = YAMI_MIN(m - i, MC);

                // Pack A
                yami_packA_f32(ib, pb, &a[i * stride_a + p], stride_a, packed_a);

                for (usize jj = 0; jj < jb; jj += NR) {
                    const usize jjb = YAMI_MIN(jb - jj, NR);
                    for (usize ii = 0; ii < ib; ii += MR) {
                        const usize iib = YAMI_MIN(ib - ii, MR);

                        // Prefetch the current micro-panel of C
                        _mm_prefetch(&c[(i + ii + 0) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 1) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 2) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 3) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 4) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 5) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 6) * stride_c + (j + jj)], _MM_HINT_T0);
                        _mm_prefetch(&c[(i + ii + 7) * stride_c + (j + jj)], _MM_HINT_T0);

                        if (iib == MR && jjb == NR) {
                            yami_ukernel_8x8_f32(pb,
                                                 &packed_a[ii * pb],
                                                 &packed_b[jj * pb],
                                                 &c[(i + ii) * stride_c + (j + jj)],
                                                 stride_c
                            );
                        } else {
                            memset(packed_c, 0, MR * NR * sizeof(f32));
                            yami_ukernel_8x8_f32(pb,
                                                 &packed_a[ii * pb],
                                                 &packed_b[jj * pb],
                                                 packed_c,
                                                 NR
                            );
                            for (usize iii = 0; iii < iib; ++iii) {
                                for (usize jjj = 0; jjj < jjb; ++jjj) c[(i + ii + iii) * stride_c + (j + jj + jjj)] += packed_c[iii * NR + jjj];
                            }
                        }
                    }
                }
            }
        }
    }
}

static YAMI_INLINE void yami_internal_gevm_f32(const usize n, const usize k,
                                               const f32 *__restrict a,
                                               const f32 *__restrict b, const usize stride_b,
                                               f32 *__restrict c) noexcept {
    for (usize j = 0; j < n; ++j) {
        for (usize p = 0; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
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
                yami_internal_gemm_f32(work->m,
                                       work->n,
                                       work->k,
                                       work->a,
                                       work->stride_a,
                                       work->b,
                                       work->stride_b,
                                       work->c,
                                       work->stride_c,
                                       (f32 *) self->packed_a,
                                       (f32 *) self->packed_b
                );
                work->progress_counter->fetch_add(1);
                break;
            }
            case YAMI_OP_GEVM: {
                yami_internal_gevm_f32(work->n,
                                       work->k,
                                       work->a,
                                       work->b,
                                       work->stride_b,
                                       work->c
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

yami_blas_ctx *yami_blas_init(int n_workers) noexcept {
    yami_blas_ctx *ctx = (yami_blas_ctx *) malloc(sizeof(yami_blas_ctx));

    const int n_cpus = get_nprocs();
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
            yami_worker *worker = &ctx->workers[i - 1];

            worker->packed_a = aligned_alloc(64, KC * MC * sizeof(f32));
            worker->packed_b = aligned_alloc(64, KC * NC * sizeof(f32));

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

    ctx->packed_a = aligned_alloc(64, MC * KC * sizeof(f32));
    ctx->packed_b = aligned_alloc(64, NC * KC * sizeof(f32));

    return ctx;
}

void yami_blas_free(yami_blas_ctx *ctx) noexcept {
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

            free(worker->packed_a), free(worker->packed_b);
        }

        free(ctx->workers);
    }
    free(ctx->packed_a), free(ctx->packed_b);
    free(ctx);
}

int yami_blas_num_workers(const yami_blas_ctx *ctx) noexcept {
    return ctx->n_workers;
}

void yami_gemm_f32(yami_blas_ctx *ctx,
                   usize m, usize n, usize k,
                   const f32 *__restrict a, usize stride_a,
                   const f32 *__restrict b, usize stride_b,
                   f32 *__restrict c, usize stride_c) noexcept {
    std::atomic_int progress{1};

    const usize n_work = n / ctx->n_workers;
    const usize leftover_n = n - n_work * (ctx->n_workers - 1);

    for (int i = 1; i < ctx->n_workers; ++i) {
        yami_worker *worker = &ctx->workers[i - 1];
        pthread_mutex_lock(&worker->mtx);

        yami_task *t = &worker->task;
        t->a = a;
        t->b = &b[(i - 1) * n_work * stride_b];
        t->c = &c[(i - 1) * n_work];
        t->k = k;
        t->n = n_work;
        t->m = m;
        t->stride_a = stride_a;
        t->stride_b = stride_b;
        t->stride_c = stride_c;
        t->progress_counter = &progress;
        t->op = YAMI_OP_GEMM;
        worker->has_work = true;

        pthread_cond_signal(&worker->cond);
        pthread_mutex_unlock(&worker->mtx);
    }

    // Do the leftover work on the main thread
    yami_internal_gemm_f32(m, leftover_n, k,
                           a,
                           stride_a,
                           &b[(ctx->n_workers - 1) * n_work * stride_b],
                           stride_b,
                           &c[(ctx->n_workers - 1) * n_work],
                           stride_c,
                           (f32 *) ctx->packed_a,
                           (f32 *) ctx->packed_b
    );

    // Wait for the other threads
    while (progress.load() != ctx->n_workers)
        ;
}

void yami_gevm_f32(yami_blas_ctx *ctx,
                   const usize n, const usize k,
                   const f32 *__restrict a,
                   const f32 *__restrict b, const usize stride_b,
                   f32 *__restrict c) noexcept {
    std::atomic_int progress{1};

    // Split N by n_workers
    const usize n_work = n / ctx->n_workers;
    // Remember that n_workers is at least 1, also the main thread should have at least
    // n_work rows of B to work on
    const usize leftover_n = n - (n_work * (ctx->n_workers - 1));

    // Enqueue tasks for the workers
    for (int i = 1; i < ctx->n_workers; ++i) {
        yami_worker *worker = &ctx->workers[i - 1];
        pthread_mutex_lock(&worker->mtx);

        yami_task *t = &worker->task;
        t->a = a;
        t->b = &b[(i - 1) * n_work * stride_b];
        t->c = &c[(i - 1) * n_work];
        t->k = k;
        t->n = n_work;
        t->stride_b = stride_b;
        t->progress_counter = &progress;
        t->op = YAMI_OP_GEVM;
        worker->has_work = true;

        pthread_cond_signal(&worker->cond);
        pthread_mutex_unlock(&worker->mtx);
    }

    // Do the leftover work on the main thread
    yami_internal_gevm_f32(leftover_n, k,
                           a,
                           &b[(ctx->n_workers - 1) * n_work * stride_b],
                           stride_b,
                           &c[(ctx->n_workers - 1) * n_work]
    );

    // Wait for the other threads
    while (progress.load() != ctx->n_workers)
        ;
}
