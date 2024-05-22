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
    const f32 *__restrict xa;
    const f32 *__restrict xb;
    f32 *__restrict res;
    usize m, n, k;
    usize stride_b, stride_c;
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
    // and are utilized only (at least for now) to speed-up GEVM which means they are quite often idle.
    // TODO: validate if this is actually still true
    pthread_cond_t cond;
    pthread_mutex_t mtx;
    yami_task task;
    pthread_t id;
    bool has_work;
};

struct yami_blas_ctx {
    // Internal buffers used for packing
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
                                               const f32 *__restrict a,
                                               const f32 *__restrict b,
                                               f32 *__restrict c, const usize stride_c) noexcept {
    alignas(32) f32 packed_c[MR * NR];

    for (usize j = 0; j < n; j += NR) {
        const usize jb = YAMI_MIN(n - j, NR);
        for (usize i = 0; i < m; i += MR) {
            const usize ib = YAMI_MIN(m - i, MR);

            // Prefetch the current micro-panel of C
            _mm_prefetch(&c[(i + 0) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 1) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 2) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 3) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 4) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 5) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 6) * stride_c + j], _MM_HINT_T0);
            _mm_prefetch(&c[(i + 7) * stride_c + j], _MM_HINT_T0);

            if (ib == MR && jb == NR) {
                yami_ukernel_8x8_f32(k, &a[i * k], &b[j * k], &c[i * stride_c + j], stride_c);
            } else {
                memset(packed_c, 0, MR * NR * sizeof(f32));
                yami_ukernel_8x8_f32(k, &a[i * k], &b[j * k], packed_c, NR);
                for (usize ii = 0; ii < ib; ++ii) {
                    for (usize jj = 0; jj < jb; ++jj) c[(i + ii) * stride_c + (j + jj)] += packed_c[ii * NR + jj];
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
                                       work->xa,
                                       work->xb,
                                       work->res,
                                       work->stride_c
                );
                work->progress_counter->fetch_add(1);
                break;
            }
            case YAMI_OP_GEVM: {
                yami_internal_gevm_f32(work->n,
                                       work->k,
                                       work->xa,
                                       work->xb,
                                       work->stride_b,
                                       work->res
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

    // Allocate the packing buffers
    // TODO: handle different dtypes each with a different alignment?
    // TODO: try different alignments
    ctx->packed_a = aligned_alloc(32, MC * KC * sizeof(f32));
    ctx->packed_b = aligned_alloc(32, NC * KC * sizeof(f32));

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
                   const usize m, const usize n, const usize k,
                   const f32 *__restrict a, const usize stride_a,
                   const f32 *__restrict b, const usize stride_b,
                   f32 *__restrict c, const usize stride_c) noexcept {
    f32 *__restrict packed_a = (f32 *) ctx->packed_a;
    f32 *__restrict packed_b = (f32 *) ctx->packed_b;

    std::atomic_int progress{};

    // 5th loop
    for (usize j = 0; j < n; j += NC) {
        const usize jb = YAMI_MIN(n - j, NC);

        const usize jb_work = (jb / NR) / ctx->n_workers * NR;
        const usize leftover_jb = jb - (jb_work * (ctx->n_workers - 1));

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

                progress.store(1);
                for (int w = 1; w < ctx->n_workers; ++w) {
                    yami_worker *worker = &ctx->workers[w - 1];
                    pthread_mutex_lock(&worker->mtx);

                    yami_task *t = &worker->task;
                    t->xa = packed_a;
                    t->xb = &packed_b[(w - 1) * jb_work * pb];
                    t->res = &c[i * stride_c + j + ((w - 1) * jb_work)];
                    t->m = ib;
                    t->n = jb_work;
                    t->k = pb;
                    t->stride_c = stride_c;
                    t->progress_counter = &progress;
                    t->op = YAMI_OP_GEMM;
                    worker->has_work = true;

                    pthread_cond_signal(&worker->cond);
                    pthread_mutex_unlock(&worker->mtx);
                }

                // Do the leftover work on the main thread
                yami_internal_gemm_f32(ib, leftover_jb, pb,
                                       packed_a,
                                       &packed_b[(ctx->n_workers - 1) * jb_work * pb],
                                       &c[i * stride_c + j + ((ctx->n_workers - 1) * jb_work)],
                                       stride_c
                );

                // Single-threaded
//                yami_internal_gemm_f32(ib, jb, pb,
//                                       packed_a,
//                                       packed_b,
//                                       &c[i * stride_c + j],
//                                       stride_c
//                );

                // Wait for the other threads
                while (progress.load() != ctx->n_workers)
                    ;
            }
        }
    }
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
        t->xa = a;
        t->xb = &b[(i - 1) * n_work * stride_b];
        t->res = &c[(i - 1) * n_work];
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

// The objective here is to compute the dot product of two vectors A, B of K elements.

// rC = load32(rCPtr)
// rIdx = 0
//
// loop:
//  rA = load32(rAPtr)
//  rB = load32(rBPtr)
//  rAB = rA * rB
//  rC = rC + rAB
//  rAPtr = rAPtr + 4
//  rBPtr = rBPtr + 4
//  rIdx = rIdx + 1
//  if rIdx < K goto loop

// Idea of reduce using AVX2:
//
//  const __m128 left  = _mm256_extractf128_ps(acc, 1);
//  const __m128 right = _mm256_castps256_ps128(acc);
//  const __m128 x128  = _mm_add_ps(left, right);
//  const __m128 x64   = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
//  const __m128 x32   = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
//  return  _mm_cvtss_f32(x32);


static YAMI_INLINE f32 reduce(const f32 *__restrict a,
                              const f32 *__restrict b) noexcept {
    // Load alpha and beta (Latency=7, TH~0.5)
    const f32x8 alpha_p = _mm256_loadu_ps(a);
    const f32x8 beta_pj = _mm256_loadu_ps(b);

    // Multiply alpha and beta element-wise acc = [a0*b0, .., a7*b7] (Latency=4, TH=0.5)
    const __m256 acc = _mm256_mul_ps(alpha_p, beta_pj);

    // Accumulate:

    // Extract the upper 4 floats [a4*b4, a5*b5, a6*b6, a7*b7] (Latency=4, TH=1)
    const __m128 upper = _mm256_extractf128_ps(acc, 1);
    // Cast the acc to __m128 to extract the lower 4 floats [a0*b0, a1*b1, a2*b2, a3*b3] (0 latency)
    const __m128 lower = _mm256_castps256_ps128(acc);

    // Sum [a4*b4 + a0*b0, a5*b5 + a1*b1, a6*b6 + a2*b2, a7*b7 + a3*b3] (Latency=4, TH=0.5)
    const __m128 partial_2 = _mm_add_ps(upper, lower);

    // Now that we have `partial_2` to get to the next step we just need to add the first two elements of partial_2
    // with its last two elements. To do so we can use `_mm_movehl_ps(partial_sum_128, partial_sum_128)` this will give us [a6*b6 + a2*b2, a7*b7 + a3*b3, a6*b6 + a2*b2, a7*b7 + a3*b3]
    // (Latency=1, TH=1).

    // Sum [a4*b4 + a0*b0, a5*b5 + a1*b1, a6*b6 + a2*b2, a7*b7 + a3*b3] + [a6*b6 + a2*b2, a7*b7 + a3*b3, a6*b6 + a2*b2, a7*b7 + a3*b3] =
    // [(a4*b4+a0*b0) + (a6*b6+a2*b2), (a5*b5+a1*b1) + (a7*b7+a3*b3), (a6*b6+a2*b2) + (a6*b6+a2*b2), (a7*b7+a3*b3) + (a7*b7+a3*b3)]
    // (Latency=4, TH=0.5)
    const __m128 partial_4 = _mm_add_ps(partial_2, _mm_movehl_ps(partial_2, partial_2));

    // 0x01 should be enough to shuffle, we just care about the first element (Latency=1, TH=0.5)
    // (Latency=4, TH=0.5)
    return _mm_cvtss_f32(_mm_add_ss(partial_4, _mm_shuffle_ps(partial_4, partial_4, 0x01)));
}

//// Load alpha and beta (Latency=7, TH~0.5)
//const f32x8 alpha_p = _mm256_loadu_ps(&a[p]);
//const f32x8 beta_pj = _mm256_loadu_ps(&b[j * stride_b + p]);
//
//// Multiply alpha and beta element-wise acc = [a0*b0, .., a7*b7] (Latency=4, TH=0.5)
//const __m256 a_b = _mm256_mul_ps(alpha_p, beta_pj);
//
//// Accumulate:
//const __m256 partial_2 = _mm256_hadd_ps(a_b, a_b);
//const __m256 partial_4 = _mm256_hadd_ps(partial_2, partial_2);
//c[j] = (c[j] + _mm256_cvtss_f32(partial_4)) + _mm_cvtss_f32(_mm256_extractf128_ps(partial_4, 1));

void yami_gevm_f32_simd(yami_blas_ctx *ctx,
                        usize n, usize k,
                        const f32 *__restrict a,
                        const f32 *__restrict b, usize stride_b,
                        f32 *__restrict c) noexcept {

    const usize pb = (k / 64) * 64;

    for (usize j = 0; j < n; ++j) {

        f32 acc_0 = 0.f, acc_1 = 0.f, acc_2 = 0.f, acc_3 = 0.f,
            acc_4 = 0.f,acc_5 = 0.f, acc_6 = 0.f, acc_7 = 0.f;
        for (usize p = 0; p < pb; p += 64) {
            _mm_prefetch(&a[p + 4 * 8], _MM_HINT_T0);
            _mm_prefetch(&a[p + 6 * 8], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 4 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 6 * 8)], _MM_HINT_T0);

            acc_0 += reduce(&a[p + 0 * 8], &b[j * stride_b + (p + 0 * 8)]);
            acc_1 += reduce(&a[p + 1 * 8], &b[j * stride_b + (p + 1 * 8)]);
            acc_2 += reduce(&a[p + 2 * 8], &b[j * stride_b + (p + 2 * 8)]);
            acc_3 += reduce(&a[p + 3 * 8], &b[j * stride_b + (p + 3 * 8)]);

            _mm_prefetch(&a[p + 8 * 8], _MM_HINT_T0);
            _mm_prefetch(&a[p + 10 * 8], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 8 * 8)], _MM_HINT_T0);
            _mm_prefetch(&b[j * stride_b + (p + 10 * 8)], _MM_HINT_T0);

            acc_4 += reduce(&a[p + 4 * 8], &b[j * stride_b + (p + 4 * 8)]);
            acc_5 += reduce(&a[p + 5 * 8], &b[j * stride_b + (p + 5 * 8)]);
            acc_6 += reduce(&a[p + 6 * 8], &b[j * stride_b + (p + 6 * 8)]);
            acc_7 += reduce(&a[p + 7 * 8], &b[j * stride_b + (p + 7 * 8)]);
        }

        c[j] += (acc_0 + acc_1) + (acc_2 + acc_3) + (acc_4 + acc_5) + (acc_6 + acc_7);

        for (usize p = pb; p < k; ++p) {
            c[j] += a[p] * b[j * stride_b + p];
        }
    }
}
