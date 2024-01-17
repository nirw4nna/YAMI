#include "yami.h"
#include <x86intrin.h>

static yami_tensor *random2d(yami_context *ctx, const char *label,
                             const usize d1, const usize d2) noexcept {
    yami_tensor *tensor = yami_tensor_2d(ctx, label, d1, d2);

    for (usize i = 0; i < d1; ++i) {
        for (usize j = 0; j < d2; ++j) {
            tensor->data[i*d2 + j] = (f32) (rand()) / (f32) RAND_MAX;
        }
    }

    return tensor;
}
//(1024, 12, 11, 11) X (1024, 12, 11, 64)
// (12, 768) x (1, 1, 768, 50257)

int main() {
    const usize n_rows_a = 12;
    const usize n_cols_a = 768;
    const usize n_rows_b = 768;
    const usize n_cols_b = 50257;

    const int n_iter = 10;

    yami_context *ctx = yami_init(yami_context_init_params{
            12,
            1024 * 1024 * 1024 * 5LL,
            0
    });

    yami_tensor *a = random2d(ctx, "a", n_rows_a, n_cols_a);
    yami_tensor *b = random2d(ctx, "b", n_rows_b, n_cols_b);
    yami_tensor *res = yami_tensor_2d(ctx, "a x b", n_rows_a, n_cols_b);

    yami_matmul(ctx, a, b, res);

    // 211 GFlops theoretical peak performance
    // 2n^3 floating ops

    const f64 start_time = yami_timer();
    const u64 start_clock = _rdtsc();
    for (int i = n_iter; i > 0; --i) {
        yami_matmul(ctx, a, b, res);
    }
    const u64 mul_clocks = _rdtsc() - start_clock;
    const f64 mul_time = yami_timer() - start_time;

    // For each element we do products = n_cols_a and sums = p-1 so that's 2p-1 FLOPS / el
    // the total number of elements in the result is n_rows_a * n_cols_b
    const usize flops = n_iter * (2 * n_cols_a - 1) * (n_rows_a * n_cols_b);
    const usize n_el = n_iter * n_rows_a * n_cols_b;
    printf("Executing %ld flops took %ld clock cycles (%.3f ms), %.1f clocks/el (%.2f Gflops)\n",
           flops, mul_clocks, mul_time * 1e3, (f64) mul_clocks / (f64) (n_el), (f64) flops / (mul_time * 1e9)
    );

    yami_free(ctx);
    return 0;
}