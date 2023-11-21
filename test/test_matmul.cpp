#include "../yami2.h"
#include <x86intrin.h>

static yami_tensor *random2d(yami_context *ctx, const char *label,
                             const size d1, const size d2) noexcept {
    yami_tensor *tensor = yami_tensor_2d(ctx, label, d1, d2);

    for (size i = 0; i < d1; ++i) {
        for (size j = 0; j < d2; ++j) {
            tensor->data[i*d2 + j] = (f32) (rand()) / (f32) RAND_MAX;
        }
    }

    return tensor;
}

int main() {
    const size block_size = 3096;
    const int n_iter = 10;

    yami_context *ctx = yami_init(yami_context_init_params{
            6,
            1024 * 1024 * 1024 * 5LL,
            0,
            nullptr, nullptr
    });

    yami_tensor *a = random2d(ctx, "a", block_size, block_size);
    yami_tensor *b = random2d(ctx, "b", block_size, block_size);
    yami_tensor *res = yami_tensor_2d(ctx, "a x b", block_size, block_size);

    yami_matmul(ctx, a, b, res);
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

    const size flops = n_iter * (2 * block_size) * (block_size * block_size);
    const size n_el = n_iter * block_size * block_size;
    printf("Executing %ld flops took %ld clock cycles (%.3f ms), %.1f clocks/el (%.2f Gflops)\n",
           flops, mul_clocks, mul_time * 1e3, (f64) mul_clocks / (f64) (n_el), (f64) flops / (mul_time * 1e9)
    );

    yami_free(ctx);
    return 0;
}