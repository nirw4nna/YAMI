#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <cstring>
#include <vector>
#include <ctime>

#ifdef __AVX2__
#   include <immintrin.h>
#endif

#ifdef YAMI_DEBUG
#   define YAMI_ASSERT(x) \
       do{                \
            if (!(x)) {      \
                fprintf(stderr, "YAMI_ASSERT: %s:%d %s\n", __FILE__, __LINE__, #x); \
                exit(EXIT_FAILURE);                                                 \
            }                   \
       } while(0)
#else
#   define YAMI_ASSERT(x) ((void)0)
#endif

constexpr static int yami_max_dimensions = 4;
constexpr static int yami_max_label = 64; // todo: remove

static inline int64_t yami_get_time_us() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec * 1'000'000ULL) + static_cast<int64_t>(ts.tv_nsec / 1'000ULL);
}

static inline int64_t yami_get_time_ms() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec * 1'000ULL) + static_cast<int64_t>(ts.tv_nsec / 1'000'000ULL);
}


struct yami_tensor {
    yami_tensor(int n_dim, const uint32_t *dim, std::string_view label);
    ~yami_tensor();

    int n_dim;
    size_t ne;
    uint32_t dimensions[yami_max_dimensions]{}; // todo: raname in shape
    const std::string label;
    float *data;
};

// Helper class that just reads a yami file and returns an array of tensors
extern std::vector<yami_tensor *> yami_load_from_file(std::string_view yami_file);

// out = xa @ xb
extern void yami_mat_mul(yami_tensor *out, const yami_tensor *xa, const yami_tensor *xb) noexcept;
// xa += xb
extern void yami_add(float *__restrict xa, const float *__restrict xb, size_t n) noexcept;
// xa = tanh(xa)
extern void yami_tanh(float *xa, size_t n) noexcept;
extern void yami_softmax(float *xa, size_t n) noexcept;
extern void yami_transpose(yami_tensor *t) noexcept;
