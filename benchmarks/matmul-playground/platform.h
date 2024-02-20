#pragma once

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#define MIN(x, y)  ((x) < (y) ? (x) : (y))

#if defined(__cplusplus)
extern "C" {
#endif

    using size = ptrdiff_t;
    using usize = size_t;
    using f32 = float;
    using f64 = double;

    static inline f64 now() {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (f64) ts.tv_sec + (f64) ts.tv_nsec * 1.e-9;
    }

#if defined(__cplusplus)
}
#endif