#include "yami2.h"

int main() {
    yami_context *ctx = yami_init(yami_context_init_params{
        .mem_size = 10,
        .scratch_mem_size = 0,
        .mem_buffer = nullptr,
        .scratch_mem_buffer = nullptr
    });

    YAMI_ASSERT(ctx != nullptr);
    yami_free(ctx);

    return EXIT_SUCCESS;
}