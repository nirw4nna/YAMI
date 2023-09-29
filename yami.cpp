#include "yami.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <cmath>

constexpr static uint8_t yami_magic[3] = {0x1A, 0xAA, 0x1F};

yami_tensor::yami_tensor(int n_dim, const uint32_t *dim, std::string_view label) : n_dim(n_dim), label(label) {
    ::memcpy(this->dimensions, dim, n_dim * sizeof(uint32_t));

    size_t n_elements = 1;
    for (int i = 0; i < n_dim; ++i) n_elements *= dim[i];
#ifdef YAMI_DEBUG
    printf("yami: new tensor \"%s\" with %u dimensions and %ld elements\n", this->label.c_str(), n_dim, n_elements);
#endif
    this->ne = n_elements;
    this->data = (float *) calloc(ne, sizeof(float));
}

yami_tensor::~yami_tensor() {
    free(this->data);
}

std::vector<yami_tensor *> yami_load_from_file(std::string_view yami_file) {
    FILE *f = fopen(yami_file.data(), "rb");
    if (f == nullptr) {
        fprintf(stderr, "Error opening %s\n", yami_file.data());
        exit(EXIT_FAILURE);
    }

    // check the header
    uint8_t buff[3];
    size_t n;
    n = fread(buff, 1, 3, f);
    if (n != 3 || ::memcmp(yami_magic, buff, 3) != 0) {
        fprintf(stderr, "Wrong header\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
#ifdef YAMI_DEBUG
    printf("Loading %ld bytes from model file \"%s\"\n", file_size, yami_file.data());
#endif
    int fd = fileno(f);
    void *data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "Error mapping %s to memory\n", yami_file.data());
        exit(EXIT_FAILURE);
    }
    const auto *ptr = (const uint8_t *)data;
    ptr += 3; // skip the magic number

    const auto n_tensors = static_cast<int>(*(const uint16_t *)ptr);
    ptr += 2;

    std::vector<yami_tensor *> tensors(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        const auto *label = (const char *)ptr;
        ptr += yami_max_label + 1;
        const auto n_dim = static_cast<int>(ptr[0]);
        ptr++;
        const auto dim = (const uint32_t *) ptr;
        ptr += yami_max_dimensions * sizeof(uint32_t);
        const auto data_size = static_cast<size_t>((*(const uint32_t *)ptr) * sizeof(float));
        ptr += 4;

        auto *t = new yami_tensor(n_dim, dim, label);
        memcpy(t->data, ptr, data_size);
        ptr += data_size;
        tensors[i] = t;
    }
    munmap(data, file_size);
    return tensors;
}

// first, very much naive, implementation to be used as a baseline
void yami_mat_mul(yami_tensor *out, const yami_tensor *xa, const yami_tensor *xb) noexcept {
    const auto rows_a = xa->dimensions[0];
    const auto cols_a = xa->dimensions[1];
    const auto cols_b = xb->dimensions[1];

    for (uint32_t i = 0; i < rows_a; ++i) {
        for (uint32_t k = 0; k < cols_a; ++k) {
            for (uint32_t j = 0; j < cols_b; ++j) {
                out->data[i*cols_b + j] += xa->data[i*cols_a + k] * xb->data[k*cols_b + j];
            }
        }
    }
}

void yami_add(float *__restrict xa, const float *__restrict xb, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] += xb[i];
    }
}

void yami_tanh(float *xa, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const auto e_p = std::exp(xa[i]);
        const auto e_n = std::exp(-xa[i]);
        xa[i] = (e_p - e_n) / (e_p + e_n);
    }
}

void yami_softmax(float *xa, size_t n) noexcept {
    auto den = 0.f;

    for (size_t i = 0; i < n; ++i) {
        den += std::exp(xa[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        xa[i] = std::exp(xa[i]) / den;
    }
}

void yami_transpose(yami_tensor *t) noexcept {
    YAMI_ASSERT(t->n_dim == 2);

    // todo: implement a basic follow the cycle in-place algorithm
    // Another possible solution to increase performance could be to read from t in row-major order
    // and write back to tmp using non-temporal write instructions to prevent cache lines from being evicted.
    auto *tmp_buff = (float *) alloca(t->ne * sizeof(float));

    const uint32_t n_rows = t->dimensions[0];
    const uint32_t n_cols = t->dimensions[1];
    for (uint32_t j = 0; j < n_cols; ++j) {
        for (uint32_t i = 0; i < n_rows; ++i) {
            // Iterate the tensor in column-major order
            tmp_buff[i+j*n_rows] = t->data[i*n_cols + j];
        }
    }
    memcpy(t->data, tmp_buff, t->ne * sizeof(float));

    const auto last_idx = t->n_dim - 1;
    const auto mid_point = static_cast<int>(t->n_dim * 0.5);
    for (int i = 0; i < mid_point; ++i) {
        const auto el = t->dimensions[i];
        t->dimensions[i] = t->dimensions[last_idx - i];
        t->dimensions[last_idx - i] = el;
    }
}