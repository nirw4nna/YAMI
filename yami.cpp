#include "yami.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <cmath>

// "YAMIF" in little endian
constexpr static uint32_t yami_magic = 0x001FAA1A;

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

yami_tensor *yami_new_tensor2d(uint32_t d1, uint32_t d2, std::string_view label) noexcept {
    uint32_t dims[2] = {d1, d2};
    return new yami_tensor(2, dims, label);
}

yami_tensor *copy_of(const yami_tensor *t) noexcept {
    auto *cpy = new yami_tensor(t->n_dim, t->dimensions, t->label);
    memcpy(cpy->data, t->data, t->ne * sizeof(float));
    return cpy;
}

std::unordered_map<std::string, yami_tensor *> yami_load_from_file(std::string_view yami_file, void *hparams, int hparams_size) {
    FILE *f = fopen(yami_file.data(), "rb");
    if (f == nullptr) {
        fprintf(stderr, "Error opening %s\n", yami_file.data());
        exit(EXIT_FAILURE);
    }

    // check the header
    uint32_t head = 0;
    size_t n;
    n = fread(&head, 1, 3, f);
    if (n != 3 || yami_magic != head) {
        fprintf(stderr, "Wrong header %08X\n", head);
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

    const auto hp_size = static_cast<int>(*(const uint16_t *)ptr);
    if (hp_size != hparams_size) {
        fprintf(stderr, "Error reading the hyperparameters, expected size is %d but got %d\n", hparams_size, hp_size);
        exit(EXIT_FAILURE);
    }
    ptr += 2;
    memcpy(hparams, ptr, hparams_size);
    ptr += hparams_size;

    const auto n_tensors = static_cast<int>(*(const uint16_t *)ptr);
    ptr += 2;

    std::unordered_map<std::string, yami_tensor *> tensors;
    for (int i = 0; i < n_tensors; ++i) {
        const auto label_size = static_cast<int>(*(const uint16_t *)ptr);
        ptr += 2;
        const auto *label = (const char *)ptr;
        ptr += label_size + 1;
        const auto n_dim = static_cast<int>(ptr[0]);
        ptr++;
        const auto dim = (const uint32_t *) ptr;
        ptr += yami_max_dimensions * sizeof(uint32_t);
        const auto data_size = static_cast<uint64_t>((*(const uint64_t *)ptr) * sizeof(float));
        ptr += 8;

        auto *t = new yami_tensor(n_dim, dim, label);
        memcpy(t->data, ptr, data_size);
        ptr += data_size;
        tensors[t->label] = t;
    }
    munmap(data, file_size);
    return tensors;
}

// first, very much naive, implementation to be used as a baseline
void yami_mat_mul(float *__restrict out, const float *__restrict xa,
                         const float *__restrict xb, const int nra, const int nca, const int ncb) noexcept {
    for (int i = 0; i < nra; ++i) {
        for (int k = 0; k < nca; ++k) {
            for (int j = 0; j < ncb; ++j) {
                out[i*ncb + j] += xa[i*nca + k] * xb[k*ncb + j];
            }
        }
    }
}

void yami_add(float *__restrict xa, const float *__restrict xb, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] += xb[i];
    }
}

void yami_add(float *__restrict xa, const float c, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] += c;
    }
}


void yami_mul(float *__restrict xa, const float *__restrict xb, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] *= xb[i];
    } 
}

void yami_mul(float *__restrict xa, const float c, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] *= c;
    }
}

void yami_div(float *__restrict xa, const float c, size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        xa[i] /= c;
    }
}

void yami_tanh(float *xa, const size_t n) noexcept {
    for (size_t i = 0; i < n; ++i) {
        const auto e_p = std::exp(xa[i]);
        const auto e_n = std::exp(-xa[i]);
        xa[i] = (e_p - e_n) / (e_p + e_n);
    }
}

void yami_softmax(float *xa, const size_t n) noexcept {
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

void yami_get_embeddings(yami_tensor *dst_emb, const yami_tensor *emb_table, const int *ctx, const int ctx_size) noexcept {
    YAMI_ASSERT(emb_table->n_dim == 2);
    const auto emb_size = emb_table->dimensions[1];
    YAMI_ASSERT(dst_emb->ne >= (ctx_size * emb_size));

    for (int i = 0; i < ctx_size; ++i) {
        YAMI_ASSERT(ctx[i] < static_cast<int>(emb_table->dimensions[0]));
        memcpy(dst_emb->data + (i*emb_size),
               emb_table->data + (ctx[i]*emb_size),
               emb_size * sizeof(float)
        );
    }
}
// Accumulate all the elements in the tensor x
// Note: it's probably better to accumulate over a given dimension
float yami_sum(const yami_tensor *x) noexcept {
    float acc = 0.f;
    for (size_t i = 0; i < x->ne; ++i) acc += x->data[i];
    return acc;
}

void yami_norm(yami_tensor *x, const yami_tensor *w, const yami_tensor *b) noexcept {
    constexpr static auto eps = 1.e-5f;

    const auto x_elements = x->ne;
    YAMI_ASSERT(x_elements == w->ne && x_elements == b->ne);
    
    const auto n_inv = 1.f / static_cast<float>(x_elements);
    const auto mean = yami_sum(x) * n_inv;
    auto var_scaled = 0.f;
    for (size_t i = 0; i < x_elements; ++i) var_scaled += ((x->data[i] - mean) * (x->data[i] - mean));
    var_scaled *= n_inv;
    var_scaled = std::sqrt(var_scaled + eps);

    yami_add(x->data, -mean, x_elements);
    yami_div(x->data, var_scaled, x_elements);
    yami_mul(x->data, w->data, x_elements);
    yami_add(x->data, b->data, x_elements);
}