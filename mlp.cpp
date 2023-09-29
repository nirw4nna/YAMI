#include "yami.h"
#include <random>


// todo: define hyperparams in a better way
constexpr static int emb_size = 14;
constexpr static int ctx_size = 8;
constexpr static int hidden_size = 256;
constexpr static int vocab_size = 27;

constexpr static char tok_stop = '.';


struct MLPModel {
    MLPModel() : rd(), rng(rd()) {
        const uint32_t emb_dim[] = {1, emb_size * ctx_size};
        this->input_embeddings = new yami_tensor(2, emb_dim, "in_emb");

        const uint32_t layer1_out_dim[] = {1, static_cast<uint32_t>(hidden_size)};
        this->layer1_out = new yami_tensor(2, layer1_out_dim, "layer1_out");

        const uint32_t layer2_out_dim[] = {1, static_cast<uint32_t>(vocab_size)};
        this->layer2_out = new yami_tensor(2, layer2_out_dim, "layer2_out");
        
        std::vector<yami_tensor*> model_tensors = yami_load_from_file("examples/mlp/model.yami");
        YAMI_ASSERT(model_tensors.size() == 5);

        this->emb = model_tensors[0];
        this->layer1_w = model_tensors[1];
        this->layer1_b = model_tensors[2];
        this->layer2_w = model_tensors[3];
        this->layer2_b = model_tensors[4];
        // In a linear layer the weights must be transposed!
        yami_transpose(this->layer1_w);
        yami_transpose(this->layer2_w);

        // Build token lookup table
        for (int i = 1; i < vocab_size; ++i) {
            this->tok_lookup[i] = static_cast<char>('a' + (i-1));
        }
        this->tok_lookup[0] = tok_stop;
    }

    ~MLPModel() {
        delete this->input_embeddings;
        delete this->emb;
        delete this->layer1_w;
        delete this->layer1_b;
        delete this->layer2_w;
        delete this->layer2_b;
        delete this->layer1_out;
        delete this->layer2_out;
    }

    std::string generate() noexcept {
        auto *ctx = (int *) alloca(ctx_size * sizeof(int));
        memset(ctx, 0, ctx_size * sizeof(int));

        const auto layer1_dim = this->layer1_out->dimensions[1];
        const auto layer2_dim = this->layer2_out->dimensions[1];
        std::vector<float> probs(layer2_dim);
        std::string out;
        while (true) {
            // 1. compute embeddings
            this->compute_embeddings(ctx);
            // 2. first layer: emb @ w + b
            yami_mat_mul(this->layer1_out, this->input_embeddings, this->layer1_w);
            yami_add(this->layer1_out->data, this->layer1_b->data, layer1_dim);
            // 3. tanh
            yami_tanh(this->layer1_out->data, layer1_dim);
            // 4. second layer: l1_out @ w + b
            yami_mat_mul(this->layer2_out, this->layer1_out, this->layer2_w);
            yami_add(this->layer2_out->data, this->layer2_b->data, layer2_dim);
            // now we have the logits in layer2_out, and we apply softmax to get the probabilities
            // 5. softmax
            yami_softmax(this->layer2_out->data, layer2_dim);

            memcpy(probs.data(), this->layer2_out->data, layer2_dim * sizeof(float));

            std::discrete_distribution<> dist(probs.begin(), probs.end());

            const int next_tok_idx = dist(rng);
            const char next_tok = this->tok_lookup[next_tok_idx];
            if (next_tok == tok_stop) {
                break;
            }
            out.push_back(next_tok);
            memmove(ctx, &ctx[1], (ctx_size - 1) * sizeof(int));
            ctx[ctx_size-1] = next_tok_idx;
        }
        return out;
    }

    inline void compute_embeddings(const int *ctx) const noexcept {
        for (int i = 0; i < ctx_size; ++i) {
            memcpy(this->input_embeddings->data + (i*emb_size),
                   this->emb->data + (ctx[i]*emb_size),
                   emb_size * sizeof(float)
            );
        }
    }

    yami_tensor *input_embeddings;
    yami_tensor *emb;
    yami_tensor *layer1_w;
    yami_tensor *layer1_b;
    yami_tensor *layer2_w;
    yami_tensor *layer2_b;
    // Buffers for intermediate computations
    yami_tensor *layer1_out;
    yami_tensor *layer2_out;

    char tok_lookup[vocab_size]{};

    std::random_device rd;
    std::mt19937 rng;
};

int main() {
    MLPModel model;

    printf("Sampling from the model:\n");

//    const auto start = yami_get_time_ms();
    for (int i = 0; i < 10; ++i) {
        printf("%s\n", model.generate().c_str());
    }
//    const auto stop = yami_get_time_ms();
//    printf("Generating 10 samples from the model took: %ld ms\n", stop-start);
}