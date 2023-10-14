#include "yami.h"
#include <cstdio>
#include <vector>
#include <cmath>


struct gpt2_hparams {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t emb_size;
    uint32_t block_size;
    uint32_t vocab_size;
};

struct transformer_block {
    yami_tensor *ln_1_w;
    yami_tensor *ln_1_b;

    // attention
    yami_tensor *c_attn_w;
    yami_tensor *c_attn_b;
    yami_tensor *c_attn_proj_w;
    yami_tensor *c_attn_proj_b;

    yami_tensor *c_fc_w;
    yami_tensor *c_fc_b;
    yami_tensor *c_proj_w;
    yami_tensor *c_proj_b;

    yami_tensor *ln_2_w;
    yami_tensor *ln_2_b;
};

struct gpt2_model {
    gpt2_hparams hparams;
    yami_tensor *wpe;
    yami_tensor *wte;

    // transformer blocks
    std::vector<transformer_block> h;

    yami_tensor *ln_f_w;
    yami_tensor *ln_f_b;
    yami_tensor *lm_head_w;

    // todo: find a better way to cleanup the context...
    std::unordered_map<std::string, yami_tensor *> tensors;
};

static void load_model(gpt2_model *model, std::string_view model_file) noexcept {
    gpt2_hparams hp{};
    auto loaded_tensors = yami_load_from_file(model_file, &hp,
                                              sizeof(gpt2_hparams));
    model->hparams = hp;

    model->wte = loaded_tensors["transformer.wte.weight"];
    model->wpe = loaded_tensors["transformer.wpe.weight"];
    const auto n_layers = static_cast<int>(hp.n_layers);
    model->h.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        transformer_block *block = &model->h[i];
        block->ln_1_w = loaded_tensors["transformer.h." + std::to_string(i) + ".ln_1.weight"];
        block->ln_1_b = loaded_tensors["transformer.h." + std::to_string(i) + ".ln_1.bias"];
        block->c_attn_w = loaded_tensors["transformer.h." + std::to_string(i) + ".attn.c_attn.weight"];
        block->c_attn_b = loaded_tensors["transformer.h." + std::to_string(i) + ".attn.c_attn.bias"];
        block->c_attn_proj_w = loaded_tensors["transformer.h." + std::to_string(i) + ".attn.c_proj.weight"];
        block->c_attn_proj_b = loaded_tensors["transformer.h." + std::to_string(i) + ".attn.c_proj.bias"];
        block->ln_2_w = loaded_tensors["transformer.h." + std::to_string(i) + ".ln_2.weight"];
        block->ln_2_b = loaded_tensors["transformer.h." + std::to_string(i) + ".ln_2.bias"];
        block->c_fc_w = loaded_tensors["transformer.h." + std::to_string(i) + ".mlp.c_fc.weight"];
        block->c_fc_b = loaded_tensors["transformer.h." + std::to_string(i) + ".mlp.c_fc.bias"];
        block->c_proj_w = loaded_tensors["transformer.h." + std::to_string(i) + ".mlp.c_proj.weight"];
        block->c_proj_b = loaded_tensors["transformer.h." + std::to_string(i) + ".mlp.c_proj.bias"];

        // Weights in the attention layer have to be transposed since OpenAI used Conv1D instead of linear layers
//        yami_transpose(block->c_attn_proj_w);
//        yami_transpose(block->c_attn_w);
//        yami_transpose(block->c_fc_w);
//        yami_transpose(block->c_proj_w);
    }
    model->ln_f_w = loaded_tensors["transformer.ln_f.weight"];
    model->ln_f_b = loaded_tensors["transformer.ln_f.bias"];
    model->lm_head_w = loaded_tensors["lm_head.weight"];

    model->tensors = loaded_tensors;
}

static void generate(gpt2_model *model, std::vector<int> &tok,
                     int max_tokens=50, float temp=1.f) {
    // Append up to max_tokens to tok or until we get and 'end of text' token.
    const auto block_size = static_cast<int>(model->hparams.block_size);
    const auto emb_size = static_cast<int>(model->hparams.emb_size);
    const float attn_norm = 1.f / std::sqrt(static_cast<float>(emb_size) / static_cast<float>(model->hparams.n_heads));

    // todo: put in the model
    auto *tok_emb = yami_new_tensor2d(max_tokens, emb_size, "tok_emb");
    auto *pos_emb = yami_new_tensor2d(max_tokens, emb_size, "pos_emb");

    // Create an "arange", use directly max_tokens, so we just have to initialize it once
    auto *pos = (int *) alloca(max_tokens * sizeof(int));
    for (int i = 0; i < max_tokens; ++i) pos[i] = i;

    const int max_ctx_size = std::min(static_cast<int>(tok.size()) + max_tokens, static_cast<int>(block_size));

    auto *tmp_q = yami_new_tensor2d(max_ctx_size, model->hparams.emb_size, "tmp_q");
    auto *tmp_k = yami_new_tensor2d(max_ctx_size, model->hparams.emb_size, "tmp_k");
    auto *tmp_v = yami_new_tensor2d(max_ctx_size, model->hparams.emb_size, "tmp_v");
    auto *attn_out = yami_new_tensor2d(max_ctx_size, model->hparams.emb_size, "attn_out");

    for (int i = 0; i < max_tokens; ++i) {
        const auto n_tok = static_cast<int>(tok.size());
        if (n_tok > block_size) {
            tok.resize(block_size);
        }

        yami_get_embeddings(tok_emb, model->wte, tok.data(), n_tok);
        yami_get_embeddings(pos_emb, model->wpe, pos, n_tok);

        // tok_emb += pos_emb
        yami_add(tok_emb->data, pos_emb->data, tok_emb->ne);
        // alias
        yami_tensor *x = tok_emb;
        yami_tensor *x_old = nullptr;
        for (auto &block : model->h) {
            // 1a. copy of X
            // 1b. layer norm 1 x
            // 2a. attention x
            // 2b. x = old_x + x
            // 3a. copy of x
            // 3b layer norm 2 x
            // 4. x @ c_fc_w + c_fc_b
            // 5. GELU
            // 6a. x @ c_proj_w
            // 6b. x = old_x + x
            x_old = copy_of(x);
            yami_norm(x, block.ln_1_w, block.ln_1_b);
            yami_mat_mul(tmp_q->data, x->data, block.c_attn_w->data,
                         static_cast<int>(n_tok),
                         static_cast<int>(emb_size),
                         static_cast<int>(emb_size)
            );
            yami_add(tmp_q->data, block.c_attn_b->data, emb_size);

            yami_mat_mul(tmp_k->data, x->data, &block.c_attn_w->data[emb_size*emb_size],
                         static_cast<int>(n_tok),
                         static_cast<int>(emb_size),
                         static_cast<int>(emb_size)
            );
            yami_add(tmp_k->data, &block.c_attn_b->data[emb_size], emb_size);

            yami_mat_mul(tmp_v->data, x->data, &block.c_attn_w->data[2*emb_size*emb_size],
                         static_cast<int>(n_tok),
                         static_cast<int>(emb_size),
                         static_cast<int>(emb_size)
            );
            yami_add(tmp_v->data, &block.c_attn_b->data[2*emb_size], emb_size);

            yami_transpose(tmp_k);
            yami_mat_mul(attn_out->data, tmp_q->data, tmp_k->data, n_tok, emb_size, n_tok);
            yami_mul(attn_out->data, attn_norm, n_tok*emb_size);

        }

    }
}

int main() {
    using namespace std::literals;

    gpt2_model gpt2;
    load_model(&gpt2, "examples/gpt2/gpt2.yami");

    constexpr static auto prompt = "Javascript is a programming language"sv;
    std::vector<int> tok {1, 2, 3};
    generate(&gpt2, tok);
    // todo:
    //  - encode the prompt
    //  - inference
    //  - decode the result

    for (const auto &kv : gpt2.tensors)
        delete kv.second;

    return EXIT_SUCCESS;
}