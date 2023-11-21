#include "yami2.h"
#include "yami_utils.h"
#include <memory>
#include <cstring>
#include <cmath>
#include <random>

struct gpt2_hparams {
    u32 n_layers;
    u32 n_heads;
    u32 emb_size;
    u32 block_size;
    u32 vocab_size;
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
    yami_context *ctx;
    std::unique_ptr<yami_bpe_tokenizer> tokenizer;

    yami_tensor *wpe;
    yami_tensor *wte;

    // transformer blocks
    std::vector<transformer_block> h;

    yami_tensor *ln_f_w;
    yami_tensor *ln_f_b;
    yami_tensor *lm_head_w;

    std::random_device rd;
    std::mt19937 rng;
    std::vector<f32> probs;

    gpt2_hparams hparams;
    u8 pad[4];
};


static void gpt2_load_model(yami_context *ctx, gpt2_model *model, const char *yami_file) noexcept {
    gpt2_hparams hp{};
    yami_model ym{};
    ym.hparams = &hp;

    yami_load_model(ctx, &ym, yami_file);

    model->ctx = ctx;
    model->hparams = *((gpt2_hparams *) ym.hparams);
    model->tokenizer = std::make_unique<yami_bpe_tokenizer>(ym.vocab, ym.encoder);

    model->wte = ym.tensors["transformer.wte.weight"];
    model->wpe = ym.tensors["transformer.wpe.weight"];

    model->h.resize(hp.n_layers);
    for (u32 i = 0; i < hp.n_layers; ++i) {
        transformer_block *block = &model->h[i];
        block->ln_1_w = ym.tensors["transformer.h." + std::to_string(i) + ".ln_1.weight"];
        block->ln_1_b = ym.tensors["transformer.h." + std::to_string(i) + ".ln_1.bias"];
        block->c_attn_w = ym.tensors["transformer.h." + std::to_string(i) + ".attn.c_attn.weight"];
        block->c_attn_b = ym.tensors["transformer.h." + std::to_string(i) + ".attn.c_attn.bias"];
        block->c_attn_proj_w = ym.tensors["transformer.h." + std::to_string(i) + ".attn.c_proj.weight"];
        block->c_attn_proj_b = ym.tensors["transformer.h." + std::to_string(i) + ".attn.c_proj.bias"];
        block->ln_2_w = ym.tensors["transformer.h." + std::to_string(i) + ".ln_2.weight"];
        block->ln_2_b = ym.tensors["transformer.h." + std::to_string(i) + ".ln_2.bias"];
        block->c_fc_w = ym.tensors["transformer.h." + std::to_string(i) + ".mlp.c_fc.weight"];
        block->c_fc_b = ym.tensors["transformer.h." + std::to_string(i) + ".mlp.c_fc.bias"];
        block->c_proj_w = ym.tensors["transformer.h." + std::to_string(i) + ".mlp.c_proj.weight"];
        block->c_proj_b = ym.tensors["transformer.h." + std::to_string(i) + ".mlp.c_proj.bias"];
    }

    model->ln_f_w = ym.tensors["transformer.ln_f.weight"];
    model->ln_f_b = ym.tensors["transformer.ln_f.bias"];
    model->lm_head_w = ym.tensors["lm_head.weight"];

    model->rng = std::mt19937(model->rd());
}

static std::vector<int> generate(gpt2_model& model, std::vector<int>&& tok) {
    constexpr static int max_tokens = 5;

    std::vector<int> generated(tok);
    yami_context *scratch_ctx = yami_ctx_scratch(model.ctx);

    std::vector<int> pos;
    f64 start_t;
    for (int i = 0; i < max_tokens; ++i) {
        yami_clear_ctx(scratch_ctx);

        YAMI_ASSERT(generated.size() < model.hparams.block_size);

        start_t = yami_timer();
        pos.resize(generated.size());
        for (int j = 0; j < (int) generated.size(); ++j) {
            pos[j] = j;
        }

        yami_tensor *tok_emb = yami_embed(scratch_ctx, model.wte, generated.data(), (size) generated.size());
        yami_tensor *pos_emb = yami_embed(scratch_ctx, model.wpe, pos.data(), (size) pos.size());

        yami_tensor *x = yami_add(scratch_ctx, tok_emb, pos_emb, true);

        yami_tensor *cur;
        const size attn_shape[yami_max_dims] = {model.hparams.block_size, (size) generated.size(),
                                                model.hparams.n_heads, model.hparams.emb_size / model.hparams.n_heads};

        const static f32 att_scale = 1.f / std::sqrt((f32) attn_shape[3]);

        int head = 0;
        for (const auto &block : model.h) {
            YAMI_LOG_INFO("head %d", head);
            head++;

            cur = yami_layer_norm(scratch_ctx, block.ln_1_w, block.ln_1_b, x, false);
            cur = yami_add(scratch_ctx,
                           yami_matmul(scratch_ctx, cur, block.c_attn_w),
                           block.c_attn_b,
                           true);


            yami_tensor *q = yami_new_tensor(scratch_ctx, 4, attn_shape, "q");
            yami_tensor *k = yami_new_tensor(scratch_ctx, 4, attn_shape, "k");
            yami_tensor *v = yami_new_tensor(scratch_ctx, 4, attn_shape, "v");

            const size ne = q->ne;
            memcpy(q->data, cur->data,ne * sizeof(f32));
            memcpy(k->data, &cur->data[ne],ne * sizeof(f32));
            memcpy(v->data, &cur->data[2*ne],ne * sizeof(f32));


            yami_tensor *q_t = yami_contiguous(scratch_ctx,
                                               yami_transpose(scratch_ctx, q, 1, 2)
            );
            yami_tensor *k_t = yami_contiguous(scratch_ctx,yami_transpose(scratch_ctx,
                                                                          yami_transpose(scratch_ctx, k, 1, 2),
                                                                          -2, -1)
            );
            yami_tensor *v_t = yami_contiguous(scratch_ctx,
                                               yami_transpose(scratch_ctx, v, 1, 2)
            );

            cur = yami_matmul(scratch_ctx, q_t, k_t);
            yami_mulc(scratch_ctx, cur, att_scale);
            yami_lt_mask(scratch_ctx, cur, yami_neg_inf);

            yami_softmax(scratch_ctx, cur, -1);

            yami_tensor *out = yami_matmul(scratch_ctx, cur, v_t);
            out = yami_contiguous(scratch_ctx,
                                  yami_transpose(scratch_ctx, out, 1, 2)
            );
            yami_reshape(out, 3, model.hparams.block_size, generated.size(), model.hparams.emb_size);

            yami_tensor *out_proj = yami_add(
                    scratch_ctx,
                    yami_matmul(scratch_ctx, out, block.c_attn_proj_w),
                    block.c_attn_proj_b,
                    true
            );

            // x = x + self.attn(self.ln_1(x))
            yami_add(scratch_ctx, x, out_proj, true);

            // x = x + m.c_proj(m.act(m.c_fc(self.ln_2(x))))
            cur = yami_layer_norm(scratch_ctx, block.ln_2_w, block.ln_2_b, x, false);
            cur = yami_add(scratch_ctx,
                           yami_matmul(scratch_ctx, cur, block.c_fc_w),
                           block.c_fc_b,
                           true
            );
            yami_gelu(scratch_ctx, cur);

            x = yami_add(scratch_ctx,
                         x,
                         yami_add(
                                 scratch_ctx,
                                 yami_matmul(scratch_ctx, cur, block.c_proj_w),
                                 block.c_proj_b,
                                 true
                         ),
                         true
            );
        }
        yami_layer_norm(scratch_ctx, model.ln_f_w, model.ln_f_b, x);

        yami_tensor *logits = yami_matmul(scratch_ctx, x, model.lm_head_w);

        yami_softmax(scratch_ctx, logits, -1);

        model.probs.resize(logits->ne);
        memcpy(model.probs.data(), logits->data, logits->ne * sizeof(f32));

        std::discrete_distribution<> dist(model.probs.begin(), model.probs.end());
        generated.push_back(dist(model.rng));

        YAMI_LOG_INFO("1 tok in %f s", yami_timer() - start_t);

        yami_mem_usage(scratch_ctx);
    }

    return generated;
}

int main() {
    yami_context *ctx = yami_init(yami_context_init_params{
            1,
            1024 * 1024 * 650L,
            1024 * 1024 * 1024 * 5L,
            nullptr,
            nullptr
    });
    gpt2_model gpt2{};
    gpt2_load_model(ctx, &gpt2, "gpt2.ymf");

    printf("\n========================\n");
    printf("GPT2 loaded successfully!\n");
    yami_mem_usage(ctx);
    printf("========================\n");

    const static std::string prompt = "Hello Bill, nice to meet you my name is Franco";
    const std::vector<int> gen = generate(gpt2, gpt2.tokenizer->encode(prompt));
    printf("> %s\n", gpt2.tokenizer->decode(gen).c_str());

    yami_free(ctx);
    return 0;
}