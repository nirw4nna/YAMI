#include "yami.h"
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
    gpt2_model(yami_context *context, const yami_model_settings *settings) : ctx(context) {
        gpt2_hparams hp{};
        yami_model ym{};
        ym.hparams = &hp;
        const f64 load_start = yami_timer();
        yami_load_model(ctx, &ym, settings->yami_file.c_str());

        hparams = *((gpt2_hparams *) ym.hparams);
        tokenizer = std::make_unique<yami_bpe_tokenizer>(ym.vocab, ym.encoder);

        wte = ym.tensors["transformer.wte.weight"];
        wpe = ym.tensors["transformer.wpe.weight"];

        h.resize(hp.n_layers);
        for (u32 i = 0; i < hp.n_layers; ++i) {
            transformer_block *block = &h[i];
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

        ln_f_w = ym.tensors["transformer.ln_f.weight"];
        ln_f_b = ym.tensors["transformer.ln_f.bias"];
        lm_head_w = ym.tensors["lm_head.weight"];

        rng = std::mt19937(settings->seed);
        time_metrics.n_tokens = settings->n_tokens;

        YAMI_LOG_INFO("random seed\t\t= %ld", settings->seed);
        YAMI_LOG_INFO("load time\t\t= %.2fs", (yami_timer() - load_start));
        YAMI_LOG_INFO("layers\t\t= %d", hp.n_layers);
        YAMI_LOG_INFO("attention heads\t= %d", hp.n_heads);
        YAMI_LOG_INFO("embedding size\t= %d", hp.n_heads);
        YAMI_LOG_INFO("vocab size\t\t= %d", hp.vocab_size);
        YAMI_LOG_INFO("block size\t\t= %d", hp.block_size);
        yami_mem_usage(ctx);
        printf("============================================================================\n");
    }

    yami_context *ctx;
    std::unique_ptr<yami_bpe_tokenizer> tokenizer;

    yami_tensor *wpe;
    yami_tensor *wte;

    std::vector<transformer_block> h;

    yami_tensor *ln_f_w;
    yami_tensor *ln_f_b;
    yami_tensor *lm_head_w;

    std::mt19937 rng;
    std::vector<f32> probs;
    gpt2_hparams hparams{};
    yami_time_metrics time_metrics{};
};

int main(int argc, char **argv) {
    yami_model_settings settings{};
    yami_arg_parse(argc, argv, &settings);

    yami_context *ctx = yami_init(yami_context_init_params{
            settings.n_workers,
            settings.main_ctx_size,
            settings.scratch_ctx_size
    });
    gpt2_model gpt2{ctx, &settings};


    const f64 start_time = yami_timer();
    std::vector<int> generated(gpt2.tokenizer->encode(settings.prompt));
    gpt2.time_metrics.encode = yami_timer() - start_time;

    yami_context *scratch_ctx = yami_ctx_scratch(gpt2.ctx);

    std::vector<int> pos;

    const usize head_size = gpt2.hparams.emb_size / gpt2.hparams.n_heads;
    const f32 att_scale = 1.f / std::sqrt((f32) head_size);
    for (int i = 0; i < settings.n_tokens; ++i) {
        yami_clear_ctx(scratch_ctx);

        YAMI_ASSERT(generated.size() < gpt2.hparams.block_size);

        const f64 gen_start = yami_timer();
        pos.resize(generated.size());
        for (int j = 0; j < (int) generated.size(); ++j) {
            pos[j] = j;
        }

        yami_tensor *tok_emb = yami_embed(scratch_ctx, gpt2.wte, generated.data(), (size) generated.size());
        yami_tensor *pos_emb = yami_embed(scratch_ctx, gpt2.wpe, pos.data(), (size) pos.size());

        yami_tensor *x = yami_add(scratch_ctx, tok_emb, pos_emb, true);

        yami_tensor *cur;

        for (const auto &block : gpt2.h) {
            cur = yami_layer_norm(scratch_ctx, block.ln_1_w, block.ln_1_b, x, false);

            cur = yami_add(scratch_ctx,
                           yami_matmul(scratch_ctx, cur, block.c_attn_w),
                           block.c_attn_b,
                           true);

            yami_tensor *q = yami_reshape(yami_split(scratch_ctx, cur, gpt2.hparams.emb_size, 0), 4,
                                          1, x->dimensions[0], gpt2.hparams.n_heads, head_size);
            yami_tensor *k = yami_reshape(yami_split(scratch_ctx, cur, gpt2.hparams.emb_size, 1), 4,
                                          1, x->dimensions[0], gpt2.hparams.n_heads, head_size);
            yami_tensor *v = yami_reshape(yami_split(scratch_ctx, cur, gpt2.hparams.emb_size, 2), 4,
                                          1, x->dimensions[0], gpt2.hparams.n_heads, head_size);

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

            yami_lt_mask(scratch_ctx, cur, YAMI_MINUS_INF);

            yami_softmax(scratch_ctx, cur, -1);

            yami_tensor *out = yami_matmul(scratch_ctx, cur, v_t);

            out = yami_contiguous(scratch_ctx,
                                  yami_transpose(scratch_ctx, out, 1, 2)
            );

            yami_reshape(out, 2, generated.size(), gpt2.hparams.emb_size);

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
        yami_layer_norm(scratch_ctx, gpt2.ln_f_w, gpt2.ln_f_b, x);

        yami_tensor *logits = yami_matmul(scratch_ctx, x, gpt2.lm_head_w);

        yami_divc(scratch_ctx, logits, settings.temperature, true);

        yami_softmax(scratch_ctx, logits, -1);
        gpt2.time_metrics.generation += (yami_timer() - gen_start);

        const f64 sampling_start = yami_timer();
        const u32 vocab_size = gpt2.hparams.vocab_size;
        gpt2.probs.resize(vocab_size);

        memcpy(gpt2.probs.data(), &logits->data[(logits->dimensions[0] - 1) * vocab_size], vocab_size * sizeof(f32));

        std::discrete_distribution<> dist(gpt2.probs.begin(), gpt2.probs.end());

        int next_tok = dist(gpt2.rng);
        gpt2.time_metrics.sampling += (yami_timer() - sampling_start);

        printf("%s", gpt2.tokenizer->decode(next_tok).c_str());
        fflush(stdout);

        generated.push_back(next_tok);
    }
    gpt2.time_metrics.total = yami_timer() - start_time;

    printf("\n");

    gpt2.time_metrics.report_timings();

    yami_free(ctx);
    return 0;
}