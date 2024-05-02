#include "yami.h"
#include "yami_utils.h"
#include <memory>
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

    // KV cache
    yami_tensor *k_cache;
    yami_tensor *v_cache;
};

struct gpt2_model {
    gpt2_model(yami_ctx *context, const yami_model_settings *settings) : ctx(context) {
        yami_model ym{};
        ym.hparams = &hparams;
        const f64 load_start = yami_timer();
        yami_load_model(ctx,
                        &ym,
                        settings->model_file.c_str(),
                        settings->tokenizer_file.c_str(),
                        settings->use_mmap
        );
        YAMI_ASSERT(ym.type == yami_models::GPT2 && ym.tokenizer == yami_tokenizers::BPE);
        mmap = std::move(ym.mmap);
        tokenizer = std::make_unique<yami_bpe_tokenizer>(ym.vocab, ym.encoder);

        wte = ym.tensors["transformer.wte.weight"];
        wpe = ym.tensors["transformer.wpe.weight"];

        h.resize(hparams.n_layers);
        const usize kv_ne = hparams.block_size * hparams.emb_size;
        for (u32 i = 0; i < hparams.n_layers; ++i) {
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

            // Allocate enough memory for the KV cache
            block->k_cache = yami_tensor_1d(ctx, "k_cache", kv_ne);
            block->v_cache = yami_tensor_1d(ctx, "v_cache", kv_ne);
        }

        ln_f_w = ym.tensors["transformer.ln_f.weight"];
        ln_f_b = ym.tensors["transformer.ln_f.bias"];
        lm_head_w = ym.tensors["lm_head.weight"];

        rng = std::mt19937{settings->seed};

        YAMI_LOG_INFO("random seed\t\t= %ld", settings->seed);
        YAMI_LOG_INFO("temperature\t\t= %.2f", (f64) settings->temperature);
        YAMI_LOG_INFO("top k\t\t= %d", settings->top_k);
        YAMI_LOG_INFO("mmap\t\t= %s", settings->use_mmap ? "true" : "false");
        YAMI_LOG_INFO("load time\t\t= %.2fs", (yami_timer() - load_start));
        YAMI_LOG_INFO("layers\t\t= %d", hparams.n_layers);
        YAMI_LOG_INFO("attention heads\t= %d", hparams.n_heads);
        YAMI_LOG_INFO("embedding size\t= %d", hparams.emb_size);
        YAMI_LOG_INFO("vocab size\t\t= %d", hparams.vocab_size);
        YAMI_LOG_INFO("block size\t\t= %d", hparams.block_size);
        YAMI_LOG_INFO("KV cache size\t= %ld MB", (usize) YAMI_B_TO_MB(kv_ne * 2 * hparams.n_layers * sizeof(f32)));
        yami_mem_usage(ctx);
        metrics.model_memory = yami_used_mem(ctx);
        printf("============================================================================\n");

        // After loading the model data set the default yami scope to 'local'
        yami_set_scope(ctx, yami_scope::LOCAL);
    }

    yami_ctx *ctx;
    std::unique_ptr<yami_bpe_tokenizer> tokenizer;
    std::unique_ptr<yami_mmap> mmap;

    yami_tensor *wpe;
    yami_tensor *wte;

    std::vector<transformer_block> h;

    yami_tensor *ln_f_w;
    yami_tensor *ln_f_b;
    yami_tensor *lm_head_w;

    std::mt19937 rng;
    gpt2_hparams hparams{};
    yami_perf_metrics metrics{};
};

int main(int argc, char **argv) {
    yami_model_settings settings {};
    yami_arg_parse(argc, argv, &settings);

    yami_ctx *ctx = yami_init(yami_init_params{
            settings.n_workers,
            settings.main_ctx_size,
            settings.scratch_ctx_size
    });
    gpt2_model gpt2{ctx, &settings};

    printf("%s", settings.prompt.c_str());
    fflush(stdout);

    const f64 start_time = yami_timer();
    std::vector<int> generated{gpt2.tokenizer->encode(settings.prompt)};
    gpt2.metrics.prompt_tokens = (int) generated.size();
    // TODO: add prompt eval time to the metrics
    gpt2.metrics.encode = yami_timer() - start_time;

    std::vector<int> pos;

    const u32 n_heads = gpt2.hparams.n_heads;
    const usize head_size = gpt2.hparams.emb_size / n_heads;
    const f32 att_scale = 1.f / std::sqrt((f32) head_size);

    int ctx_size = 0;
    for (int i = 0; i < settings.n_tokens; ++i) {
        yami_clear_ctx(ctx);

        YAMI_ASSERT(ctx_size < (int) gpt2.hparams.block_size);

        const f64 gen_start = yami_timer();
        pos.resize(generated.size());
        for (int j = 0; j < (int) generated.size(); ++j) {
            pos[j] = ctx_size + j;
        }

        yami_tensor *tok_emb = yami_embed(ctx, gpt2.wte, generated.data(), generated.size());
        yami_tensor *pos_emb = yami_embed(ctx, gpt2.wpe, pos.data(), pos.size());

        yami_tensor *x = yami_add(ctx, tok_emb, pos_emb, true);

        yami_tensor *cur;

        for (const auto &block : gpt2.h) {
            cur = yami_layer_norm(ctx, block.ln_1_w, block.ln_1_b, x);

            cur = yami_add(ctx,
                           yami_matmul(ctx, cur, block.c_attn_w),
                           block.c_attn_b,
                           true
            );

            yami_tensor *q = yami_reshape(yami_split(ctx, cur, gpt2.hparams.emb_size, 0), 4,
                                          1, x->dim[yami_tensor_dim(x, 0)], n_heads, head_size);
            yami_tensor *k_cur = yami_reshape(yami_split(ctx, cur, gpt2.hparams.emb_size, 1), 4,
                                              1, x->dim[yami_tensor_dim(x, 0)], n_heads, head_size);
            yami_tensor *v_cur = yami_reshape(yami_split(ctx, cur, gpt2.hparams.emb_size, 2), 4,
                                              1, x->dim[yami_tensor_dim(x, 0)], n_heads, head_size);

            yami_tensor *q_t = yami_contiguous(ctx,
                                               yami_transpose(ctx, q, 1, 2)
            );
            // [B, nh, T, hs]
            yami_tensor *k_cur_t = yami_contiguous(ctx,
                                                 yami_transpose(ctx, k_cur, 1, 2)
            );
            yami_tensor *v_cur_t = yami_contiguous(ctx,
                                               yami_transpose(ctx, v_cur, 1, 2)
            );

            // KV cache
            yami_tensor *k_old = yami_view_4d(ctx, block.k_cache, 1, n_heads, ctx_size, head_size);
            yami_tensor *k = yami_concat(ctx, k_old, k_cur_t, 2);

            yami_copy(k, block.k_cache);

            yami_tensor *k_t = yami_contiguous(ctx,
                                               yami_transpose(ctx, k, -2, -1)
            );

            yami_tensor *v_old = yami_view_4d(ctx, block.v_cache, 1, n_heads, ctx_size, head_size);
            yami_tensor *v_t = yami_concat(ctx, v_old, v_cur_t, 2);
            yami_copy(v_t, block.v_cache);

            cur = yami_matmul(ctx, q_t, k_t);

            yami_mulc(ctx, cur, att_scale);

            yami_lt_mask(ctx, cur, YAMI_MINUS_INF, ctx_size);

            cur = yami_softmax(ctx, cur);

            yami_tensor *out = yami_matmul(ctx, cur, v_t);

            out = yami_contiguous(ctx,
                                  yami_transpose(ctx, out, 1, 2)
            );

            yami_reshape(out, 2, generated.size(), gpt2.hparams.emb_size);

            yami_tensor *out_proj = yami_add(
                    ctx,
                    yami_matmul(ctx, out, block.c_attn_proj_w),
                    block.c_attn_proj_b,
                    true
            );

            // x = x + self.attn(self.ln_1(x))
            yami_add(ctx, x, out_proj, true);

            // x = x + m.c_proj(m.act(m.c_fc(self.ln_2(x))))
            cur = yami_layer_norm(ctx, block.ln_2_w, block.ln_2_b, x);

            cur = yami_add(ctx,
                           yami_matmul(ctx, cur, block.c_fc_w),
                           block.c_fc_b,
                           true
            );

            yami_gelu(ctx, cur);

            x = yami_add(ctx,
                         x,
                         yami_add(
                                 ctx,
                                 yami_matmul(ctx, cur, block.c_proj_w),
                                 block.c_proj_b,
                                 true
                         ),
                         true
            );
        }
        x = yami_layer_norm(ctx, gpt2.ln_f_w, gpt2.ln_f_b, x);

        yami_tensor *logits = yami_matmul(ctx, x, gpt2.lm_head_w);
        const int vocab_size = (int) gpt2.hparams.vocab_size;

        // Select the last row of logits
        logits = yami_view_1d(ctx, logits, vocab_size, (logits->dim[yami_tensor_dim(logits, 0)] - 1) * vocab_size);
        gpt2.metrics.generation += (yami_timer() - gen_start);
        // TODO: if it's the first loop this is not `generation` but `prompt eval`

        const f64 sampling_start = yami_timer();

        int next_tok;
        if (std::fpclassify(settings.temperature) == FP_ZERO) {
            // Always take the most likely token
            const yami_token next = yami_top_k(logits->data, vocab_size)[0];
            next_tok = next.idx;
        } else {
            yami_divc(ctx, logits, settings.temperature, true);
            if (settings.top_k != 0) {
                // If top_k is set crop the logits to the top k most likely ones
                const f32 smallest_of_the_k = yami_top_k(logits->data, vocab_size, settings.top_k).back().value;
                yami_mask_if(ctx, logits, yami_mask_flag::LOWER, smallest_of_the_k, YAMI_MINUS_INF);
            }
            logits = yami_softmax(ctx, logits);

            std::discrete_distribution<> dist{logits->data, logits->data + logits->ne};

            next_tok = dist(gpt2.rng);
        }
        gpt2.metrics.sampling += (yami_timer() - sampling_start);

        if (next_tok == 50256) // end-of-text
            break;

        printf("%s", gpt2.tokenizer->decode(next_tok).c_str());
        fflush(stdout);

        ctx_size += (int) generated.size();
        generated.clear();
        generated.push_back(next_tok);

        gpt2.metrics.inference_memory += yami_used_mem(ctx);
    }
    gpt2.metrics.total = yami_timer() - start_time;

    printf("\n");

    gpt2.metrics.generated_tokens = ctx_size - gpt2.metrics.prompt_tokens + 1;
    gpt2.metrics.report();

    yami_free(ctx);
    return 0;
}