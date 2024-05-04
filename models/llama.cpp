#include "yami.h"
#include "yami_utils.h"
#include <memory>
#include <random>


struct llama_hparams {
    u32 dim;
    u32 n_layers;
    u32 n_heads;
    u32 vocab_size;
    u32 multiple_of;
    f32 norm_eps;
    u32 max_seq_len;
};

struct transformer_block {
    yami_tensor *wq, *wk, *wv, *wo;
    yami_tensor *k_cache, *v_cache;
    yami_tensor *ff_w1, *ff_w2, *ff_w3;
    yami_tensor *attn_norm;
    yami_tensor *ff_norm;
};

struct llama_model {
    llama_model(yami_context *context, const yami_model_settings *settings) : ctx(context) {
        yami_model ym{};
        ym.hparams = &hparams;

        const f64 load_start = yami_timer();

        yami_load_model(
                ctx,
                &ym,
                settings->model_file.c_str(),
                settings->tokenizer_file.c_str(),
                settings->use_mmap
        );
        YAMI_ASSERT(ym.type == yami_models::LLAMA && ym.tokenizer == yami_tokenizers::SP);

        mmap = std::move(ym.mmap);
        tokenizer = std::make_unique<yami_llama_tokenizer>(std::move(ym.vocab), ym.scores);

        const f64 encode_start = yami_timer();
        emb = tokenizer->encode(settings->prompt, true);
        metrics.encode = yami_timer() - encode_start;
        metrics.prompt_tokens = emb.size();

        tok_embeddings = ym.tensors["tok_embeddings.weight"];

        layers.resize(hparams.n_layers);
        const usize kv_ne = (settings->n_tokens + metrics.prompt_tokens) * hparams.dim;
        for (u32 i = 0; i < hparams.n_layers; ++i) {
            transformer_block *block = &layers[i];
            block->wq = ym.tensors["layers." + std::to_string(i) + ".attention.wq.weight"];
            block->wk = ym.tensors["layers." + std::to_string(i) + ".attention.wk.weight"];
            block->wv = ym.tensors["layers." + std::to_string(i) + ".attention.wv.weight"];
            block->wo = ym.tensors["layers." + std::to_string(i) + ".attention.wo.weight"];
            block->ff_w1 = ym.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"];
            block->ff_w2 = ym.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"];
            block->ff_w3 = ym.tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"];
            block->attn_norm = ym.tensors["layers." + std::to_string(i) + ".attention_norm.weight"];
            block->ff_norm = ym.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"];

            // KV cache
            block->k_cache = yami_tensor_1d(ctx, "k_cache", kv_ne);
            block->v_cache = yami_tensor_1d(ctx, "v_cache", kv_ne);
        }

        norm = ym.tensors["norm.weight"];
        output = ym.tensors["output.weight"];

        rng = std::mt19937{settings->seed};

        YAMI_LOG_INFO("random seed\t= %ld", settings->seed);
        YAMI_LOG_INFO("temperature\t= %.2f", (f64) settings->temperature);
        YAMI_LOG_INFO("top k\t\t= %ld", settings->top_k);
        YAMI_LOG_INFO("mmap\t\t= %s", settings->use_mmap ? "true" : "false");
        YAMI_LOG_INFO("load time\t\t= %.2fs", (yami_timer() - load_start));
        YAMI_LOG_INFO("layers\t\t= %d", hparams.n_layers);
        YAMI_LOG_INFO("attention heads\t= %d", hparams.n_heads);
        YAMI_LOG_INFO("embedding size\t= %d", hparams.dim);
        YAMI_LOG_INFO("vocab size\t\t= %d", hparams.vocab_size);
        YAMI_LOG_INFO("multiple of\t= %d", hparams.multiple_of);
        YAMI_LOG_INFO("max sequence len\t= %d", hparams.max_seq_len);
        YAMI_LOG_INFO("KV cache size\t= %ld MB", (usize) YAMI_B_TO_MB(kv_ne * 2 * hparams.n_layers * sizeof(f32)));
        yami_mem_usage(ctx);
        metrics.model_memory = yami_used_mem(ctx);
        printf("============================================================================\n");

    }

    yami_context *ctx;
    std::unique_ptr<yami_llama_tokenizer> tokenizer;
    std::unique_ptr<yami_mmap> mmap;
    std::vector<int> emb;

    yami_tensor *tok_embeddings;

    std::vector<transformer_block> layers;

    yami_tensor *norm;
    yami_tensor *output;

    std::mt19937 rng;
    llama_hparams hparams{};
    yami_perf_metrics metrics{};
};

// Notes on debugging:
//  - all the shapes match Pytorch
//  > verify the actual numbers (especially RoPE)

int main(int argc, char **argv) {
    yami_model_settings settings{};
    yami_arg_parse(argc, argv, &settings);

    yami_context *ctx = yami_init(yami_context_init_params{
        settings.n_workers,
        settings.main_ctx_size,
        settings.scratch_ctx_size
    });

    llama_model model{ctx, &settings};

    printf("%s", settings.prompt.c_str());
    fflush(stdout);

    const f64 start_time = yami_timer();
    std::vector<int> generated{model.tokenizer->encode(settings.prompt, true)};
    model.metrics.prompt_tokens = generated.size();
    model.metrics.encode = yami_timer() - start_time;

    yami_context *scratch_ctx = yami_ctx_scratch(ctx);

    std::vector<int> pos;
    usize ctx_size = 0;

    const u32 n_heads = model.hparams.n_heads;
    const u32 vocab_size = model.hparams.dim;
    const u32 head_size = vocab_size / n_heads;
    const f32 scale = 1.f / std::sqrt((f32) head_size);
    for (int i = 0; i < settings.n_tokens; ++i) {
        yami_clear_ctx(scratch_ctx);

        YAMI_ASSERT(ctx_size < model.hparams.max_seq_len);

        const f64 gen_start = yami_timer();
        pos.resize(generated.size());
        for (int j = 0; j < (int) generated.size(); ++j) {
            pos[j] = ctx_size + j;
        }

        // [seq_len, vocab_size]
        yami_tensor *block_in = yami_embed(scratch_ctx, model.tok_embeddings,
                                      generated.data(), generated.size()
        );

        yami_tensor *cur;
        for (const auto &block : model.layers) {
            yami_tensor *x = yami_rms_norm(scratch_ctx, block.attn_norm, block_in, false);

            // Attention layer
            {
                // QKV = [1, seq_len, n_heads, head_size]
                yami_tensor *q = yami_reshape(
                        yami_matmul(scratch_ctx, x, block.wq), 4,
                        1, x->dimensions[0], n_heads, head_size
                );
                yami_tensor *k_cur = yami_reshape(
                        yami_matmul(scratch_ctx, x, block.wk), 4,
                        1, x->dimensions[0], n_heads, head_size
                );
                yami_tensor *v_cur = yami_reshape(
                        yami_matmul(scratch_ctx, x, block.wv), 4,
                        1, x->dimensions[0], n_heads, head_size
                );
                // RoPE, must be applied before caching
                q = yami_rope(scratch_ctx, q, head_size, false, ctx_size);
                k_cur = yami_rope(scratch_ctx, k_cur, head_size, true, ctx_size);

                // Create the actual K tensor by concat-ing with K cache
                yami_tensor *k_old = yami_view_4d(
                        scratch_ctx, block.k_cache,
                        1, ctx_size,
                        n_heads, head_size
                );
                yami_tensor *k = yami_concat(scratch_ctx, k_old, k_cur, 1);
                yami_copy(k, block.k_cache);

                // Create the actual V tensor by concat-ing with V cache
                yami_tensor *v_old = yami_view_4d(
                        scratch_ctx, block.v_cache,
                        1, ctx_size,
                        n_heads, head_size
                );
                yami_tensor *v = yami_concat(scratch_ctx, v_old, v_cur, 1);
                yami_copy(v, block.v_cache);


                // [n_heads, seq_len, head_size]
                q = yami_contiguous(scratch_ctx,
                                    yami_transpose(scratch_ctx,q, 1, 2)
                );
                // [n_heads, head_size, seq_len]
                k = yami_contiguous(scratch_ctx,
                                    yami_transpose(scratch_ctx,
                                                   yami_transpose(scratch_ctx, k, 1, 2),
                                                   -2, -1)
                );
                // [n_heads, seq_len + ctx_size, head_size]
                v = yami_contiguous(scratch_ctx,
                                    yami_transpose(scratch_ctx,v, 1, 2)
                );

                yami_tensor *qk_scaled = yami_mulc(scratch_ctx,
                                                   yami_matmul(scratch_ctx, q, k),
                                                   scale
                );
                yami_tensor *qk_masked = yami_lt_mask(scratch_ctx, qk_scaled, YAMI_MINUS_INF, ctx_size);

                yami_tensor *scores = yami_softmax(scratch_ctx, qk_masked);
                yami_tensor *out = yami_matmul(scratch_ctx, scores, v);

                cur = yami_reshape(
                        yami_contiguous(scratch_ctx,
                                        yami_transpose(scratch_ctx, out, 1, 2)
                        ),
                        2, x->dimensions[0], x->dimensions[1]
                );
                cur = yami_matmul(scratch_ctx, cur, block.wo);
            }

            // in_feed_forward = x + self.attention.forward(self.attention_norm(x))
            yami_tensor *in_ff = yami_add(scratch_ctx, block_in, cur);

            // Feed Forward layer
            {
                cur = yami_rms_norm(scratch_ctx, block.ff_norm, in_ff, false);

                yami_tensor *x_w3 = yami_matmul(scratch_ctx, cur, block.ff_w3);
                cur = yami_matmul(scratch_ctx, cur, block.ff_w1);
                cur = yami_mul(scratch_ctx,
                            yami_swiglu(scratch_ctx, cur),
                            x_w3
                );
                cur = yami_matmul(scratch_ctx, cur, block.ff_w2);
            }
            // next_layer_in = attn_layer_out + self.feed_forward.forward(self.ffn_norm(attn_layer_out))
            block_in = yami_add(scratch_ctx, cur, in_ff, true);
        }

        yami_rms_norm(scratch_ctx, model.norm, block_in);
        yami_tensor *logits = yami_matmul(scratch_ctx, block_in, model.output);

        // Select the last row
        logits = yami_view_1d(scratch_ctx, logits,
                              vocab_size, (logits->dimensions[0] - 1) * vocab_size
        );
        model.metrics.generation += (yami_timer() - gen_start);

        const f64 sampling_start = yami_timer();

        int next_tok;
        if (std::fpclassify(settings.temperature) == FP_ZERO) {
            // Always take the most likely token
            const yami_token next = yami_top_k(logits->data, vocab_size)[0];
            next_tok = next.idx;
        } else {
            yami_divc(scratch_ctx, logits, settings.temperature, true);
            if (settings.top_k != 0) {
                // If top_k is set crop the logits to the top k most likely ones
                const f32 smallest_of_the_k = yami_top_k(logits->data, vocab_size, settings.top_k).back().value;
                yami_mask_if(scratch_ctx, logits, yami_mask::LOWER, smallest_of_the_k, YAMI_MINUS_INF);
            }
            yami_softmax(scratch_ctx, logits);

            std::discrete_distribution<> dist(logits->data, logits->data + logits->ne);

            next_tok = dist(model.rng);
        }
        model.metrics.sampling += (yami_timer() - sampling_start);

        if (next_tok == yami_llama_tokenizer::eos_id)
            break;

//        printf("%s", model.tokenizer->decode(next_tok).c_str());
//        fflushstdout);

        ctx_size += generated.size();
        generated.clear();
        generated.push_back(next_tok);

        model.metrics.inference_memory += yami_used_mem(scratch_ctx);
    }
    model.metrics.total = yami_timer() - start_time;

    printf("\n");

    model.metrics.generated_tokens = ctx_size - model.metrics.prompt_tokens;
    model.metrics.report();

    yami_free(ctx);

    return EXIT_SUCCESS;
}