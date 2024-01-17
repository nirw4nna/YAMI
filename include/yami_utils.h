#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "yami.h"


struct yami_model {
    std::unordered_map<std::string, yami_tensor *> tensors;
    void *hparams;
    std::vector<std::string> vocab;
    std::vector<std::string> encoder;
};

struct yami_model_settings {
    usize main_ctx_size;
    usize scratch_ctx_size;
    usize seed;
    usize top_k;
    int n_tokens;
    int n_workers;
    f32 temperature;
    std::string prompt;
    std::string yami_file;
};

struct yami_perf_metrics {
    f64 encode;
    f64 generation;
    f64 sampling;
    f64 total;
    usize model_memory;
    usize inference_memory;
    int generated_tokens;
    int prompt_tokens;

    inline void report() const noexcept {
        YAMI_LOG_INFO("prompt tokens\t\t= %d", prompt_tokens);
        YAMI_LOG_INFO("generated tokens\t= %d", generated_tokens);
        YAMI_LOG_INFO("model size\t\t= %.2f MB", YAMI_B_TO_MB(model_memory));
        YAMI_LOG_INFO("inference memory\t= %.2f MB", YAMI_B_TO_MB(inference_memory / generated_tokens));
        YAMI_LOG_INFO("encode time\t\t= %.2fms", encode * 1000.);
        YAMI_LOG_INFO("generation time\t\t= %.2fms\t(%.2fms/tok,\t%.2f tokens/s)",
                      generation * 1000.,
                      (generation * 1000.) / generated_tokens,
                      generated_tokens / generation);
        YAMI_LOG_INFO("sampling time\t\t= %.2fms\t(%.2fms/tok,\t%.2f tokens/s)",
                      sampling * 1000.,
                      (sampling * 1000.) / generated_tokens,
                      generated_tokens / sampling);
        YAMI_LOG_INFO("total time\t\t= %.2fms\t(%.2fms/tok,\t%.2f tokens/s)",
                      total * 1000.,
                      (total * 1000.) / generated_tokens,
                      generated_tokens / total);
    }
};

struct yami_token {
    f32 value;
    usize idx;
};

struct yami_bpe_tokenizer {
    yami_bpe_tokenizer(const std::vector<std::string>& bpe_pairs,
                       const std::vector<std::string>& encoder);

    std::vector<int> encode(const std::string& text) noexcept;
    std::string decode(int bpe_idx) noexcept;

private:
    // Map each value between 0 and 255 to a unicode symbol
    std::unordered_map<int, std::string> byte_encoder_;
    std::unordered_map<std::string, int> byte_decoder_;
    // Each bpe pair is just a string with two tokens separated by a single whitespace
    std::unordered_map<std::string, int> bpe_ranks_;
    std::unordered_map<std::string, int> bpe_encoder_;
    std::unordered_map<int, std::string> bpe_decoder_;
};

extern void yami_load_model(yami_context *ctx,
                            yami_model *model,
                            const char *yami_file) noexcept;

extern void yami_arg_parse(int argc, char **argv,
                           yami_model_settings *settings) noexcept;

extern std::vector<yami_token> yami_top_k(const f32 *values,
                                          usize ne,
                                          usize k = 1) noexcept;