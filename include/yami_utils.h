// Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the MIT license
// (https://opensource.org/license/mit).

#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include "yami.h"
#include <queue>
#include <memory>

enum yami_models : u8 {
    GPT2,
    LLAMA,
    MODEL_INVALID,
};

enum yami_tokenizers : u8 {
    BPE,
    SP,
    TOKENIZER_INVALID,
};

static constexpr const char* YAMI_MODELS[2] = {
        "GPT2",
        "LLaMA",
};

static constexpr const char* YAMI_TOKENIZERS[2] = {
        "BPE",
        "SP",
};

struct yami_mmap {
    // mmap has alignment requirements so, since the tokenizer is actually quite small compared
    // to the weights just mmap the entire file and add the weights offset to the returned pointer
    yami_mmap(const char *yami_file, usize offset);
    ~yami_mmap();

    size mapping_size;
    void *data;
};

struct yami_model {
    std::unordered_map<std::string, yami_tensor *> tensors;
    void *hparams;
    std::vector<std::string> vocab;
    std::vector<f32> scores;
    std::vector<std::string> encoder;
    // These are the values actually loaded from file, they must be manually verified before constructing
    // the model to make sure that a valid file for that model has been loaded.
    yami_models type;
    yami_tokenizers tokenizer;
    // The mmap object must be manually moved to the actual model struct in order to keep alive the memory mapping.
    std::unique_ptr<yami_mmap> mmap;
};

struct yami_model_settings {
    usize main_ctx_size;
    usize scratch_ctx_size;
    usize seed;
    int top_k;
    int n_tokens;
    int n_workers;
    f32 temperature;
    std::string prompt;
    std::string yami_file;
    bool use_mmap;
};

struct yami_perf_metrics {
    // Time required to encode the prompt
    f64 encode;
    // Time required for prompt evaluation
    f64 prompt_eval;
    // Time required to generate the logits for the next token
    f64 generation;
    // Time required to sample the next token given its logits
    f64 sampling;
    // Total time spent in the forward loop
    f64 total;

    usize model_memory;
    usize inference_memory;
    int generated_tokens;
    int prompt_tokens;

    YAMI_INLINE void report() const noexcept {
        YAMI_LOG_INFO("prompt tokens\t\t= %d", prompt_tokens);
        YAMI_LOG_INFO("generated tokens\t= %d", generated_tokens);
        YAMI_LOG_INFO("model size\t\t= %.2f MB", YAMI_B_TO_MB(model_memory));
        YAMI_LOG_INFO("inference memory\t= %.2f MB", YAMI_B_TO_MB((f64) inference_memory / generated_tokens));
        YAMI_LOG_INFO("encode time\t\t= %.2fms", encode * 1000.);
        YAMI_LOG_INFO("prompt eval time\t= %.2fms\t(%.2fms/tok,\t%.2f tokens/s)",
                      prompt_eval * 1000.,
                      (prompt_eval * 1000.) / prompt_tokens,
                      prompt_tokens / prompt_eval);
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

struct yami_bpe_tokenizer {
    static constexpr int eos_id = 50256;

    yami_bpe_tokenizer(const std::vector<std::string> &bpe_pairs,
                       const std::vector<std::string> &encoder);

    std::vector<int> encode(const std::string &text) noexcept;
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

// SP implementation inspired by https://github.dev/google/sentencepiece
struct yami_sp_symbol {
    // Indexes in the symbols vector of the previous and next symbol wrt. this one.
    size prev, next;
    // This symbols, viewed as a string.
    std::string_view piece;
};

// A bigram is just a pair of symbols, each symbol is represented by its index in the symbols vector.
struct yami_sp_bigram {
    struct comparator {
        bool operator() (const yami_sp_bigram &b1, const yami_sp_bigram &b2) {
            return b1.score < b2.score || (b1.score == b2.score && b1.left > b2.left);
        }
    };
    // The two symbols that make up this bigram.
    size left, right;
    f32 score;
    // The sum of the length of the two symbols.
    usize len;
};

struct yami_llama_tokenizer {
    static constexpr int unk_id = 0, bos_id = 1, eos_id = 2;

    yami_llama_tokenizer(std::vector<std::string> &&vocab,
                         const std::vector<f32> &scores);

    std::vector<int> encode(const std::string &text, bool bos = false) noexcept;
    std::string decode(int idx) const noexcept;

private:
    void maybe_add(const std::vector<yami_sp_symbol> &symbols,
                   size left, size right) noexcept;

    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::string, int> tok_to_id_;
    std::unordered_map<std::string, f32> tok_to_score_;
    std::priority_queue<yami_sp_bigram,
            std::vector<yami_sp_bigram>,
            yami_sp_bigram::comparator> agenda_;
};

struct yami_token {
    f32 value;
    int idx;
};

extern void yami_load_model(yami_ctx *ctx,
                            yami_model *model,
                            const char *yami_file,
                            bool use_mmap) noexcept;

extern void yami_arg_parse(int argc, char **argv,
                           yami_model_settings *settings) noexcept;

extern std::vector<yami_token> yami_top_k(const f32 *values,
                                          int ne,
                                          int k = 1) noexcept;