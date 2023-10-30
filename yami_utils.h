#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "yami2.h"

struct yami_model {
    std::unordered_map<std::string, yami_tensor *> tensors;
    void *hparams;
    std::vector<std::string> vocab;
    std::vector<std::string> encoder;
};

struct yami_bpe_tokenizer {
    yami_bpe_tokenizer(const std::vector<std::string>& bpe_pairs,
                       const std::vector<std::string>& encoder);

    std::vector<int> encode(const std::string& text) noexcept;
    std::string decode(const std::vector<int>& bpe_idx) noexcept;

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