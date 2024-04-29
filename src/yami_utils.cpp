#include "yami_utils.h"
#include <unordered_set>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <fstream>

#define YAMI_HEADER_SIZE    ((usize) 6) // header + version number
#define YAMI_VERSION        ((u8) 1)

#define read(DST, siz, n, F) YAMI_ASSERT(fread(DST, siz, n, F) == (n))

constexpr static u8 yami_tokenizer_header[YAMI_HEADER_SIZE] = {0x59, 0x41, 0x4D, 0x49, 0x54, YAMI_VERSION};
constexpr static u8 yami_model_header[YAMI_HEADER_SIZE] = {0x59, 0x41, 0x4D, 0x49, 0x4D, YAMI_VERSION};


static FILE *open_and_check(const char *yami_file, const u8 *header) noexcept {
    FILE *f = fopen(yami_file, "rb");
    YAMI_ASSERT(f != nullptr);

    u8 buff[YAMI_HEADER_SIZE];
    read(buff, 1, YAMI_HEADER_SIZE, f);

    if (memcmp(header, buff, YAMI_HEADER_SIZE) != 0) {
        fclose(f);
        YAMI_LOG_FATAL("error reading tokenizer from \"%s\"", yami_file);
    }

    return f;
}

yami_mmap::yami_mmap(const char *file) {
    FILE *f = open_and_check(file, yami_model_header);

    fseek(f, 0, SEEK_END);
    file_size = ftell(f);

    YAMI_LOG_INFO("loading %ld%s from \"%s\"",
                  (size) ((file_size < 1'000'000) ? YAMI_B_TO_KB(file_size) : YAMI_B_TO_MB(file_size)),
                  (file_size < 1'000'000) ? "KB" : "MB", file);

    int fd = fileno(f);
    data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    fclose(f);
    if (data == MAP_FAILED) {
        YAMI_LOG_FATAL("error mapping \"%s\" to memory", file);
    }
}

yami_mmap::~yami_mmap() {
    munmap(data, file_size);
}

static inline void bytes_from_unicode(const int unicode, int *n, u8 *buff) noexcept {
    if (unicode >= 0 && unicode <= 0x7F) {
        buff[0] = (u8) unicode;
        *n = 1;
    } else {
        buff[0] = (u8) (0xC0 | (unicode >> 6));
        buff[1] = (u8) (0x80 | (unicode & 0x3F));
        *n = 2;
    }
}

static inline std::string to_utf8(const int unicode) noexcept {
    YAMI_ASSERT(unicode >= 0 && unicode <= 0x7FF);
    std::string utf8;
    utf8.reserve(2);

    u8 bytes[2];
    int n;
    bytes_from_unicode(unicode, &n, bytes);

    for (int i = 0; i < n; ++i)
        utf8 += (char) bytes[i];

    return utf8;
}

static inline int from_utf8(const std::string& text, usize *offset = nullptr) noexcept {
    if (text.empty())
        return -1;

    usize d = 0;
    offset = offset == nullptr ? &d : offset;

    int b0 = (int) ((u8)text[*offset]);

    // Up to 2-byte UTF-8
    if (b0 <= 0x7F) {
        YAMI_ASSERT(text.size() > *offset);
        (*offset)++;
    } else if ((b0 & 0xE0) == 0xC0) {
        YAMI_ASSERT(text.size() > *offset + 1);
        b0 = ((b0 & 0x1F) << 6) | (text[*offset + 1] & 0x3F);
        (*offset) += 2;
    } else {
        YAMI_LOG_FATAL("unsupported UTF-8 0x%X", b0);
    }

    return b0;
}

yami_bpe_tokenizer::yami_bpe_tokenizer(const std::vector<std::string>& bpe_pairs,
                                       const std::vector<std::string>& encoder) {
    for (int c = u'!'; c <= u'~'; ++c) {
        byte_encoder_[c] = to_utf8(c);
    }

    for (int c = u'¡'; c <= u'¬'; ++c) {
        byte_encoder_[c] = to_utf8(c);
    }

    for (int c = u'®'; c <= u'ÿ'; ++c) {
        byte_encoder_[c] = to_utf8(c);
    }

    int n = 0;
    // The values that are not yet inside the map (the ones which are not 'printable')
    // will be encoded using a unicode symbol starting from 256.
    for (int raw = 0; raw < 256; ++raw) {
        if (byte_encoder_.find(raw) == byte_encoder_.end()) {
            byte_encoder_[raw] = to_utf8(256 + n);
            n++;
        }
    }

    // Reverse mapping
    for (const auto &it : byte_encoder_) {
        byte_decoder_[it.second] = it.first;
    }

    for (usize i = 0; i < bpe_pairs.size(); ++i)
        bpe_ranks_[bpe_pairs[i]] = (int) i;

    for (usize i = 0; i < encoder.size(); ++i) {
        const auto& el = encoder[i];
        bpe_encoder_[el] = (int) i;
        bpe_decoder_[(int) i] = el;
    }
}

static inline std::unordered_set<std::string> get_bigrams(const std::vector<std::string>& word) noexcept {
    std::unordered_set<std::string> bigrams;

    std::string prev = word[0];
    std::string c;
    for (usize i = 1; i < word.size(); ++i) {
        c = word[i];
        bigrams.emplace(prev.append(' ' + c));
        prev = c;
    }
    return bigrams;
}

// last letter LATIN CAPITAL LETTER N WITH ACUTE (U+0143)
static inline YAMI_STRICTLY_PURE bool is_letter(const int unicode) noexcept {
    return (unicode >= 0x41 && unicode <= 0x5A) || (unicode >= 0x61 && unicode <= 0x7A) ||
           (unicode >= 0xC0 && unicode <= 0xD6) || (unicode >= 0xD8 && unicode <= 0xF6) ||
           (unicode >= 0xF8 && unicode <= 0x131) || (unicode >= 0x134 && unicode <= 0x143);
}

static inline YAMI_STRICTLY_PURE bool is_digit(const int unicode) noexcept {
    return (unicode >= 0x30 && unicode <= 0x39) || (unicode == 0xB2) ||
           (unicode == 0xB3) || (unicode == 0xB9);
}

static inline YAMI_STRICTLY_PURE bool is_whitespace(const int unicode) noexcept {
    return (unicode >= 0x08 && unicode <= 0x0D) || (unicode >= 0x1C && unicode <= 0x20) ||
           (unicode == 0x85) || (unicode == 0xA0);
}

std::vector<int> yami_bpe_tokenizer::encode(const std::string& text) noexcept {
    std::vector<std::string> tokens;
    std::vector<int> bpe_idx;
    // Conservative assumption to prevent excessive allocations
    tokens.reserve(text.size());
    bpe_idx.reserve(text.size());

    // Build a list of all the utf-8 characters in text
    std::vector<std::string> text_utf8;
    text_utf8.reserve(text.size());
    for (usize i = 0; i < text.size();) {
        int codepoint = from_utf8(text, &i);
        text_utf8.emplace_back(to_utf8(codepoint));
    }

    // Now the hard part: get tokens based on this OpenAI regex:
    // r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    std::string token;
    bool next_is_digit = false;
    bool next_is_letter = false;
    bool next_is_space = false;
    bool next_is_other = false;
    bool split = true;
    for (usize i = 0; i < text_utf8.size(); ++i) {
        const std::string& c = text_utf8[i];
        const std::string& c_1 = text_utf8.size() > (i + 1) ? text_utf8[i + 1] : "";
        const std::string& c_2 = text_utf8.size() > (i + 2) ? text_utf8[i + 2] : "";
        split = false;

        // 's 't 'm 'd
        if (c == "\'" && (c_1 == "s" || c_1 == "t" || c_1 == "m" || c_1 == "d")) {
            // Split
            if (!token.empty())
                tokens.emplace_back(token);

            token = c;
            token += c_1;
            tokens.emplace_back(token);
            // Skip the next c as it's part of this contraction
            token = "";
            i++;
            continue;
        }

        // 're 've 'll
        if (c == "\'" && ((c_1 == "r" && c_2 == "e") || (c_1 == "v" && c_2 == "e") || (c_1 == "l" && c_2 == "l"))) {
            // Split
            if (!token.empty())
                tokens.emplace_back(token);

            token = c;
            token += c_1;
            token += c_2;
            tokens.emplace_back(token);
            // Skip the next 2 c as it's part of this contraction
            token = "";
            i += 2;
            continue;
        }

        const int c_unicode = from_utf8(c);
        const int c_1_unicode = from_utf8(c_1);

        if (!next_is_digit && !next_is_space && !next_is_letter && !next_is_other) {
            if (is_letter(c_unicode) || (token.empty() && c == " " && is_letter(c_1_unicode))) {
                next_is_letter = true;
            } else if (is_digit(c_unicode) || (token.empty() && c == " " && is_digit(c_1_unicode))) {
                next_is_digit = true;
            } else if ((!is_digit(c_unicode) && !is_letter(c_unicode) && !is_whitespace(c_unicode)) ||
                       (token.empty() && c == " " && !is_digit(c_1_unicode) && !is_letter(c_1_unicode) && !is_whitespace(c_1_unicode))) {
                next_is_other = true;
            } else if (is_whitespace(c_unicode) && is_whitespace(c_1_unicode)) {
                next_is_space = true;
            } else if (is_whitespace(c_unicode)) {
                split = true;
            }
        } else if ((next_is_letter && !is_letter(c_unicode)) ||
                   (next_is_digit && !is_digit(c_unicode)) ||
                   (next_is_other && (is_digit(c_unicode) || is_letter(c_unicode) || is_whitespace(c_unicode))) ||
                   (next_is_space && (is_digit(c_1_unicode) || is_letter(c_1_unicode)))) {
            split = true;
        }

        if (split) {
            if (!token.empty())
                tokens.emplace_back(token);

            token = c;
            next_is_space = false;
            next_is_digit = false;
            next_is_letter = false;
            next_is_other = false;
        } else {
            token += c;
        }
    }
    if (!token.empty())
        tokens.emplace_back(token);

    std::unordered_set<std::string> bigrams;
    u8 unicode_bytes_buff[2];
    int unicode_bytes_n;
    for (const auto &tok : tokens) {

        // 1. Encode the token
        std::vector<std::string> encoded_tok;
        for (usize i = 0; i < tok.size();){
            const int unicode = from_utf8(tok, &i);
            bytes_from_unicode(unicode, &unicode_bytes_n, unicode_bytes_buff);
            for (int k = 0; k < unicode_bytes_n; ++k)
                encoded_tok.emplace_back(byte_encoder_[unicode_bytes_buff[k]]);
        }

        // 2. Perform the merges

        // 2.1 Get all the bigrams in the encoded token
        bigrams = get_bigrams(encoded_tok);

        if (!bigrams.empty()) {
            while (true) {
                std::string next_bigram;
                // 2.2 Find the next bigram which is the one with the lowest bpe rank
                next_bigram = *std::min_element(bigrams.begin(), bigrams.end(),
                                                [&](const std::string& a, const std::string& b) {
                                                    constexpr static int max_int = std::numeric_limits<int>::max();
                                                    const int a_score = bpe_ranks_.find(a) == bpe_ranks_.end() ? max_int : bpe_ranks_[a];
                                                    const int b_score = bpe_ranks_.find(b) == bpe_ranks_.end() ? max_int : bpe_ranks_[b];
                                                    return a_score < b_score;
                                                });
                // Verify that next_bigram is actually a valid bigram!
                if (bpe_ranks_.find(next_bigram) == bpe_ranks_.end())
                    break;

                const std::string& first = next_bigram.substr(0, next_bigram.find(' '));
                const std::string& second = next_bigram.substr(next_bigram.find(' ') + 1);
                const std::string& first_second = first + second;

                // 2.3 Find all the occurrences of (first, second) and merge them in one token
                std::vector<std::string> merged_tok;
                usize i = 0;
                while (i < encoded_tok.size()) {
                    const auto j = std::find(encoded_tok.begin() + i, encoded_tok.end(), first);
                    if (j == encoded_tok.end()) {
                        // No occurrences of first in encoded_tok, done
                        for (usize k = i; k < encoded_tok.size(); ++k)
                            merged_tok.emplace_back(encoded_tok[k]);

                        break;
                    }
                    const usize j_idx = std::distance(encoded_tok.begin(), j);
                    for (usize k = i; k < j_idx; ++k)
                        merged_tok.emplace_back(encoded_tok[k]);

                    i = j_idx;

                    // If first is followed by second merge them into one
                    if (encoded_tok[i] == first && i < encoded_tok.size() - 1 && encoded_tok[i + 1] == second) {
                        merged_tok.emplace_back(first_second);
                        i += 2;
                    } else {
                        merged_tok.emplace_back(first);
                        i++;
                    }
                }
                encoded_tok = std::move(merged_tok);
                if (encoded_tok.size() == 1)
                    break;
                bigrams = get_bigrams(encoded_tok);
            }
        }
        for (const auto &et : encoded_tok)
            bpe_idx.emplace_back(bpe_encoder_[et]);
    }

    return bpe_idx;
}


std::string yami_bpe_tokenizer::decode(const int bpe_idx) noexcept {
    std::string decoded;

    std::string merged;
    merged += bpe_decoder_[bpe_idx];
    for (usize i = 0; i < merged.size();) {
        const int unicode = from_utf8(merged, &i);
        const std::string &as_str = to_utf8(unicode);
        decoded += (byte) byte_decoder_[as_str];
    }

    return decoded;
}

static inline YAMI_STRICTLY_PURE int utf8_len(const char c) noexcept {
    // UTF-8 encoding length depends on the first bits of the first byte:
    //  - 0xxx: 1 byte
    //  - 110x: 2 bytes
    //  - 1110: 3 bytes
    //  - 1111: 4 bytes
    // Values like 10xx must not appear here because it's the format of the trailing bytes,
    // if they do appear it means that this function was called on the wrong byte.
    const int lookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 2, 2, 3, 4};
    return lookup[(c >> 4) & 0x0F];
}

yami_llama_tokenizer::yami_llama_tokenizer(std::vector<std::string> &&vocab,
                                           const std::vector<f32> &scores) : id_to_tok_(vocab) {
    const size n_vocab = id_to_tok_.size();
    tok_to_id_.reserve(n_vocab);
    tok_to_score_.reserve(n_vocab);

    for (size i = 0; i < n_vocab; ++i) {
        tok_to_id_[id_to_tok_[i]] = i;
        tok_to_score_[id_to_tok_[i]] = scores[i];
    }
}

std::vector<int> yami_llama_tokenizer::encode(const std::string &text, const bool bos) noexcept {
    // Underlying buffer for the string_view, if the string is not empty Sentencepiece will add a ' '.
    const std::string &text_buffer = text.empty() ? text : (' ' + text);
    std::string_view padded_text{text_buffer};

    std::vector<yami_sp_symbol> symbols;
    symbols.reserve(padded_text.size());

    for (usize i = 0, counter = 0; i < padded_text.size(); ++counter) {
        const int tok_len = utf8_len(padded_text[i]);
        const size prev = ((size) counter) - 1;
        const size next = i + tok_len >= padded_text.size() ? -1 : counter + 1; // EOS = -1
        symbols.emplace_back(yami_sp_symbol{prev, next, padded_text.substr(i, tok_len)});

        i += tok_len;
    }

    for (usize i = 1; i < symbols.size(); ++i) {
        maybe_add(symbols, i - 1, i);
    }

    while (!agenda_.empty()) {
        const yami_sp_bigram top = agenda_.top();
        agenda_.pop();

        yami_sp_symbol &top_left = symbols[top.left];
        yami_sp_symbol &top_right = symbols[top.right];
        
        // If already merged, continue
        if (top_left.piece.empty() || top_right.piece.empty() ||
            top_left.piece.size() + top_right.piece.size() != top.len)
            continue;
        
        // Left will become the merged symbol
        top_left.piece = std::string_view{top_left.piece.data(), top_left.piece.size() + top_right.piece.size()};
        top_right.piece = std::string_view{""};
        
        // Update next/prev references
        top_left.next = top_right.next;
        if (top_right.next >= 0)
            symbols[top_right.next].prev = top.left;

        // Add new pairs
        maybe_add(symbols, top_left.prev, top.left);
        maybe_add(symbols, top.left, top_left.next);
    }

    std::vector<int> res;

    if (bos) res.push_back(bos_id);

    for (size i = 0; i != -1; i = symbols[i].next) {
        const yami_sp_symbol &symbol = symbols[i];
        const auto &tok_it = tok_to_id_.find(std::string(symbol.piece));
        if (tok_it == tok_to_id_.end()) {
            // Fixme: I still don't know if this could happen...
            YAMI_ASSERT(false);
        } else {
            res.push_back(tok_it->second);
        }
    }
    return res;
}

std::string yami_llama_tokenizer::decode(const int idx) const noexcept {
    return this->id_to_tok_[idx];
}

void yami_llama_tokenizer::maybe_add(const std::vector<yami_sp_symbol> &symbols,
                                     const size left, const size right) noexcept {
    if (left < 0 || right < 0)
        return;

    const yami_sp_symbol &symbols_left = symbols[left];
    const yami_sp_symbol &symbols_right = symbols[right];

    std::string_view piece{symbols_left.piece.data(), symbols_left.piece.size() + symbols_right.piece.size()};

    const auto &score_it = tok_to_score_.find(std::string(piece));
    if (score_it != tok_to_score_.end()) {
        agenda_.push(yami_sp_bigram{left, right, score_it->second, piece.size()});
    }
}

static void load_tokenizer(yami_model *model, const char *tokenizer_file) noexcept {
    FILE *f = open_and_check(tokenizer_file, yami_tokenizer_header);

    yami_tokenizers type;
    read(&type, sizeof(yami_tokenizers), 1, f);

    if (type != yami_tokenizers::BPE && type != yami_tokenizers::SP) {
        fclose(f);
        YAMI_LOG_FATAL("unknown tokenizer type %d", type);
    }
    model->tokenizer = type;

    YAMI_LOG_INFO("loading %s tokenizer from \"%s\"", model->tokenizer == yami_tokenizers::BPE ? "BPE" : "SP", tokenizer_file);

    std::string tok;
    switch (model->tokenizer) {
        case yami_tokenizers::BPE: {
            u32 len;
            u16 tok_size;
            {
                // Encoder
                read(&len, sizeof(u32), 1, f);
                model->encoder.reserve(len);
                for (u32 i = 0; i < len; ++i) {
                    tok.clear();
                    read(&tok_size, sizeof(u16), 1, f);
                    tok.resize(tok_size);
                    read(tok.data(), 1, tok_size, f);
                    model->encoder.emplace_back(tok);
                }
            }

            {
                // Vocab
                read(&len, sizeof(u32), 1, f);
                model->vocab.reserve(len);

                for (u32 i = 0; i < len; ++i) {
                    tok.clear();
                    read(&tok_size, sizeof(u16), 1, f);
                    tok.resize(tok_size);
                    read(tok.data(), 1, tok_size, f);
                    model->vocab.emplace_back(tok);
                }
            }
            break;
        }
        case yami_tokenizers::SP: {
            u32 vocab_size;
            read(&vocab_size, sizeof(u32), 1, f);

            model->vocab.reserve(vocab_size);
            model->scores.reserve(vocab_size);

            u16 tok_size;
            f32 score;
            for (u32 i = 0; i < vocab_size; ++i) {
                tok.clear();
                read(&tok_size, sizeof(u16), 1, f);
                tok.resize(tok_size);
                read(tok.data(), 1, tok_size, f);
                read(&score, sizeof(f32), 1, f);
                model->vocab.emplace_back(std::move(tok));
                model->scores.emplace_back(score);
            }
            break;
        }
        default: {
            YAMI_LOG_FATAL("unknown tokenizer %d", model->tokenizer);
        }
    }
    fclose(f);
}

struct tensor_metadata {
    char label[YAMI_LABEL_SIZE];
    u8 dtype, n_dims;
    u64 dimensions[YAMI_MAX_DIMS];
    u64 offset;
};

void yami_load_model(yami_context *ctx, yami_model *model,
                     const char *model_file, const char *tokenizer_file,
                     const bool use_mmap) noexcept {
    load_tokenizer(model, tokenizer_file);
    if (use_mmap)
        model->mmap = std::make_unique<yami_mmap>(model_file);

    FILE *f = fopen(model_file, "rb");
    YAMI_ASSERT(f != nullptr);

    // There is no need to check the header as it's already checked by mmap
    fseek(f, YAMI_HEADER_SIZE, SEEK_SET);

    yami_models type;
    read(&type, sizeof(yami_models), 1, f);

    if(type != yami_models::GPT2 && type != yami_models::LLAMA) {
        fclose(f);
        YAMI_LOG_FATAL("unknown model type %d", type);
    }

    model->type = type;

    YAMI_LOG_INFO("loading %s model from \"%s\"", type == yami_models::LLAMA ? "LLaMA" : "GPT2", model_file);

    u16 hp_size;
    read(&hp_size, sizeof(u16), 1, f);

    read(model->hparams, 1, hp_size, f);

    u16 n_tensors;
    read(&n_tensors, sizeof(u16), 1, f);

    std::vector<tensor_metadata> tensors_metadata;
    tensors_metadata.resize(n_tensors);
    for (u16 i = 0; i < n_tensors; ++i) {
        read(&tensors_metadata[i].label, 1, 64, f);
        read(&tensors_metadata[i].dtype, 1, 1, f);
        read(&tensors_metadata[i].n_dims, 1, 1, f);
        read(&tensors_metadata[i].dimensions, sizeof(u64), YAMI_MAX_DIMS, f);
        read(&tensors_metadata[i].offset, sizeof(u64), 1, f);
    }

    std::unordered_map<std::string, yami_tensor *> tensors;
    for (u16 i = 0; i < n_tensors; ++i) {
        const tensor_metadata &metadata = tensors_metadata[i];
        yami_tensor *tensor;
        if (use_mmap) {
            tensor = yami_new_tensor(ctx,
                                     (int) metadata.n_dims,
                                     metadata.dimensions,
                                     metadata.label,
                                     (byte *) model->mmap->data + metadata.offset);
        } else {
            tensor = yami_new_tensor(ctx,
                                     (int) metadata.n_dims,
                                     metadata.dimensions,
                                     metadata.label);
            read(tensor->data, sizeof(f32), tensor->ne, f);
        }
        tensors[tensor->label] = tensor;
    }
    model->tensors = std::move(tensors);
    fclose(f);
}

static usize parse_mem_arg(const char *const mem_str) noexcept {
    char *multiplier;
    usize mem = (usize) ::strtol(mem_str, &multiplier, 10);
    if (errno == ERANGE || errno == EINVAL) {
        YAMI_LOG_FATAL("\"%s\" is not a valid memory size", mem_str);
    }

    switch (multiplier[0]) {
        case 0:
            break;
        case 'K':
            mem *= 1024L;
            break;
        case 'M':
            mem *= 1024 * 1024L;
            break;
        case 'G':
            mem *= 1024 * 1024 * 1024L;
            break;
        default:
            YAMI_LOG_FATAL("unknown multiplier \"%c\"", multiplier[0]);
    }

    return mem;
}

void yami_arg_parse(int argc, char **argv, yami_model_settings *settings) noexcept {
    settings->n_workers = 1;
    settings->n_tokens = 100;
    settings->temperature = 1.f;
    settings->model_file = "yami_model.bin";
    settings->tokenizer_file = "yami_tokenizer.bin";
    settings->main_ctx_size = 1024*1024*1024L;
    settings->scratch_ctx_size = 1024*1024*1024L;
    settings->seed = time(nullptr);
    settings->top_k = 0;
    settings->use_mmap = true;

    for (int i = 1; i < argc; ++i) {
        if ((i+1) < argc) {
            if (strcmp("-i", argv[i]) == 0 || strcmp("--input", argv[i]) == 0) {
                settings->prompt = argv[++i];
            } else if (strcmp("-w", argv[i]) == 0 || strcmp("--workers", argv[i]) == 0) {
                const int n_workers = (int) ::strtol(argv[++i], nullptr, 10);
                if (errno == ERANGE || errno == EINVAL) {
                    YAMI_LOG_FATAL("\"%s\" is not a valid number of workers", argv[i]);
                }
                settings->n_workers = n_workers;
            } else if (strcmp("-t", argv[i]) == 0 || strcmp("--temp", argv[i]) == 0) {
                const f32 temp = ::strtof(argv[++i], nullptr);
                if (errno == ERANGE) {
                    YAMI_LOG_FATAL("\"%s\" is not a valid temperature", argv[i]);
                }
                settings->temperature = temp;
            } else if (strcmp("-k", argv[i]) == 0 || strcmp("--top-k", argv[i]) == 0) {
                const int top_k = (usize) ::strtol(argv[++i], nullptr, 10);
                if (errno == ERANGE || errno == EINVAL) {
                    YAMI_LOG_FATAL("\"%s\" is not a valid value for K", argv[i]);
                }
                settings->top_k = top_k;
            } else if (strcmp("-m", argv[i]) == 0 || strcmp("--model", argv[i]) == 0) {
                settings->model_file = argv[++i];
            } else if (strcmp("-T", argv[i]) == 0 || strcmp("--tokenizer", argv[i]) == 0) {
                settings->tokenizer_file = argv[++i];
            } else if (strcmp("-s", argv[i]) == 0 || strcmp("--seed", argv[i]) == 0) {
                const usize seed = (usize) ::strtol(argv[++i], nullptr, 10);
                if (errno == ERANGE || errno == EINVAL) {
                    YAMI_LOG_FATAL("\"%s\" is not a valid seed", argv[i]);
                }
                settings->seed = seed;
            } else if (strcmp("-n", argv[i]) == 0 || strcmp("--new-tokens", argv[i]) == 0) {
                const int n = (int) ::strtol(argv[++i], nullptr, 10);
                if (errno == ERANGE || errno == EINVAL) {
                    YAMI_LOG_FATAL("\"%s\" is not a valid number of tokens", argv[i]);
                }
                settings->n_tokens = n;
            } else if (strcmp("-M", argv[i]) == 0 || strcmp("--main-mem", argv[i]) == 0) {
                settings->main_ctx_size = parse_mem_arg(argv[++i]);
            } else if (strcmp("-S", argv[i]) == 0 || strcmp("--scratch-mem", argv[i]) == 0) {
                settings->scratch_ctx_size = parse_mem_arg(argv[++i]);
            }
        } else if (strcmp("--no-mmap", argv[i]) == 0) {
            settings->use_mmap = false;
        } else if (strcmp("--help", argv[i]) == 0) {
            printf("Usage: %s [options]\nOptions:\n", argv[0]);
            printf("  --help\t\tDisplay this message.\n");
            printf("  -i, --input\t\tInput prompt.\n");
            printf("  -n, --new-tokens\tNumber of new tokens to generate (default=100).\n");
            printf("  -w, --workers\t\tNumber of workers to use (default=1).\n");
            printf("  -t, --temp\t\tModel temperature (default=1.0).\n");
            printf("  -k, --top-k\t\tTop k sampling (default=disabled).\n");
            printf("  -m, --model\t\tPath to the model file (default=yami_model.bin).\n");
            printf("  -T, --tokenizer\tPath to the model file (default=yami_tokenizer.bin).\n");
            printf("  -s, --seed\t\tSeed to use for generation (default=time).\n");
            printf("  -M, --main-mem\tMemory to allocate for the main context, must be enough to store the model weights (default=1G).\n");
            printf("  -S, --scratch-mem\tMemory to allocate for the scratch context used to store intermediate results (default=1G).\n");
            printf("  --no-mmap\t\tDon't use the mmap function to load the weights (default=yes).\n");

            exit(EXIT_SUCCESS);
        }
    }

    if (settings->prompt.empty()) {
        YAMI_LOG_FATAL("missing input prompt");
    }
}

// Fixme: I don't really like the idea of allocating a dynamic array this big (50K elements),
//  a possible solution could be to use alloca + qsort and then simply return the first K elements as a vector.
std::vector<yami_token> yami_top_k(const f32 *values, const usize ne,
                                   const usize k) noexcept {
    std::vector<yami_token> res;
    res.reserve(ne);

    for (usize i = 0; i < ne; ++i)
        res.emplace_back(yami_token{values[i], i});

    std::sort(res.begin(), res.end(), [](const yami_token &xa, const yami_token &xb) {
       return xa.value < xb.value;
    });

    res.resize(k);
    return res;
}