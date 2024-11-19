# Copyright (c) 2023-2024, Christian Gilli <christian.gilli11@gmail.com>
# All rights reserved.
#
# This code is licensed under the terms of the MIT license
# (https://opensource.org/license/mit).

import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import sys
from pathlib import Path
from tokenizer import LlamaTokenizer
from typing import Dict, Tuple
import math
import time
import os
import json

# Relative path to the folder obtained by using the 'download.sh' script at https://github.com/facebookresearch/llama
MODEL_FOLDER = 'examples/llama2/llama-2-7b'

# Implementation heavily inspired by the great https://github.com/karpathy/llama2.c

@dataclass
class LlamaHparams:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    # MLP hidden layer size will be multiple of
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 2048

    @classmethod
    def from_meta(cls, params: Dict, vocab_size: int):
        res = cls()
        res.dim = params['dim']
        res.n_layers = params['n_layers']
        res.n_heads = params['n_heads']
        res.norm_eps = params['norm_eps']
        res.vocab_size = vocab_size
        return res


# In YAMI it should be enough to precompute the sin and cos lookup tables
def _compute_freqs(dim: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1. / (10_000. ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _reshape_freqs(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Taken straight from the original LLaMA GitHub repository
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(shape)


def _rope(
        q: torch.Tensor,
        k: torch.Tensor,
        freq_cos: torch.Tensor,
        freq_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)

    freq_cos = _reshape_freqs(freq_cos, q_r)
    freq_sin = _reshape_freqs(freq_sin, q_r)

    q_out_r = q_r * freq_cos - q_i * freq_sin
    q_out_i = q_r * freq_sin + q_i * freq_cos
    k_out_r = k_r * freq_cos - k_i * freq_sin
    k_out_i = k_r * freq_sin + k_i * freq_cos

    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(3)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(3)

    return q_out.type_as(q), k_out.type_as(k)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, norm_eps: float):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        return out * self.weight


class Attention(nn.Module):

    def __init__(self, params: LlamaHparams):
        super().__init__()
        self.n_kv_heads = params.n_heads
        self.n_heads = params.n_heads
        self.head_size = params.dim // params.n_heads
        self.wq = nn.Linear(params.dim, self.n_heads * self.head_size, bias=False)
        self.wk = nn.Linear(params.dim, self.n_kv_heads * self.head_size, bias=False)
        self.wv = nn.Linear(params.dim, self.n_kv_heads * self.head_size, bias=False)
        self.wo = nn.Linear(params.n_heads * self.head_size, params.dim, bias=False)
    
        # Attention mask, should use Flash Attention
        mask = torch.full((1, 1, params.max_seq_len, params.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor, freq_cos: torch.Tensor, freq_sin: torch.Tensor):
        block_size, seq_len, _ = x.shape
        
        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(block_size, seq_len, self.n_heads, self.head_size)
        k = k.view(block_size, seq_len, self.n_kv_heads, self.head_size)
        v = v.view(block_size, seq_len, self.n_kv_heads, self.head_size)
        
        # Apply RoPE encoding
        q, k = _rope(q, k, freq_cos, freq_sin)

        # Do not use grouped multiquery attention

        # (block_size, n_heads, seq_len, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_size)
        scores = scores + self.mask[:, :, :seq_len, :seq_len]
        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(block_size, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):

    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(self, id: int, params: LlamaHparams):
        super().__init__()
        self.id = id
        self.n_heads = params.n_heads
        self.dim = params.dim
        self.head_size = params.dim // params.n_heads
        self.attention = Attention(params)
        self.feed_forward = FeedForward(params.dim, params.multiple_of)
        self.attention_norm = RMSNorm(params.dim, params.norm_eps)
        self.ffn_norm = RMSNorm(params.dim, params.norm_eps)

    def forward(self, x, freq_cos, freq_sin):
        h = x + self.attention.forward(self.attention_norm(x), freq_cos, freq_sin)
        return h + self.feed_forward.forward(self.ffn_norm(h))
    

class LlamaModel(nn.Module):

    def __init__(self, params: LlamaHparams):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, params) for layer_id in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        
        freq_cos, freq_sin = _compute_freqs(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

    def forward(self, tokens: torch.Tensor):
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        freq_cos = self.freq_cos[:seq_len]
        freq_sin = self.freq_sin[:seq_len]
        for layer in self.layers:
            h = layer(h, freq_cos, freq_sin)
        h = self.norm(h)

        # logits
        return self.output(h[:, [-1], :])
    
    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        # idx is the input tensor containing the encoding of the prompt. It's supposed to be a 1xN tensor.
        for _ in range(max_new_tokens):
            idx_real = idx if len(idx) < self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_real)

            # Apply temperature to the last row of each batch
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_from_meta():
    with open(os.path.join(MODEL_FOLDER, 'params.json')) as f:
        params = json.load(f)
    
    checkpoint_path = list(Path(MODEL_FOLDER).glob('consolidated.*.pth'))[0]
    return torch.load(checkpoint_path), params
    

if __name__ == '__main__':
    model_dict, params = load_from_meta()
    hparams = LlamaHparams.from_meta(params, model_dict['tok_embeddings.weight'].shape[0])
    model = LlamaModel(hparams)
    model.load_state_dict(model_dict, strict=False)
    del model_dict

    model.eval()
    print('LLaMA2 model loaded successfully!')
    tokenizer = LlamaTokenizer(f'{MODEL_FOLDER}/tokenizer.model')
    prompt = 'Javascript is a programming language designed'
    print(f'[In]: {prompt}')
    
    idx = tokenizer.encode(prompt, bos=True, eos=False)
    MAX_TOKENS = 10
    with torch.no_grad():
        start = time.perf_counter()
        response_tokens = model.generate(torch.tensor(idx, dtype=torch.long)[None, ...], max_new_tokens=MAX_TOKENS)
        delay = time.perf_counter() - start
        print(f'[Out]: {tokenizer.decode(response_tokens[0].tolist())}')
        print(f'Took {round(delay, 2)}s to generate {MAX_TOKENS} tokens ({round(MAX_TOKENS / delay, 2)} tok/s)')        