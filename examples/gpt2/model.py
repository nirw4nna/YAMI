import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bpe import Encoder
from torch.autograd import profiler
import time
from dataclasses import dataclass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
# Simple trick to import a file from outside the default path.
# To enable Pylance add this also to "Python > Analysis > Extra Paths"
from convert_yami import *


_MODEL_TYPE = 'gpt2'


@dataclass
class GPT2Hparams(Hparams):
   # default hyperparameters for GPT-2 small
   n_layers: int = 12
   n_heads: int = 12
   emb_size: int = 768
   block_size: int = 1024
   vocab_size: int = 50257


class LayerNorm(nn.Module):

  def __init__(self, n_features, epsilon=1e-5):
    super().__init__()
    self.epsilon = epsilon
    self.register_parameter('weight', nn.Parameter(torch.ones((n_features))))
    self.register_parameter('bias', nn.Parameter(torch.zeros((n_features))))

  def forward(self, input):
    mean = input.mean(-1, keepdim=True)
    var = input.var(-1, keepdim=True)
    
    self.out = (input - mean) / (var + self.epsilon)**0.5
    self.out = self.out * self.weight + self.bias
    
    return self.out


# Activation function used in GPT2
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiheadAttention(nn.Module):
    def __init__(self, hparams: GPT2Hparams):
        super().__init__()
        self.block_size = hparams.block_size
        self.emb_size = hparams.emb_size
        self.n_heads = hparams.n_heads
        # Stacked attention, contains the projections of both Q, K and V
        self.c_attn = nn.Linear(self.emb_size, 3 * self.emb_size)
        self.c_proj = nn.Linear(self.emb_size, self.emb_size)
        # Causal mask
        self.tril = torch.tril(torch.ones(self.block_size, self.block_size).view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.shape # (block size, context size, emb size)
        # assert emb_size % n_head == 0
        attn = self.c_attn(x)
        q, k, v = attn.split(self.emb_size, dim=2) # (B, T, C)
        q = q.view(B, T, self.n_heads, self.emb_size // self.n_heads).transpose(1, 2) # (B, nh, T, hs) given EMB_SIZE is a multiple of N_HEADS
        k = k.view(B, T, self.n_heads, self.emb_size // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.emb_size // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        k_t = k.transpose(-2, -1)
        # Self Attention (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        q_k = q @ k_t
        attention = q_k * q.size(-1) ** -0.5
        attention = attention.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        out = attention @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hparams: GPT2Hparams):
        super().__init__()
        self.ln_1 = LayerNorm(hparams.emb_size)
        self.attn = MultiheadAttention(hparams)
        self.ln_2 = LayerNorm(hparams.emb_size)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(hparams.emb_size, hparams.emb_size * 4),
            c_proj = nn.Linear(hparams.emb_size * 4, hparams.emb_size),
            act = GELU()
        ))

    def forward(self, x):
        m = self.mlp
        x = x + self.attn(self.ln_1(x))
        x = x + m.c_proj(m.act(m.c_fc(self.ln_2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, hparams: GPT2Hparams):
        super().__init__()
        self.hparams = hparams
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(hparams.block_size, hparams.emb_size), # token embedding
            wte = nn.Embedding(hparams.vocab_size, hparams.emb_size), # positional embedding
            h = nn.ModuleList([TransformerBlock(hparams) for _ in range(hparams.n_layers)]),
            ln_f = LayerNorm(hparams.emb_size)
        ))
        self.lm_head = nn.Linear(hparams.emb_size, hparams.vocab_size, bias=False)

        n_params = sum([p.numel() for p in self.transformer.parameters()])
        n_params += sum([p.numel() for p in self.lm_head.parameters()])
        print(f'Model has {round(n_params / 1e6)}M parameters')

    @classmethod
    def from_hf(cls, hparams: GPT2Hparams()):
        hf_model = GPT2LMHeadModel.from_pretrained(_MODEL_TYPE)
        hf_data = hf_model.state_dict()
        # GPT2 uses Conv1D instead of a Linear layer which means we have to transpose the weights
        to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        my_model = GPT(hparams)
        my_data = my_model.state_dict()
        assert len(my_data) == len(hf_data)
        for k, v in hf_data.items():
            if any([k.endswith(t) for t in to_transpose]):
                # v.shape in reverse order (transpose)
                assert v.shape[::-1] == my_data[k].shape
                with torch.no_grad():
                    my_data[k].copy_(v.t())
            else:
                assert v.shape == my_data[k].shape
                with torch.no_grad():
                    my_data[k].copy_(v)

        return my_model
    
    @staticmethod
    def export(model_file: str, tokenizer_file: str):
        hparams = GPT2Hparams()
        model = GPT.from_hf(hparams)
        model.eval()
        tokenizer = Encoder.from_pretrained()
        export_model(model_file, Model.GPT2, model.state_dict(), hparams,
                     ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight', 'lm_head.weight'])
        export_tokenizer(tokenizer_file, Tokenizer.BPE, encoder=tokenizer.encoder, vocab=tokenizer.bpe_ranks)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(T))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, sample=True):
        for _ in range(max_new_tokens):
            start_ = time.perf_counter()
            idx_real = idx if len(idx) < self.hparams.block_size else idx[:, -self.hparams.block_size:]
            # with profiler.profile(record_shapes=True) as prof:
            #     with profiler.record_function('model_inference'):
            logits = self(idx_real)

            # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
            # matmul_events = [event for event in prof.function_events if "addmm" in event.name or "mm" in event.name]
            # unique = set([e.name for e in prof.function_events])
            # print(unique)
            # print(f'\n\nNext token computation took {prof.self_cpu_time_total/1000.}ms:')
            # total_time_us = 0
            # for i, me in enumerate(prof.function_events):
            #     print(f'\t{i}) {me.name} shapes={me.input_shapes} time={me.self_cpu_time_total_str}')
            #     total_time_us += me.self_cpu_time_total

            # print(f'Performing {len(matmul_events)} matmuls took {total_time_us / 1000.}ms')
            # Apply temperature to the last row of each bach
            logits = logits[:, -1, :] / temp
            # top_k?
            v, _ = torch.topk(logits, 10)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
            print(f'1 tok took {(time.perf_counter() - start_) * 1000.}ms')

        return idx


if __name__ == '__main__':
    # GPT.export('yami_model.bin', 'yami_tokenizer.bin')
    gpt = GPT2LMHeadModel.from_pretrained(_MODEL_TYPE)
    n_tokens = 100
    gpt.eval()
    tokenizer = Encoder.from_pretrained()
    prompt = "Building a website can be done in ten simple steps"
    encoded_prompt = [tokenizer.encode(prompt)]
    idx = torch.tensor(encoded_prompt)
    start = time.perf_counter()
    resp = tokenizer.decode(gpt.generate(idx, max_new_tokens=n_tokens).cpu().squeeze().tolist())
    print(f'GPT2: {resp}\n')
    print(f'Time per token: {(time.perf_counter() - start) / n_tokens * 1000.}ms')
