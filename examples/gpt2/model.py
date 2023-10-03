import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bpe import Encoder

"""
GPT-2 small:
- 12 layers
- 12 head of attention
- 768 embedding size

GPT-2 large:
- 36 layers
- 20 head of attention
- 1280 embedding size
"""
# MODEL_TYPE = 'gpt2-large'
MODEL_TYPE = 'gpt2'
# Hyperparameters

# GPT-2 small
N_LAYERS = 12
N_HEADS = 12
EMB_SIZE = 768
BLOCK_SIZE = 1024
VOCAB_SIZE = 50257

# GPT-2 Large
# N_LAYERS = 36
# N_HEADS = 20
# EMB_SIZE = 1280
# BLOCK_SIZE = 1024
# VOCAB_SIZE = 50257

"""
LayerNorm is very similar to BatchNorm, the objective is always to normalize (0 mean, 1 std) the samples
but with BatchNorm we normalize a column, with LayerNorm we normalize the row.
What this means is that the implementation is actually quite simpler because there is no need to keep track
of the running mean and standard deviation, we can just compute them for each input we get.
"""
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
    def __init__(self):
       super().__init__()
       # Stacked attention, contains the projections of both Q, K and V
       self.c_attn = nn.Linear(EMB_SIZE, 3 * EMB_SIZE)
       self.c_proj = nn.Linear(EMB_SIZE, EMB_SIZE)
       # Causal mask
       self.tril = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE).view(1, 1, BLOCK_SIZE, BLOCK_SIZE))
    
    def forward(self, x):
        B, T, C = x.shape # (block size, context size, emb size)
        # assert emb_size % n_head == 0

        q, k, v = self.c_attn(x).split(EMB_SIZE, dim=2) # (B, T, C)
        q = q.view(B, T, N_HEADS, EMB_SIZE // N_HEADS).transpose(1, 2) # (B, nh, T, hs) given EMB_SIZE is a multiple of N_HEADS
        k = k.view(B, T, N_HEADS, EMB_SIZE // N_HEADS).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, N_HEADS, EMB_SIZE // N_HEADS).transpose(1, 2) # (B, nh, T, hs)

        # Self Attention (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        attention = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
        attention = attention.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        out = attention @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNorm(EMB_SIZE)
        self.attn = MultiheadAttention()
        self.ln_2 = LayerNorm(EMB_SIZE)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(EMB_SIZE, EMB_SIZE * 4),
            c_proj = nn.Linear(EMB_SIZE * 4, EMB_SIZE),
            act = GELU()
        ))

    def forward(self, x):
        m = self.mlp
        x = x + self.attn(self.ln_1(x))
        x = x + m.c_proj(m.act(m.c_fc(self.ln_2(x))))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
           wpe = nn.Embedding(BLOCK_SIZE, EMB_SIZE), # token embedding
           wte = nn.Embedding(VOCAB_SIZE, EMB_SIZE), # positional embedding
           h = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)]),
           ln_f = LayerNorm(EMB_SIZE)
        ))
        self.lm_head = nn.Linear(EMB_SIZE, VOCAB_SIZE, bias=False)
        
        n_params = sum([p.numel() for p in self.transformer.parameters()])
        n_params += sum([p.numel() for p in self.lm_head.parameters()])
        print(f'Model has {round(n_params / 1e6)}M parameters')

    @classmethod
    def from_hf(cls):
        hf_model = GPT2LMHeadModel.from_pretrained(MODEL_TYPE)
        hf_data = hf_model.state_dict()
        # GPT2 uses Conv1D instead of a Linear layer which means we have to transpose the weights
        to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        my_model = GPT()
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
            idx_real = idx if len(idx) < BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits = self(idx_real)
            # Apply temperature to the last row of each bach
            logits = logits[:, -1, :] / temp
            # top_k?
            probs = F.softmax(logits, dim=-1)
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


if __name__ == '__main__':
    gpt = GPT.from_hf()
    # gpt = GPT2LMHeadModel.from_pretrained(MODEL_TYPE)
    gpt.eval()
    # tokenizer = GPT2Tokenizer.from_pretrained(MODEL_TYPE)
    tokenizer = Encoder.from_pretrained()
    while True:
        proompt = input('YOU: ')
        # encoded_proompt = tokenizer(proompt, return_tensors='pt')
        encoded_proompt = [tokenizer.encode(proompt)]
        # idx = encoded_proompt['input_ids']
        idx = torch.tensor(encoded_proompt)
        resp = tokenizer.decode(gpt.generate(idx, max_new_tokens=50).cpu().squeeze().tolist())
        print(f'YAMI-GPT2: {resp}\n')
