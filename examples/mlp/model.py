import torch
import torch.nn as nn
from torch.nn import functional as F
import time
# import matplotlib.pyplot as plt

torch.manual_seed(1337)

INFERENCE = True
EXPORT_MODEL = False
MODEL_FILE = 'simple_mlp.ptrch'

# Hyperparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_emb_dim = 14
hidden_size = 256
batch_size = 64
block_size = 8
training_steps = 10_000

# Build the dataset
words = open('names.txt', 'r').read().splitlines()
S_TOK = '.'
characters = sorted(list(set(''.join(words))))
chtoi = {c:i+1 for i, c in enumerate(characters)}
chtoi[S_TOK] = 0
itoch = {i:c for c, i in chtoi.items()}


def build_dataset(words):
  # prepare data
  X, Y = [], []

  for w in words:
      context = [0] * block_size
      for c in w + S_TOK:
          c_idx = chtoi[c]
          X.append(context)
          Y.append(c_idx)
          context = context[1:] + [c_idx]

  X = torch.tensor(X, device=device)
  Y = torch.tensor(Y, device=device)
  return X, Y


vocab_size = len(itoch)
print(f'Vocabulary size: {vocab_size}')

X_train, Y_train = build_dataset(words[:int(0.9*len(words))])

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 14, device=device)
        self.layers = nn.Sequential(
            nn.Linear(112, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 27, device=device)
        )
    
    def forward(self, idx, targets=None):
        tok_emb = self.emb(idx).view(-1, 112)
        logits = self.layers(tok_emb)
        
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self):
        out = []
        ctx = [0] * block_size
        while True:
          logits, loss = self(torch.tensor([ctx], device=device))
          probs = F.softmax(logits, dim=1)
          next_idx = torch.multinomial(probs, num_samples=1, replacement=True).item()
          next_tok = itoch[next_idx]
          if next_tok == S_TOK:
            break
          out.append(next_tok)
          ctx = ctx[1:] + [next_idx]
    
        return ''.join(out)
    

model = Model()

if not INFERENCE:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(training_steps):
        b_idx = torch.randint(0, X_train.shape[0], (batch_size, ), device=device)
        xb = X_train[b_idx]
        yb = Y_train[b_idx]

        logits, loss = model(xb, yb)
        if step % 100 == 0:
            losses.append(loss.item())
            print(f'step {step} loss: {loss.item()}')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
else:
    # load the model from state_dict
    model.load_state_dict(torch.load(MODEL_FILE))

# sampling
start = time.monotonic_ns()
for _ in range(10):
  print(model.generate())
stop = time.monotonic_ns()
print(f'Generating 10 samples from the model took {(stop-start)/1_000_000}ms')
if EXPORT_MODEL:
    torch.save(model.state_dict(), MODEL_FILE)

# plt.plot(losses)
# plt.show()