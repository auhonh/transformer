import torch

batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0

lr = 1e-3
max_iters = 5000
eval_interval = 500
eval_iters = 200

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
