import torch

batch_size = 64
block_size = 256
n_embd = 64
n_head = 4
n_layer = 3
dropout = 0.2

lr = 1e-3
max_iters = 6000
eval_interval = 300
eval_iters = 100

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
