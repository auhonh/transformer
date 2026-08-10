import torch

from config import device
from model import BabyGPT
from tokenizer import decode, vocab_size

model = BabyGPT(vocab_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=500)
print(decode(out[0].tolist()))
