import sys

import torch

from config import device
from model import BabyGPT
from tokenizer import decode, encode, vocab_size

model = BabyGPT(vocab_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "How are you"
full_prompt = prompt + " => "
idx = torch.tensor([encode(full_prompt)], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=100)

generated = decode(out[0].tolist())[len(full_prompt):]
translation = generated.split("\n")[0].strip()
print(translation)
