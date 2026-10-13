import torch
import torch.nn as nn
import torch.optim as optim

from all_minilm_l6.toxicity_model import ToxicityModel
from toxicity_training import train_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
torch.set_default_device(device)
torch.manual_seed(1)

model = ToxicityModel()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)
loss_fn = nn.BCEWithLogitsLoss()

train_model(
    device = device,
    input_filename='SP-all-MiniLM-L6-v2.jsonl',
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    batch_size=512,
    epochs=200
)

torch.save(model.state_dict(), 'SP-all-MiniLM-L6-v2.safetensors')