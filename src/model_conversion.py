import ai_edge_torch
import torch
import torch.nn as nn
from all_minilm_l6.toxicity_model import ToxicityModel
from collections import OrderedDict


model = ToxicityModel() 
model.load_state_dict(torch.load("SP-all-MiniLM-L6-v2.safetensors", weights_only=True, map_location=torch.device('cpu')))

sample_inputs = torch.randn(1, 384)
edge_model = ai_edge_torch.convert(model.eval(), (torch.randn(1, 384),))

edge_output = edge_model(*(sample_inputs),)
torch_output = model.forward(sample_inputs)
print(sample_inputs)

print(edge_output)
print(torch_output)

# Save converted model to disk.
edge_model.export('./SP-all-MiniLM-L6-v2.tflite')
