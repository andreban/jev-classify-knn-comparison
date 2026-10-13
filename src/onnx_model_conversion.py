import torch
from all_minilm_l6.toxicity_model import ToxicityModel

torch_model = ToxicityModel()
torch_model.load_state_dict(torch.load("SP-all-MiniLM-L6-v2.safetensors", weights_only=True, map_location=torch.device('cpu')))
torch_model.eval()

example_inputs = (torch.randn(1, 384),)
# print(example_inputs.shape)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
onnx_program.optimize()

onnx_program.save("SP-all-MiniLM-L6-v2.onnx")

