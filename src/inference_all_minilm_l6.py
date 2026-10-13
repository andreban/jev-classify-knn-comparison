import torch
import sys
from all_minilm_l6.toxicity_model import ToxicityModel
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # vec is 384
input = ' '.join(sys.argv[1:])
embeddings = model.encode([input])
embeddings = torch.tensor([embeddings[0]])

model = ToxicityModel()
model.load_state_dict(torch.load("SP-all-MiniLM-L6-v2.safetensors", weights_only=True, map_location=torch.device('cpu')))
model.eval()

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
result = F.sigmoid(model(embeddings)) >= 0.5

print(f'Input: "{input}"')
for (i, label) in enumerate(labels):
    print(f'\t - {label}: {result[0][i]}')
