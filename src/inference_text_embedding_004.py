import os
import torch
import sys
from dotenv import load_dotenv
from text_embedding_004.toxicity_model import ToxicityModel
from torch.nn import functional as F
import google.genai as genai
from google.genai import types

load_dotenv()
GOOGLE_GENAI_API_KEY = os.environ["GEMINI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-004"
OUTPUT_DIMENSIONALITY = 768

genai = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

input = ' '.join(sys.argv[1:])
embedding_response = genai.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=input,
    config=types.EmbedContentConfig(task_type="CLASSIFICATION")
)

embeddings = torch.tensor([embedding_response.embeddings[0].values])

model = ToxicityModel()
model.load_state_dict(torch.load("toxicity.safetensors", weights_only=True, map_location=torch.device('cpu')))
model.eval()

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
result = F.sigmoid(model(embeddings)) >= 0.5

print(f'Input: "{input}"')
for (i, label) in enumerate(labels):
    print(f'\t - {label}: {result[0][i]}')
