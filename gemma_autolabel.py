from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import pandas as pd
import requests
import torch

model_path = '/group-volume/Sentinel/LLMs/HH/google/gemma-3-12b-it'
prompt_path = './gemma-prompt.txt'

model = Gemma3ForconditionalGeneration.from_pretrained(
  model_path, local_files_only=True, device_map='auto').eval()
processor.chat_template = open(prompt_path, 'r', encoding='utf-8').read()

def moderate(messages):
  inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensros='pt"
  ).to(model.device, dtype=torch.bfloat16)

  input_len = inputs["input_ids"].shape[-1]

  with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

train = pd.read_json('sentinel_i_v0.1.0_trainval.jsonl', lines=True)
train['autolabel_gemma_result'] = ''

for i in range(len(train)):
  image =image.open(f"/group-volume/Sentinel-I/dataset/train_dataset/{train.iloc[i]['image_path']}")
  messages = [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
      "role": "user"
      "content": [
        {"type": "image", "image": image},
        {"type": "textg", "text": "."}
      ]
    }
  ]
  train.iloc[i, 6] = moderate(messages)

train.to_json('sentinel_i_v0.1.0_trainval_autolabel_gemma_3_12b.jsonl', orient='records', lines=True)
