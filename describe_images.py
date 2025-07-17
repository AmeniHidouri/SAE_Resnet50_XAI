import os
import torch
from PIL import Image
from tqdm import tqdm
import json

from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
image_folder = "/home/ameni/Documents/PASTA_code-main/datasets/train/images"  # Change si besoin 
output_file = "captions.json"

# 2. Chargement du modèle
print(" Chargement du modèle BLIP-2...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
model.eval()

# 3. Lecture des images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

captions = {}

# 4. Génération des descriptions
print(f"Génération des descriptions pour {len(image_files)} images...")
for image_name in tqdm(image_files):
    image_path = os.path.join(image_folder, image_name)
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions[image_name] = caption
    except Exception as e:
        print(f"Erreur pour {image_name} : {e}")
        captions[image_name] = "Erreur de traitement"

# 5. Sauvegarde dans un fichier JSON
with open(output_file, "w") as f:
    json.dump(captions, f, indent=4, ensure_ascii=False)

print(f"Descriptions sauvegardées dans {output_file}")
