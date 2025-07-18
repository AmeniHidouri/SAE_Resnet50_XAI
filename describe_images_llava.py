from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import json
import os

# Modèle LLaVA HF
model_id = "llava-hf/llava-1.5-7b-hf"

print("Chargement du modèle...")
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Dossier contenant les images
image_folder = "/home/ameni/Documents/PASTA_code-main/datasets/train/images"
output_file = "llava_descriptions.json"

# Dictionnaire pour stocker les descriptions
descriptions = {}

# Extensions d'images valides
valid_extensions = ('.png', '.jpg', '.jpeg')

# Parcourir les fichiers dans le dossier
for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(valid_extensions):
        continue
    
    image_path = os.path.join(image_folder, image_name)
    print(f"Traitement de l'image : {image_name}")
    
    # Charger l'image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        continue
    
    # Préparer la conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    
    # Appliquer le modèle
    inputs = processor(
        text=processor.apply_chat_template(conversation, add_generation_prompt=True),
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.device, torch.float16)
    
    # Générer la description
    try:
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        description = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        # Nettoyer la description
        description_clean = description.split("What is shown in this image?")[-1].strip()
        
        # Stocker la description
        descriptions[image_name] = description_clean
    except Exception as e:
        print(f"Erreur lors de la génération de la description pour {image_name}: {e}")
        continue

# Enregistrement dans JSON
try:
    with open(output_file, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"\nDescriptions enregistrées dans : {output_file}")
except Exception as e:
    print(f"Erreur lors de l'enregistrement du fichier JSON : {e}")