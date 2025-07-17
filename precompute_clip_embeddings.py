import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

def prepare_and_embed(dataset_path="./datasets", output_dir="data_npy"):
    print("Chargement et partitionnement du dataset...")

    os.makedirs(dataset_path, exist_ok=True)
    dataset = load_dataset("ENSTA-U2IS/Cats_Dogs_Cars")
    full_data = dataset["train"]
    total_indices = list(range(len(full_data)))

    # Split 70% train, 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        total_indices, test_size=0.3, random_state=42, stratify=full_data["label"]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=[full_data["label"][i] for i in temp_idx]
    )

    dataset["train"] = full_data.select(train_idx)
    dataset["validation"] = full_data.select(val_idx)
    dataset["test"] = full_data.select(test_idx)

    # Sauvegarde des images et labels
    print("Sauvegarde des images et des labels...")
    for phase in ["train", "val", "test"]:
        os.makedirs(os.path.join(dataset_path, phase, "images"), exist_ok=True)

    phase_mapping = {"train": "train", "val": "validation", "test": "test"}

    for phase in ["train", "val", "test"]:
        split = phase_mapping[phase]
        images = dataset[split]["image"]
        labels = dataset[split]["label"]

        for i, img in enumerate(images):
            img_path = os.path.join(dataset_path, phase, "images", f"image_{i:06d}.jpg")
            img.save(img_path)

        np.save(os.path.join(dataset_path, phase, "labels.npy"), np.array(labels))

    # Génération des embeddings CLIP
    print("Génération des embeddings CLIP...")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for phase in ["train", "val", "test"]:
        split = phase_mapping[phase]
        images = dataset[split]["image"]
        labels = dataset[split]["label"]

        all_embeddings = []
        for img in tqdm(images, desc=f"Embedding {phase}"):
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().squeeze()
                all_embeddings.append(embedding)

        embeddings_array = np.stack(all_embeddings)
        labels_array = np.array(labels)

        # Sauvegardes standard avec prefixe "clip_"
        np.save(os.path.join(output_dir, f"clip_embedding_catsdogscars_{phase}.npy"), embeddings_array)
        np.save(os.path.join(output_dir, f"clip_labels_catsdogscars_{phase}.npy"), labels_array)

        # Sauvegardes avec noms attendus par le code PASTA (compatibilité)
        np.save(os.path.join(output_dir, f"X_catsdogscars_{phase}.npy"), embeddings_array)
        np.save(os.path.join(output_dir, f"y_catsdogscars_{phase}.npy"), labels_array)

    print("Embeddings CLIP et labels sauvegardés avec succès.")

if __name__ == "__main__":
    prepare_and_embed()
