import os
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from tqdm import tqdm

def precompute_images_resnet50(input_dir, output_dir):
    """
    Transforme les images d'un dataset pour ResNet50 et les sauvegarde sous forme de .npy
    """

    transform = ResNet50_Weights.DEFAULT.transforms()
    splits = ['train', 'validation', 'test']

    for split in splits:
        print(f"Traitement du dossier : {split}")
        split_dir = os.path.join(input_dir, split, 'images')
        label_file = os.path.join(input_dir, split, 'labels.txt')

        # Lire les labels
        with open(label_file, 'r') as f:
            lines = f.readlines()
        samples = [line.strip().split() for line in lines]

        images_np = []
        labels_np = []

        for image_name, label in tqdm(samples):
            image_path = os.path.join(split_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            images_np.append(image_tensor.numpy())
            labels_np.append(int(label))

        # Sauvegarder sous forme .npy
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f'images_{split}.npy'), np.array(images_np))
        np.save(os.path.join(output_dir, f'labels_{split}.npy'), np.array(labels_np))

        print(f"{split} : {len(images_np)} images sauvegardées.")
if __name__ == "__main__":
 precompute_images_resnet50(
     input_dir='./datasets',
     output_dir='./data_npy_resnet50'
 )
