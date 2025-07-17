import os
import shutil
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm # Pour une barre de progression
import numpy as np
# --- Configuration ---
# Nom du dataset sur Hugging Face (remplacez par le nom exact de votre dataset)
# Par exemple, "Bingsu/cats_vs_dogs" ou "wangr/cars_vs_trucks"
# Vous devrez vérifier le nom exact de votre dataset "catsdogscars" sur le Hub.
# Si votre dataset est juste "ImageFolder" localement, cette partie sera différente.
HF_DATASET_NAME = "ENSTA-U2IS/Cats_Dogs_Cars" # Exemple, à remplacer par votre dataset réel
# Ou si c'est un dataset ImageFolder local déjà chargé :
# HF_DATASET_PATH = "/chemin/vers/votre/dossier/ImageFolder_racine"

# Dossier de sortie où le nouveau dataset sera créé
OUTPUT_ROOT_DIR = "/home/ameni/Documents/PASTA_code-main/datasets/catsdogscars_partitionned"

# Ratios de split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1 # Le reste

# Assurez-vous que la somme des ratios est 1.0
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Les ratios de split doivent sommer à 1.0"

# --- 1. Charger le Dataset depuis Hugging Face ---
print(f"Chargement du dataset '{HF_DATASET_NAME}' depuis Hugging Face...")
# Si votre dataset est déjà en 3 splits (train/validation/test) sur HF, vous pouvez le charger directement
# Si ce n'est pas le cas, vous devrez peut-être charger un split unique (ex: 'train') et le re-splitter.

# Pour un dataset qui a un split unique (ex: 'train') et que nous allons re-splitter
try:
    dataset = load_dataset(HF_DATASET_NAME, split='train')
    print(f"Dataset chargé. Nombre total d'images: {len(dataset)}")
    
    # Récupérer les noms des classes (labels)
    # Les datasets HF ont souvent un objet 'features' qui contient la description des colonnes.
    # Pour les datasets d'images, le label est souvent un ClassLabel.
    if 'label' in dataset.features and hasattr(dataset.features['label'], 'names'):
        class_names = dataset.features['label'].names
    else:
        # Si 'label' n'est pas un ClassLabel, ou si les noms ne sont pas là,
        # vous devrez peut-être les déduire manuellement ou savoir ce qu'ils sont.
        # Par exemple, si vos labels sont 0, 1, 2, et que vous savez qu'ils correspondent à 'cars', 'cats', 'dogs'
        # class_names = ['cars', 'cats', 'dogs']
        # Pour ce script, nous allons nous baser sur le comportement standard de Hugging Face.
        print("Avertissement: Impossible de déduire les noms de classes automatiquement du dataset Hugging Face 'label' feature.")
        print("Vérifiez la structure de votre dataset sur le Hub ou définissez `class_names` manuellement.")
        # Pour l'exemple, supposons que les labels numériques correspondent à des noms par défaut si le dataset ne fournit pas les noms:
        # Si vous savez que 0=cat, 1=dog, 2=car, etc.
        # N.B. C'est une étape cruciale, si vos labels HF ne correspondent pas à l'ordre alphabétique de vos dossiers
        # Ou si le dataset est mal structuré, cela peut créer des erreurs.
        # Les noms de dossiers de sortie seront 'class_0', 'class_1', etc. par défaut de toute façon.
        class_names = [f"class_{i}" for i in range(dataset.features['label'].num_classes)] # Fallback
        
    print(f"Classes détectées: {class_names}")

except Exception as e:
    print(f"Erreur lors du chargement du dataset Hugging Face: {e}")
    print("Vérifiez le nom du dataset ou si le split 'train' existe.")
    exit()

# --- 2. Préparer les données (liste des chemins et labels) ---
# Extraire les chemins d'images temporaires (ou objets image) et les labels
# Note: Hugging Face Datasets charge les images en tant qu'objets PIL par défaut.
# Pour les sauvegarder, nous allons itérer et les sauver.
all_images = []
all_labels = []

# Collecter les images et labels
print("Collecte des images et labels...")
for item in tqdm(dataset):
    # 'image' et 'label' sont les noms de colonnes par défaut pour les datasets d'images sur HF.
    # Si votre dataset a des noms de colonnes différents (ex: 'img_data', 'category'), ajustez ici.
    all_images.append(item['image']) # L'objet PIL.Image lui-même
    all_labels.append(item['label'])

# Convertir en tableaux NumPy pour sklearn
all_images_np = np.array(all_images, dtype=object) # Utiliser dtype=object pour stocker des objets PIL
all_labels_np = np.array(all_labels)

# --- 3. Splitter les données en train/val/test ---
print("Partitionnement des données en ensembles d'entraînement, validation et test...")

# Premier split: train vs (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    all_images_np, all_labels_np,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=all_labels_np, # Important pour maintenir la proportion des classes
    random_state=42 # Pour la reproductibilité
)

# Deuxième split: val vs test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), # Calcul du ratio pour le deuxième split
    stratify=y_temp, # Maintenir la proportion des classes
    random_state=42
)

# Créer un dictionnaire pour faciliter le traitement
splits = {
    'train': {'images': X_train, 'labels': y_train},
    'val': {'images': X_val, 'labels': y_val},
    'test': {'images': X_test, 'labels': y_test},
}

print(f"Taille de l'ensemble d'entraînement: {len(X_train)}")
print(f"Taille de l'ensemble de validation: {len(X_val)}")
print(f"Taille de l'ensemble de test: {len(X_test)}")

# --- 4. Sauvegarder les images dans la nouvelle structure de dossiers ---
print(f"Sauvegarde des images dans la structure '{OUTPUT_ROOT_DIR}'...")

for split_name, data in splits.items():
    split_dir = os.path.join(OUTPUT_ROOT_DIR, split_name)
    
    for class_name in class_names:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    print(f"Sauvegarde du split '{split_name}'...")
    for i, (image_pil, label_idx) in enumerate(tqdm(zip(data['images'], data['labels']), total=len(data['images']))):
        class_folder = class_names[label_idx]
        
        # Le nom du fichier sera basé sur l'index, mais vous pouvez aussi utiliser un hash ou un UUID
        image_filename = f"{label_idx}_{i:06d}.jpg" # Ex: 0_000001.jpg, 1_000002.jpg
        image_path = os.path.join(split_dir, class_folder, image_filename)
        
        try:
            image_pil.save(image_path)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image {image_filename}: {e}")

print("\nPartitionnement et sauvegarde terminés !")
print(f"Votre dataset est maintenant disponible dans: {OUTPUT_ROOT_DIR}")
print("Vous pouvez maintenant mettre à jour les chemins dans vos scripts de PyTorch pour utiliser ce nouveau dataset.")