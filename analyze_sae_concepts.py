import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE # Make sure these are installed: pip install scikit-learn seaborn matplotlib pandas

# Import the TopKSAE class definition
from overcomplete.sae.topk_sae import TopKSAE # Adjust path if necessary

# --- Configuration (DOIT CORRESPONDRE AU SCRIPT PRÉCÉDENT) ---
MODEL_PATH = './models_pkl/best_resnet50_catsdogscars_SAE_FULL_MODEL.pth'
DATA_DIR_FOR_ANALYSIS = '/home/ameni/Documents/PASTA_code-main/datasets/catsdogscars_partitionned/val'
OUTPUT_VIS_DIR = './sae_neuron_visualizations' # Where existing visualizations are, and new reports will go
NUM_IMAGES_TO_SHOW_PER_NEURON = 8
SAE_LATENT_DIM = 256
NUM_CLASSES = 3
TOP_K = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Définition de l'Architecture du Modèle (Copie Conforme de train_sae_resnet50.py) ---
# Ceci doit être un miroir exact de la classe SAE_ResNet50 dans votre script d'entraînement.
class SAE_ResNet50(nn.Module):
    def __init__(self, num_classes, latent_dim=256, top_k=64):
        super(SAE_ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.feature_adapter = nn.Linear(2048, latent_dim)
        self.sae = TopKSAE(
            input_shape=latent_dim, 
            nb_concepts=latent_dim,
            top_k=top_k,
            encoder_module="mlp_ln_1",
            device=device
        )
        self.classifier = nn.Linear(latent_dim, num_classes) 

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet_features(x)
            features = torch.flatten(features, 1)
        adapted_features = self.feature_adapter(features)
        sparse_latent_features, _, reconstructed_sae_input = self.sae(adapted_features)
        output_cls = self.classifier(sparse_latent_features)
        return output_cls, reconstructed_sae_input, sparse_latent_features # Return sparse_latent_features

# Instantiate and load model
model = SAE_ResNet50(NUM_CLASSES, latent_dim=SAE_LATENT_DIM, top_k=TOP_K).to(device)
try:
    state_dict_loaded = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in state_dict_loaded:
        model.load_state_dict(state_dict_loaded['model_state_dict'])
    else:
        model.load_state_dict(state_dict_loaded)
    print(f"Modèle chargé avec succès depuis: {MODEL_PATH}")
    model.eval()
    for param in model.resnet.parameters():
        param.requires_grad = False
except Exception as e:
    print(f"ERREUR lors du chargement du modèle: {e}")
    exit()

# --- Préparation du jeu de données pour l'analyse ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
try:
    dataset = datasets.ImageFolder(root=DATA_DIR_FOR_ANALYSIS, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
except Exception as e:
    print(f"ERREUR lors du chargement du dataset: {e}")
    exit()

class_names = dataset.classes
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
print(f"Classes détectées: {class_names}")

# --- Collecte des activations du SAE (ré-exécutée pour ce script) ---
neuron_activations_data = {i: [] for i in range(SAE_LATENT_DIM)}
activations_by_class = {class_name: [] for class_name in class_names}
all_latent_features = []
all_labels = []

print("Re-collecte des activations du SAE pour l'analyse des concepts...")
num_processed_images = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        # if i >= 100: # Optional: Limit for faster testing if your dataset is huge
        #     break
        
        inputs = inputs.to(device)
        label_idx = labels.item()
        current_class_name = idx_to_class[label_idx]
        
        _, _, sparse_latent_features_tensor = model(inputs)
        current_activations = sparse_latent_features_tensor[0].cpu().numpy()

        original_image_path = dataset.samples[i][0]
        original_image_pil = Image.open(original_image_path).convert('RGB')

        for neuron_idx in range(SAE_LATENT_DIM):
            activation_value = current_activations[neuron_idx]
            neuron_activations_data[neuron_idx].append((activation_value, original_image_pil, original_image_path, current_class_name)) # Store class name too
        
        all_latent_features.append(current_activations)
        all_labels.append(label_idx)
        activations_by_class[current_class_name].append(current_activations)
        num_processed_images += 1

if num_processed_images == 0:
    print("\nERREUR: Aucune image traitée. Vérifiez le dataset.")
    exit()

# Convertir les listes en tableaux NumPy
latent_features_matrix = np.array(all_latent_features)
numerical_labels_array = np.array(all_labels)

# --- Préparer les données pour la heatmap des activations moyennes par classe ---
mean_activations_per_class = {}
for class_name in class_names:
    class_activations_list = [act for act, lbl in zip(all_latent_features, all_labels) if idx_to_class[lbl] == class_name]
    class_activations = np.array(class_activations_list)
    if class_activations.size > 0:
        mean_activations_per_class[class_name] = np.mean(class_activations, axis=0)
    else:
        mean_activations_per_class[class_name] = np.zeros(SAE_LATENT_DIM)

heatmap_df = pd.DataFrame(mean_activations_per_class).T
heatmap_df.columns = [f'Neurone {i}' for i in range(SAE_LATENT_DIM)]

# --- Analyse de la Heatmap pour l'identification des neurones clés ---
print("\nAnalyse de la heatmap pour identifier les neurones intéressants...")

# Seuil d'activation pour considérer un neurone "actif"
ACTIVATION_THRESHOLD = 0.5 # Ajustez ce seuil si nécessaire (ex: 0.1, 0.5, 1.0)

# Pour chaque neurone, identifier les classes où il est fortement activé
neuron_class_activation_map = {} # {neuron_idx: [list_of_highly_activated_classes]}
for col_idx, neuron_col_name in enumerate(heatmap_df.columns):
    activated_classes = []
    for row_idx, class_name in enumerate(heatmap_df.index):
        if heatmap_df.loc[class_name, neuron_col_name] > ACTIVATION_THRESHOLD:
            activated_classes.append(class_name)
    neuron_class_activation_map[col_idx] = activated_classes

# Catégoriser les neurones
specialized_neurons = [] # Actifs pour 1 seule classe
shared_neurons = []      # Actifs pour 2+ classes
inactive_neurons = []    # Non actifs pour aucune classe

for neuron_idx, classes in neuron_class_activation_map.items():
    if len(classes) == 1:
        specialized_neurons.append(neuron_idx)
    elif len(classes) >= 2:
        shared_neurons.append(neuron_idx)
    else:
        inactive_neurons.append(neuron_idx)

print(f"\nIdentification des neurones:")
print(f"- Neurones spécialisés (actifs pour 1 classe): {len(specialized_neurons)} neurones")
print(f"- Neurones partagés (actifs pour 2+ classes): {len(shared_neurons)} neurones")
print(f"- Neurones inactifs (sous le seuil de {ACTIVATION_THRESHOLD}): {len(inactive_neurons)} neurones")

# Optionnel: Trier les neurones spécialisés/partagés par leur activation maximale pour un meilleur ordre d'examen
specialized_neurons_sorted = sorted(specialized_neurons, key=lambda n_idx: np.max(heatmap_df.iloc[:, n_idx]), reverse=True)
shared_neurons_sorted = sorted(shared_neurons, key=lambda n_idx: np.max(heatmap_df.iloc[:, n_idx]), reverse=True)

# --- Processus d'annotation interactive ---
print("\n--- Début du processus d'annotation des concepts ---")
print("Pour chaque neurone, vous verrez ses images les plus activantes.")
print("Entrez un concept qui décrit ce que le neurone semble détecter.")
print("Si vous ne voyez pas de concept clair, vous pouvez laisser vide ou entrer '??'.")
print("Entrez 'q' pour quitter à tout moment.")

# Fichier pour sauvegarder les annotations
annotations_file = os.path.join(OUTPUT_VIS_DIR, 'sae_neuron_concepts_annotations.txt')
if os.path.exists(annotations_file):
    print(f"Attention: Le fichier d'annotations '{annotations_file}' existe déjà et sera mis à jour.")
    # Load existing annotations to avoid re-asking for already annotated neurons
    existing_annotations = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':', 1)
                neuron_id = int(parts[0].replace('Neurone ', ''))
                concept = parts[1].strip()
                existing_annotations[neuron_id] = concept
else:
    existing_annotations = {}

# Liste des neurones à examiner
neurons_to_examine = specialized_neurons_sorted + shared_neurons_sorted

# Supprimer les doublons et les neurones déjà annotés
neurons_to_examine = sorted(list(set(neurons_to_examine) - set(existing_annotations.keys())))

print(f"\n{len(neurons_to_examine)} neurones à examiner. ({len(existing_annotations)} déjà annotés)")

# Loop through selected neurons for interactive annotation
for i, neuron_idx in enumerate(neurons_to_examine):
    print(f"\n--- Neurone {neuron_idx} ({i+1}/{len(neurons_to_examine)}) ---")
    
    # Afficher les classes où ce neurone est fortement activé
    classes_info = ", ".join(neuron_class_activation_map.get(neuron_idx, ["Inactif"]))
    print(f"Classes activantes: {classes_info}")

    # Afficher la heatmap pour ce neurone (facultatif, juste pour le contexte)
    # plt.figure(figsize=(3, len(class_names) * 0.8))
    # sns.heatmap(heatmap_df[[f'Neurone {neuron_idx}']], cmap='viridis', annot=True, fmt=".2f", cbar=False, yticklabels=True)
    # plt.title(f'Act. Moy. Neurone {neuron_idx}')
    # plt.tight_layout()
    # plt.show() # Requires interactive matplotlib backend

    # Display top activating images
    top_n_images = sorted(neuron_activations_data[neuron_idx], key=lambda x: x[0], reverse=True)[:NUM_IMAGES_TO_SHOW_PER_NEURON]
    
    if not top_n_images:
        print(f"Aucune image activante pour le Neurone {neuron_idx}. Skip.")
        continue

    fig, axes = plt.subplots(1, NUM_IMAGES_TO_SHOW_PER_NEURON, figsize=(NUM_IMAGES_TO_SHOW_PER_NEURON * 2.5, 3.5))
    if NUM_IMAGES_TO_SHOW_PER_NEURON == 1:
        axes = [axes]

    fig.suptitle(f"Neurone SAE {neuron_idx}: Images les plus activantes", fontsize=14)

    for j, (activation_value, img_pil, img_path, class_name) in enumerate(top_n_images):
        ax = axes[j]
        ax.imshow(img_pil)
        ax.set_title(f"Act: {activation_value:.2f}\nClass: {class_name}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show(block=False) # Display the plot non-blocking
    plt.pause(0.1) # Give time for the plot to render

    # Get user input for concept
    concept = input("Concept (e.g., 'Roue de voiture', 'Fourrure animale', '??'): ").strip()
    
    # Close the figure
    plt.close(fig)

    if concept.lower() == 'q':
        print("Arrêt de l'annotation.")
        break
    
    # Save annotation
    with open(annotations_file, 'a') as f:
        f.write(f"Neurone {neuron_idx}: {concept}\n")
    existing_annotations[neuron_idx] = concept # Update in memory

print("\n--- Processus d'annotation terminé. ---")
print(f"Concepts sauvegardés dans: {annotations_file}")

# --- Optional: Summarize current annotations ---
print("\n--- Récapitulatif des annotations ---")
for neuron_id in sorted(existing_annotations.keys()):
    print(f"Neurone {neuron_id}: {existing_annotations[neuron_id]}")