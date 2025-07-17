import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns 

# If TopKSAE is in 'overcomplete/sae/topk_sae.py' relative to your project root, this is correct.
# If it's directly in the same folder as this script, you might need: from topk_sae import TopKSAE
from overcomplete.sae.topk_sae import TopKSAE

# --- Configuration du script ---
# Chemin vers votre modèle entraîné complet (ResNet + SAE + Classifier)
MODEL_PATH = './models_pkl/best_resnet50_catsdogscars_SAE_FULL_MODEL.pth'
# Chemin vers le dossier de vos images de validation (ou de test/entraînement)
# Assurez-vous que ce dossier contient les sous-dossiers de classes (e.g., 'Cars', 'Cats', 'Dogs')
DATA_DIR_FOR_ANALYSIS = '/home/ameni/Documents/PASTA_code-main/datasets/catsdogscars_partitionned/val' # Match your training script's val path
# Nombre d'images à afficher pour chaque neurone du SAE
NUM_IMAGES_TO_SHOW_PER_NEURON = 8
# Dossier où les visualisations seront sauvegardées
OUTPUT_VIS_DIR = './sae_neuron_visualizations'

# Ces dimensions DOIVENT correspondre EXACTEMENT à celles utilisées lors de l'entraînement
RESNET_OUTPUT_DIM = 2048 # Output of ResNet backbone before feature_adapter
SAE_LATENT_DIM = 256     # <<< CONFIRMED from your training script
NUM_CLASSES = 3          # CONFIRMED from your training script
TOP_K = 64               # CONFIRMED from your training script's SAE instantiation

# --- Vérification du périphérique (GPU ou CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# --- Définition de l'Architecture du Modèle (Copie Conforme de train_sae_resnet50.py) ---
# Ceci doit être un miroir exact de la classe SAE_ResNet50 dans votre script d'entraînement.
class SAE_ResNet50(nn.Module):
    def __init__(self, num_classes, latent_dim=256, top_k=64):
        super(SAE_ResNet50, self).__init__()
        
        # 1. Backbone ResNet50
        self.resnet = models.resnet50(weights=None) # No ImageNet pretrained weights needed, we'll load custom ones
        
        # Remove the final classification layer to use as feature extractor
        self.resnet_features = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Feature adapter layer
        self.feature_adapter = nn.Linear(2048, latent_dim)

        # Sparse Autoencoder (SAE)
        self.sae = TopKSAE(
            input_shape=latent_dim, 
            nb_concepts=latent_dim,
            top_k=top_k,
            encoder_module="mlp_ln_1",
            device=device # Use 'device' here, not 'DEVICE' from global scope in init
        ) # .to(device) will be called on the whole model later
        
        # Classification head after SAE
        self.classifier = nn.Linear(latent_dim, num_classes) 

    def forward(self, x):
        # Pass through the (frozen) ResNet backbone
        with torch.no_grad(): # Ensure no gradients are computed for the backbone
            features = self.resnet_features(x)  # (Batch_size, 2048, 1, 1)
            features = torch.flatten(features, 1) # Flatten to (Batch_size, 2048)

        # Pass through the feature adapter (this layer is trainable)
        adapted_features = self.feature_adapter(features) # (Batch_size, latent_dim)

        # Pass through the SAE. TopKSAE's forward returns:
        # (sparse_latent_features, raw_latent_codes, reconstructed_input)
        sparse_latent_features, _, reconstructed_sae_input = self.sae(adapted_features)
        
        # Pass sparse latent features to the classification head
        output_cls = self.classifier(sparse_latent_features)
        
        # Return outputs necessary for original training / state_dict compatibility
        return output_cls, reconstructed_sae_input, sparse_latent_features # Retourne aussi sparse_latent_features pour l'analyse

# Instantiate the combined model
model = SAE_ResNet50(NUM_CLASSES, latent_dim=SAE_LATENT_DIM, top_k=TOP_K).to(device)

# --- Chargement du modèle entraîné complet ---
try:
    state_dict_loaded = torch.load(MODEL_PATH, map_location=device)
    
    # Check if the state_dict is nested (e.g., {'model_state_dict': ...})
    if 'model_state_dict' in state_dict_loaded:
        model.load_state_dict(state_dict_loaded['model_state_dict'])
    else:
        # If the state_dict contains the model directly (as saved by torch.save(model.state_dict(), ...))
        # The key names should perfectly match 'SAE_ResNet50' module names.
        model.load_state_dict(state_dict_loaded)
        
    print(f"Modèle chargé avec succès depuis: {MODEL_PATH}")
    model.eval() # Set the model to evaluation mode
    
    # After loading, explicitly freeze the resnet backbone again (redundant but safe)
    for param in model.resnet.parameters():
        param.requires_grad = False
    print("ResNet backbone weights are frozen for inference.")

except Exception as e:
    print(f"ERREUR lors du chargement du modèle: {e}")
    print("Vérifiez le chemin du modèle, et assurez-vous que la définition de la classe SAE_ResNet50")
    print("dans ce script correspond EXACTEMENT à celle utilisée lors de l'entraînement.")
    exit()

# --- Préparation du jeu de données pour l'analyse ---
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Standard size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    dataset = datasets.ImageFolder(root=DATA_DIR_FOR_ANALYSIS, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
except Exception as e:
    print(f"ERREUR lors du chargement du dataset: {e}")
    print(f"Vérifiez que le chemin '{DATA_DIR_FOR_ANALYSIS}' existe et contient des sous-dossiers de classes (ex: 'Cars', 'Cats').")
    exit()

print(f"Chargement de {len(dataset)} images depuis {DATA_DIR_FOR_ANALYSIS} pour l'analyse des activations...")

# --- NEW: Retrieve actual class names from the dataset ---
class_names = dataset.classes
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
print(f"Classes détectées: {class_names}") # Print raw list
print(f"Mapping Index -> Class Name: {idx_to_class}") # Print dictionary for verification

# --- Collecte des activations du SAE ---
neuron_activations_data = {i: [] for i in range(SAE_LATENT_DIM)}
activations_by_class = {class_name: [] for class_name in class_names} # Initialize for all detected classes

all_latent_features = []
all_labels = []
all_top_activated_neurons = []

print("Collecte des activations du SAE et des features latentes...")
num_processed_images = 0 # Counter for processed images

with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        # Limit the number of images if the dataset is huge, for faster testing of PCA/t-SNE
        # For example, process only the first 500 images
        # if i >= 500:
        #     print(f"Processed {i} images. Stopping collection early for testing purposes.")
        #     break

        inputs = inputs.to(device)
        label_idx = labels.item()
        
        if label_idx not in idx_to_class:
            print(f"AVERTISSEMENT: Label {label_idx} trouvé dans le dataset mais non présent dans idx_to_class. Saut de l'image {i}.")
            continue # Skip this image if its label is unexpected

        current_class_name = idx_to_class[label_idx]
        
        try:
            _, _, sparse_latent_features_tensor = model(inputs)
        except Exception as e:
            print(f"ERREUR lors du passage forward du modèle pour l'image {i}: {e}. Saut de cette image.")
            continue

        current_activations = sparse_latent_features_tensor[0].cpu().numpy() # (latent_dim,)
        
        # Collect data for neuron visualizations (top images for each neuron)
        original_image_path = dataset.samples[i][0]
        original_image_pil = Image.open(original_image_path).convert('RGB')

        for neuron_idx in range(SAE_LATENT_DIM):
            activation_value = current_activations[neuron_idx]
            neuron_activations_data[neuron_idx].append((activation_value, original_image_pil, original_image_path))
        
        # Collect data for PCA/t-SNE and heatmap
        all_latent_features.append(current_activations)
        all_labels.append(label_idx)
        activations_by_class[current_class_name].append(current_activations)

        top_neuron_idx = np.argmax(current_activations)
        all_top_activated_neurons.append(top_neuron_idx)
        num_processed_images += 1

print(f"Collecte des activations terminée. Nombre d'images traitées: {num_processed_images}")

if num_processed_images == 0:
    print("\nERREUR FATALE: Aucune image n'a été traitée. Vérifiez le chemin du dataset, le dataloader, et les éventuelles erreurs durant la boucle de collecte.")
    exit()

print("Démarrage de la génération des visualisations...\n")

# --- Génération et Sauvegarde des Visualisations par Neurone (images les plus activantes) ---
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

for neuron_idx in range(SAE_LATENT_DIM):
    sorted_images_for_neuron = sorted(neuron_activations_data[neuron_idx], key=lambda x: x[0], reverse=True)
    top_n_images = sorted_images_for_neuron[:NUM_IMAGES_TO_SHOW_PER_NEURON]

    if not top_n_images:
        # print(f"Warning: No activations found for neuron {neuron_idx}. Skipping visualization.")
        continue

    fig, axes = plt.subplots(1, NUM_IMAGES_TO_SHOW_PER_NEURON, figsize=(NUM_IMAGES_TO_SHOW_PER_NEURON * 3, 4))
    if NUM_IMAGES_TO_SHOW_PER_NEURON == 1:
        axes = [axes]

    fig.suptitle(f"Neurone SAE {neuron_idx}: Images les plus activantes", fontsize=16)

    for j, (activation_value, img_pil, img_path) in enumerate(top_n_images):
        ax = axes[j]
        ax.imshow(img_pil)
        class_name_from_path = os.path.basename(os.path.dirname(img_path))
        ax.set_title(f"Act: {activation_value:.4f}\nClass: {class_name_from_path}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, f'sae_neuron_{neuron_idx}_top_activations.png'))
    plt.close(fig)

print(f"\nToutes les visualisations des neurones sont sauvegardées dans le dossier: {OUTPUT_VIS_DIR}")
print("Ouvrez les fichiers PNG dans ce dossier et analysez les motifs récurrents pour identifier les concepts de chaque neurone.")
print("Ce processus d'analyse visuelle est essentiel pour l'interprétabilité de votre SAE.")

# Convertir les listes en tableaux NumPy
latent_features_matrix = np.array(all_latent_features)
numerical_labels_array = np.array(all_labels)
top_activated_neurons_array = np.array(all_top_activated_neurons)

# --- NEW: Check if latent_features_matrix is empty or malformed ---
if latent_features_matrix.size == 0 or latent_features_matrix.shape[0] < 2:
    print("\nERREUR: La matrice des features latentes est vide ou contient moins de 2 échantillons. Impossible d'effectuer PCA/t-SNE.")
    print("Vérifiez le chemin de votre dataset et la collecte des activations.")
else:
    print(f"\nMatrice de features latentes pour PCA/t-SNE de forme: {latent_features_matrix.shape}")
    print(f"Exemple de labels numériques: {numerical_labels_array[:5]}")
    print(f"Exemple de neurones top activés: {top_activated_neurons_array[:5]}")

    # --- This is the line that was causing issues. Ensure it has valid inputs ---
    try:
        class_names_for_plotting = np.array([idx_to_class[label] for label in numerical_labels_array])
        print(f"Exemple de noms de classes pour le plotting: {class_names_for_plotting[:5]}")
    except KeyError as e:
        print(f"\nERREUR: 'KeyError' lors de la création de class_names_for_plotting. Le label {e} n'existe pas dans idx_to_class.")
        print("Vérifiez le contenu de numerical_labels_array et de idx_to_class.")
        exit() # Exit if this critical step fails

    # --- NEW: Préparer les données pour la heatmap des activations moyennes par classe ---
    mean_activations_per_class = {}
    for class_name in class_names:
        class_activations_list = [act for act, lbl in zip(all_latent_features, all_labels) if idx_to_class[lbl] == class_name]
        class_activations = np.array(class_activations_list)
        if class_activations.size > 0:
            mean_activations_per_class[class_name] = np.mean(class_activations, axis=0)
        else:
            mean_activations_per_class[class_name] = np.zeros(SAE_LATENT_DIM)

    heatmap_data = pd.DataFrame(mean_activations_per_class).T
    heatmap_data.columns = [f'Neurone {i}' for i in range(SAE_LATENT_DIM)]

    # --- NEW: Heatmap des Activations Moyennes des Neurones par Classe ---
    plt.figure(figsize=(16, 6))
    sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f", linewidths=.5, linecolor='lightgray',
                cbar_kws={'label': 'Activation Moyenne'})
    plt.title('Activation Moyenne des Neurones SAE par Classe', fontsize=18)
    plt.xlabel('Neurones SAE', fontsize=14)
    plt.ylabel('Classes', fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_mean_activations_heatmap_by_class.png'))
    plt.close()
    print("Heatmap des activations moyennes par classe sauvegardée.")

    # --- NEW: Distribution des activations pour un échantillon de neurones ---
    plt.figure(figsize=(15, 8))
    sample_neurons = [0, 10, 50, 100, 150, 200, 250] # Example neurons
    for n_idx in sample_neurons:
        if n_idx < SAE_LATENT_DIM and neuron_activations_data[n_idx]:
            activations = [x[0] for x in neuron_activations_data[n_idx]]
            sns.histplot(activations, kde=True, bins=50, label=f'Neurone {n_idx}', alpha=0.6)
    plt.title('Distribution des Activations des Neurones SAE (Échantillon)', fontsize=16)
    plt.xlabel('Valeur d\'Activation', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_neuron_activation_distributions.png'))
    plt.close()
    print("Histogrammes des distributions d'activation des neurones sauvegardés.")


    # --- PCA ---
    print("Application de PCA...")
    pca = PCA(n_components=2)
    latent_features_pca = pca.fit_transform(latent_features_matrix)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_features_pca[:, 0], y=latent_features_pca[:, 1],
        hue=class_names_for_plotting, palette='viridis',
        s=50,
        alpha=0.7
    )
    plt.title('PCA des features latentes du SAE (couleur par classe)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_latent_pca_by_class.png'))
    plt.close()
    print("PCA (par classe) sauvegardée.")

    # PCA colorée par neurone le plus activé
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_features_pca[:, 0], y=latent_features_pca[:, 1],
        hue=top_activated_neurons_array, palette='tab20',
        s=50,
        alpha=0.7
    )
    plt.title('PCA des features latentes du SAE (couleur par neurone le plus activé)', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Neurone Top', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_latent_pca_by_top_neuron.png'))
    plt.close()
    print("PCA (par neurone top) sauvegardée.")


    # --- t-SNE ---
    print("Application de t-SNE (cela peut prendre du temps)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=1000)
    latent_features_tsne = tsne.fit_transform(latent_features_matrix)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_features_tsne[:, 0], y=latent_features_tsne[:, 1],
        hue=class_names_for_plotting, palette='viridis',
        s=50,
        alpha=0.7
    )
    plt.title('t-SNE des features latentes du SAE (couleur par classe)', fontsize=16)
    plt.xlabel('Composante t-SNE 1', fontsize=12)
    plt.ylabel('Composante t-SNE 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Classe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_latent_tsne_by_class.png'))
    plt.close()
    print("t-SNE (par classe) sauvegardée.")

    # t-SNE coloré par neurone le plus activé
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=latent_features_tsne[:, 0], y=latent_features_tsne[:, 1],
        hue=top_activated_neurons_array, palette='tab20',
        s=50,
        alpha=0.7
    )
    plt.title('t-SNE des features latentes du SAE (couleur par neurone le plus activé)', fontsize=16)
    plt.xlabel('Composante t-SNE 1', fontsize=12)
    plt.ylabel('Composante t-SNE 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Neurone Top', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIS_DIR, 'sae_latent_tsne_by_top_neuron.png'))
    plt.close()
    print("t-SNE (par neurone top) sauvegardée.")

print("\nAnalyse de l'espace latent du SAE terminée.")
print(f"Vérifiez les visualisations dans le dossier: {OUTPUT_VIS_DIR}")