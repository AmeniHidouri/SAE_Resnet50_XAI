import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm # Pour une barre de progression visuelle

# Assurez-vous que le chemin vers votre module overcomplete est accessible dans PYTHONPATH
# ou que le fichier topk_sae.py est dans le même dossier ou un sous-dossier correctement importé.
from overcomplete.sae.topk_sae import TopKSAE

# --- Hyperparameters ---
NUM_CLASSES = 3
LATENT_DIM = 256
BATCH_SIZE = 8
NUM_EPOCHS = 50 # <-- Réglé à 50 époques pour une meilleure convergence
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")

if DEVICE.type == 'cuda':
    torch.cuda.empty_cache() # Vide le cache de la mémoire GPU
    print("CUDA cache cleared.")

# --- Data Transformations ---
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_DATA_PATH = "/home/ameni/Documents/PASTA_code-main/datasets/catsdogscars_partitionned/train"
VAL_DATA_PATH = "/home/ameni/Documents/PASTA_code-main/datasets/catsdogscars_partitionned/val"

train_dataset = datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DATA_PATH, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Classes détectées par ImageFolder (train): {train_dataset.classes}")
print(f"Mapping des classes (train): {train_dataset.class_to_idx}")
print(f"Nombre de classes détectées (train): {len(train_dataset.classes)}")

if len(train_dataset.classes) != NUM_CLASSES:
    print(f"\nATTENTION: NUM_CLASSES ({NUM_CLASSES}) ne correspond pas "
          f"au nombre de classes réelles détectées par ImageFolder ({len(train_dataset.classes)}).\n"
          "Cela peut causer l'erreur 'Assertion t >= 0 && t < n_classes' failed. "
          "Vérifiez la structure de vos dossiers de dataset.")

# --- Modèle Combiné: SAE_ResNet50 ---
model_path = "/home/ameni/Documents/PASTA_code-main/models_pkl/best_resnet50_catsdogscars.pth" # Ton fichier de poids ResNet

class SAE_ResNet50(nn.Module):
    def __init__(self, num_classes, latent_dim=256, top_k=64):
        super(SAE_ResNet50, self).__init__()
        
        # 1. Backbone ResNet50 initialisé SANS poids pré-entraînés ImageNet.
        # Les poids de ton fichier catsdogscars.pth seront chargés séparément.
        self.resnet = models.resnet50(weights=None) 
        
        # Le gel des paramètres se fera après le chargement des poids spécifiques à ton ResNet.
            
        # Supprimer la couche de classification finale du ResNet pour l'utiliser comme extracteur de features
        # Il est important de ne pas inclure la couche 'fc' (final classification) du ResNet
        self.resnet_features = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Couche d'adaptation des features du ResNet (2048) à la dimension latente (256) du SAE
        # Cette couche est entraînable.
        self.feature_adapter = nn.Linear(2048, latent_dim)

        # Sparse Autoencoder (SAE)
        # L'input_shape du SAE est maintenant `latent_dim` car il prend la sortie de `feature_adapter`
        self.sae = TopKSAE(
            input_shape=latent_dim, 
            nb_concepts=latent_dim,
            top_k=top_k,
            encoder_module="mlp_ln_1",
            device=DEVICE
        ).to(DEVICE)
        
        # Tête de classification après le SAE
        self.classifier = nn.Linear(latent_dim, num_classes) 

    def forward(self, x):
        # Passer par le backbone ResNet gelé (pas de calcul de gradient ici)
        with torch.no_grad():
            features = self.resnet_features(x)  # (Batch_size, 2048, 1, 1)
            features = torch.flatten(features, 1) # Aplatir à (Batch_size, 2048)

        # Passer par la couche d'adaptation (cette couche est entraînable)
        adapted_features = self.feature_adapter(features) # (Batch_size, latent_dim)

        # Passer par le SAE. Le forward du TopKSAE retourne :
        # (features_latentes_parsimonieuses, codes_latents_bruts, features_reconstruites)
        sparse_latent_features, _, reconstructed_sae_input = self.sae(adapted_features)
        
        # Passer les features latentes parcimonieuses à la tête de classification
        output_cls = self.classifier(sparse_latent_features)
        
        # Retourner l'output de classification, la reconstruction du SAE, 
        # et les features originales (avant SAE) pour le calcul de la perte de reconstruction
        return output_cls, reconstructed_sae_input, adapted_features

# Instancier le modèle combiné
model = SAE_ResNet50(NUM_CLASSES, latent_dim=LATENT_DIM, top_k=64).to(DEVICE)

# --- Chargement et Gel des Poids du Backbone ResNet (ton fichier .pth) ---
if os.path.exists(model_path):
    print(f"Loading ResNet50 backbone weights from {model_path}...")
    try:
        # Charger tous les poids sauvegardés depuis ton fichier .pth
        state_dict_loaded = torch.load(model_path, map_location='cpu')
        
        # Créer un state_dict pour le sous-module `model.resnet`
        resnet_state_dict_to_load = {}
        for k, v in state_dict_loaded.items():
            # Tenter de charger les clés avec ou sans les préfixes 'resnet.' ou 'resnet50.'
            # Les clés dans ton .pth sont préfixées par 'resnet.' comme vu dans 'Unexpected keys'
            if k.startswith('resnet.'): 
                # Retire le préfixe 'resnet.' pour qu'il corresponde au `resnet` pur
                resnet_state_dict_to_load[k[len('resnet.'):]] = v
            # Gérer les cas où le .pth contenait un modèle complet 'SAE_ResNet50'
            # et nous voulons extraire SEULEMENT les couches du backbone ResNet
            elif not (k.startswith('sae.') or k.startswith('classifier.') or k.startswith('feature_adapter.')):
                # Si la clé ne commence par aucun des préfixes des autres modules
                # et ne se termine pas par '.num_batches_tracked' (qui est géré automatiquement)
                if not k.endswith('.num_batches_tracked'):
                    resnet_state_dict_to_load[k] = v
        
        # Charger ces poids filtrés dans le sous-module `model.resnet`.
        # `strict=False` est crucial ici pour ignorer les clés du fichier .pth qui ne sont pas dans `model.resnet`
        # (comme l'ancienne couche FC, ou d'autres parties si ton .pth contenait un modèle plus complet).
        missing_keys_resnet, unexpected_keys_resnet = model.resnet.load_state_dict(resnet_state_dict_to_load, strict=False)
        print("Missing keys in ResNet backbone (after loading your .pth):", missing_keys_resnet)
        print("Unexpected keys in ResNet backbone (after loading your .pth):", unexpected_keys_resnet)
        
        # --- GEL DES POIDS DU BACKBONE APRÈS LE CHARGEMENT ---
        # Cette étape est vitale pour ne pas modifier les poids de ton ResNet pré-entraîné.
        for param in model.resnet.parameters():
            param.requires_grad = False
        print("ResNet backbone weights loaded and frozen.")

    except Exception as e:
        print(f"Error loading ResNet backbone weights: {e}")
        print("ResNet backbone will be randomly initialized (not recommended) and frozen.")
        # Si le chargement échoue, geler quand même pour éviter l'entraînement accidentel du backbone non initialisé.
        for param in model.resnet.parameters():
            param.requires_grad = False
else:
    print(f"No saved ResNet50 backbone found at {model_path}. Starting with randomly initialized ResNet (not recommended) and frozen.")
    # Si le fichier .pth n'existe pas, geler le ResNet non entraîné.
    for param in model.resnet.parameters():
        param.requires_grad = False

# --- Loss & Optimizer ---
criterion_classification = nn.CrossEntropyLoss()
criterion_reconstruction = nn.MSELoss() # Perte MSE pour la reconstruction du SAE

# Optimiser UNIQUEMENT les paramètres qui sont entraînés :
# la couche d'adaptation des features, le SAE, et la tête de classification.
optimizer = optim.Adam(
    list(model.feature_adapter.parameters()) + 
    list(model.sae.parameters()) +
    list(model.classifier.parameters()),
    lr=LEARNING_RATE
)

# --- Train function ---
def train():
    best_val_accuracy = 0.0 # Pour suivre et sauvegarder le meilleur modèle
    # Le chemin pour sauvegarder le modèle complet (SAE + Classifier)
    full_model_save_path = model_path.replace(".pth", "_SAE_FULL_MODEL.pth") 
    
    for epoch in range(NUM_EPOCHS):
        model.train() # Met les modules entraînés (feature_adapter, SAE, classifier) en mode entraînement
        running_loss = 0.0
        running_cls_loss = 0.0
        running_rec_loss = 0.0
        correct, total = 0, 0

        # Utilisez tqdm pour une barre de progression visuelle pendant l'entraînement
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            # Le forward du modèle retourne (output_classification, reconstruction_du_sae, input_original_du_sae)
            classification_output, reconstructed_sae_input, original_sae_input = model(inputs)

            loss_cls = criterion_classification(classification_output, labels)
            sae_reconstruction_weight = 0.1 # Poids de la perte de reconstruction
            loss_reconstruction = criterion_reconstruction(reconstructed_sae_input, original_sae_input)

            loss = loss_cls + sae_reconstruction_weight * loss_reconstruction

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cls_loss += loss_cls.item()
            running_rec_loss += loss_reconstruction.item()

            _, preds = torch.max(classification_output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")
        
        # Validation après chaque époque
        val_accuracy = validate()

        # Sauvegarde le meilleur modèle complet basé sur la précision de validation
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Sauvegarder l'état complet du modèle (SAE, Adapter, Classifier)
            os.makedirs(os.path.dirname(full_model_save_path), exist_ok=True)
            torch.save(model.state_dict(), full_model_save_path)
            print(f"Saved best full model with Validation Accuracy: {best_val_accuracy:.4f} to {full_model_save_path}")
            
    print(f"Training complete. Best Validation Accuracy: {best_val_accuracy:.4f}")

# --- Validation function ---
def validate():
    model.eval() # Met tous les modules en mode évaluation
    correct, total = 0, 0
    val_loss = 0.0
    val_cls_loss = 0.0
    val_rec_loss = 0.0

    with torch.no_grad(): # Désactive le calcul de gradients en validation
        # Utilisez tqdm pour une barre de progression visuelle pendant la validation
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            classification_output, reconstructed_sae_input, original_sae_input = model(inputs)

            loss_cls = criterion_classification(classification_output, labels)
            sae_reconstruction_weight = 0.1
            loss_reconstruction = criterion_reconstruction(reconstructed_sae_input, original_sae_input)
            loss = loss_cls + sae_reconstruction_weight * loss_reconstruction

            val_loss += loss.item()
            val_cls_loss += loss_cls.item()
            val_rec_loss += loss_reconstruction.item()

            _, preds = torch.max(classification_output, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_val_loss = val_loss / len(val_loader)
    avg_val_cls_loss = val_cls_loss / len(val_loader)
    avg_val_rec_loss = val_rec_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f} (CLS: {avg_val_cls_loss:.4f}, REC: {avg_val_rec_loss:.4f}) | Validation Accuracy: {acc:.4f}")
    return acc # Retourne la précision de validation

# --- Run ---
if __name__ == "__main__":
    train()