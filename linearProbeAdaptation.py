import os
import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Chargement des données (Train ET Test cette fois-ci)
root = os.path.expanduser("~/.cache")
train_dataset = CIFAR100(root=root, download=True, train=True, transform=preprocess)
test_dataset = CIFAR100(root=root, download=True, train=False, transform=preprocess)

# Fonction pour extraire les features de toutes les images
def get_features(dataset):
    all_features = []
    all_labels = []
    
    # On utilise un batch_size plus grand car pas de gradient à calculer (plus rapide)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    print(f"Extraction des features pour {len(dataset)} images...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True) # Normalisation importante !
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels)

# 3. Extraction (C'est l'étape lourde, le "Frozen Backbone")
print("--- Traitement du TRAIN set ---")
train_features, train_labels = get_features(train_dataset)

print("\n--- Traitement du TEST set ---")
test_features, test_labels = get_features(test_dataset)

# 4. Entraînement du Linear Probe (Logistic Regression)
# C'est exactement la méthode décrite dans l'Appendix A.3 du papier
print("\nEntraînement du classifieur linéaire (Linear Probe)...")
# C = paramètre de régularisation inverse (trouvé par grid search dans le papier)
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# 5. Évaluation
print("\nÉvaluation...")
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100

print(f"\nPrécision Linear Probe CLIP sur CIFAR-100 : {accuracy:.2f}%")
print(f"Rappel Zero-Shot (Ensembling) : ~63.72%")