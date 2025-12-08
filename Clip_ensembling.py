import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loading the model
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset CIFAR-100
dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 4. IMPROVEMENT: Prompt Ensembling : instead of a single sentence, we use multiple to "guide" the model from different angles
templates = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}."
]

print("Creating text embeddings with Ensembling...")
cifar100_classes = dataset.classes

# We will store the average of the features for each class
zeroshot_weights = []

with torch.no_grad():
    for classname in tqdm(cifar100_classes):
        texts = [template.format(classname) for template in templates] 
        
        texts_tokenized = clip.tokenize(texts).to(device) 
        class_embeddings = model.encode_text(texts_tokenized) 
        
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

# Evaluation loop
correct_count = 0
total_count = 0

print("Starting classification...")
model.eval()
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculation of similarity with the "ensembled" weights
        similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
        values, indices = similarity.topk(1)
        
        correct_count += indices.view(-1).eq(labels).sum().item()
        total_count += labels.size(0)

accuracy = 100 * correct_count / total_count
print(f"\nAccuracy with Prompt Ensembling: {accuracy:.2f}%")