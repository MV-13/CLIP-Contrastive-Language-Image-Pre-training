import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du device : {device}")

# Charging the CLIP model
# The paper recommends ViT-L/14@336px for the best performance[cite: 239], but ViT-B/32 is much lighter
model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device)
print(f"Model {model_name} loaded.")

# Preparation of Dataset (CIFAR-100)
dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess) # Using the 'preprocess' transformation provided by CLIP, not the default Torchvision one
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Preparation of Textual Prompts (Zero-Shot)
# CLIP needs sentences, not just words. The paper suggests "A photo of a {label}."
cifar100_classes = dataset.classes
text_prompts = [f"a photo of a {class_name}." for class_name in cifar100_classes]
text_inputs = clip.tokenize(text_prompts).to(device)

# Calculation of Textual Features
print("Encoding text descriptions...")
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Evaluation loop
correct_count = 0
total_count = 0

print("Starting Zero-Shot classification...")
model.eval()
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Calculation of similarity (Dot product)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Prediction (class with the highest probability)
        values, indices = similarity.topk(1)
        
        # Calculation of accuracy
        correct_count += indices.view(-1).eq(labels).sum().item()
        total_count += labels.size(0)

# Final Result
accuracy = 100 * correct_count / total_count
print(f"\nZero-Shot Accuracy on CIFAR-100 ({len(dataset)} images) : {accuracy:.2f}%")
print("Note: A random model would achieve 1%. A supervised ResNet-50 achieves ~75-80%.")