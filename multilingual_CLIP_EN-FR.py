import os
import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================================
# Visualization of Multilingual CLIP Performance on CIFAR-100
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) # or "ViT-L/14@336px"

dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Partial translation dictionary for CIFAR-100 classes (English to French)
translation_dict = {
    'apple': 'pomme', 'aquarium_fish': 'poisson d\'aquarium', 'baby': 'bébé',
    'bear': 'ours', 'beaver': 'castor', 'bed': 'lit', 'bee': 'abeille',
    'beetle': 'scarabée', 'bicycle': 'vélo', 'bottle': 'bouteille',
    'bowl': 'bol', 'boy': 'garçon', 'bridge': 'pont', 'bus': 'bus',
    'butterfly': 'papillon', 'camel': 'chameau', 'can': 'canette',
    'castle': 'château', 'caterpillar': 'chenille', 'cattle': 'bétail',
    'chair': 'chaise', 'chimpanzee': 'chimpanzé', 'clock': 'horloge',
    'cloud': 'nuage', 'cockroach': 'cafard', 'couch': 'canapé',
    'crab': 'crabe', 'crocodile': 'crocodile', 'cup': 'tasse',
    'dinosaur': 'dinosaure', 'dolphin': 'dauphin', 'elephant': 'éléphant',
    'flatfish': 'poisson plat', 'forest': 'forêt', 'fox': 'renard',
    'girl': 'fille', 'hamster': 'hamster', 'house': 'maison',
    'kangaroo': 'kangourou', 'keyboard': 'clavier', 'lamp': 'lampe',
    'lawn_mower': 'tondeuse à gazon', 'leopard': 'léopard', 'lion': 'lion',
    'lizard': 'lézard', 'lobster': 'homard', 'man': 'homme',
    'maple_tree': 'érable', 'motorcycle': 'moto', 'mountain': 'montagne',
    'mouse': 'souris', 'mushroom': 'champignon', 'oak_tree': 'chêne',
    'orange': 'orange', 'orchid': 'orchidée', 'otter': 'loutre',
    'palm_tree': 'palmier', 'pear': 'poire', 'pickup_truck': 'camionnette',
    'pine_tree': 'pin', 'plain': 'plaine', 'plate': 'assiette',
    'poppy': 'coquelicot', 'porcupine': 'porc-épic', 'possum': 'opossum',
    'rabbit': 'lapin', 'raccoon': 'raton laveur', 'ray': 'raie',
    'road': 'route', 'rocket': 'fusée', 'rose': 'rose',
    'sea': 'mer', 'seal': 'phoque', 'shark': 'requin',
    'shrew': 'musaraigne', 'skunk': 'mouffette', 'skyscraper': 'gratte-ciel',
    'snail': 'escargot', 'snake': 'serpent', 'spider': 'araignée',
    'squirrel': 'écureuil', 'streetcar': 'tramway', 'sunflower': 'tournesol',
    'sweet_pepper': 'poivron', 'table': 'table', 'tank': 'char',
    'telephone': 'téléphone', 'television': 'télévision', 'tiger': 'tigre',
    'tractor': 'tracteur', 'train': 'train', 'trout': 'truite',
    'tulip': 'tulipe', 'turtle': 'tortue', 'wardrobe': 'armoire',
    'whale': 'baleine', 'willow_tree': 'saule', 'wolf': 'loup',
    'woman': 'femme', 'worm': 'ver'
}

cifar100_classes_en = dataset.classes
cifar100_classes_fr = [translation_dict.get(cls, cls) for cls in cifar100_classes_en]

templates_en = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of many {}.",
]

templates_fr = [
    "une photo d'un {}.",
    "une photo floue d'un {}.",
    "une photo de plusieurs {}.",
]

def compute_zeroshot_weights(classnames, templates):
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(classnames, desc="Computing weights"):
            texts = [template.format(classname) for template in templates]
            texts_tokenized = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts_tokenized)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1).to(device)

def evaluate_per_class(zeroshot_weights, dataloader):
    """Evaluate per-class performance for detailed analysis"""
    class_correct = torch.zeros(100).to(device)
    class_total = torch.zeros(100).to(device)
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
            _, indices = similarity.topk(1)
            
            correct = indices.view(-1).eq(labels)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    accuracies = (class_correct / class_total * 100).cpu().numpy()
    return accuracies

# Computing weights
print("Computing English weights...")
weights_en = compute_zeroshot_weights(cifar100_classes_en, templates_en)

print("\nComputing French weights...")
weights_fr = compute_zeroshot_weights(cifar100_classes_fr, templates_fr)

print("\nComputing Multilingual weights...")
weights_multi = (weights_en + weights_fr) / 2
weights_multi /= weights_multi.norm(dim=0, keepdim=True)

# Evaluating per class
print("\nEvaluating English...")
acc_per_class_en = evaluate_per_class(weights_en, dataloader)

print("\nEvaluating French...")
acc_per_class_fr = evaluate_per_class(weights_fr, dataloader)

print("\nEvaluating Multilingual...")
acc_per_class_multi = evaluate_per_class(weights_multi, dataloader)

# ============================================================================
# VISUALISATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CLIP Multilingual Performance Analysis on CIFAR-100', fontsize=16, fontweight='bold')

# 1. Bar chart of overall performance
ax1 = axes[0, 0]
languages = ['English', 'French', 'Multilingual\n(EN+FR)']
accuracies = [acc_per_class_en.mean(), acc_per_class_fr.mean(), acc_per_class_multi.mean()]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(languages, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Zero-Shot Performance', fontsize=13, fontweight='bold')
ax1.set_ylim([0, 70])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Distribution of per-class performance
ax2 = axes[0, 1]
ax2.hist([acc_per_class_en, acc_per_class_fr, acc_per_class_multi], 
         bins=20, label=['English', 'French', 'Multilingual'], 
         color=colors, alpha=0.6, edgecolor='black')
ax2.set_xlabel('Accuracy per class (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of classes', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Per-Class Accuracies', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, linestyle='--')

# 3. Comparison English vs French (scatter plot)
ax3 = axes[1, 0]
ax3.scatter(acc_per_class_en, acc_per_class_fr, alpha=0.6, s=50, color='purple', edgecolor='black', linewidth=0.5)
ax3.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=2, label='Perfect agreement')
ax3.set_xlabel('English Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('French Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('English vs French Performance (per class)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xlim([0, 100])
ax3.set_ylim([0, 100])

# Calculate correlation
correlation = np.corrcoef(acc_per_class_en, acc_per_class_fr)[0, 1]
ax3.text(5, 90, f'Correlation: {correlation:.3f}', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Multilingual gain (per class)
ax4 = axes[1, 1]
gain = acc_per_class_multi - acc_per_class_en
sorted_indices = np.argsort(gain)
top_gainers = sorted_indices[-10:]  # Top 10 classes benefiting the most
top_losers = sorted_indices[:10]    # Top 10 classes losing the most

# Display top gainers
colors_gain = ['green' if g > 0 else 'red' for g in gain[top_gainers]]
ax4.barh(range(10), gain[top_gainers], color=colors_gain, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(10))
ax4.set_yticklabels([cifar100_classes_en[i] for i in top_gainers], fontsize=9)
ax4.set_xlabel('Accuracy Gain/Loss (%)', fontsize=11, fontweight='bold')
ax4.set_title('Top 10 Classes: Multilingual vs English', fontsize=13, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

output_filename = 'multilingual_clip_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n Visualization saved as '{output_filename}' in current directory")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

print(f"\n OVERALL ACCURACY:")
print(f"   English:      {acc_per_class_en.mean():.2f}% (±{acc_per_class_en.std():.2f}%)")
print(f"   French:       {acc_per_class_fr.mean():.2f}% (±{acc_per_class_fr.std():.2f}%)")
print(f"   Multilingual: {acc_per_class_multi.mean():.2f}% (±{acc_per_class_multi.std():.2f}%)")

print(f"\n CROSS-LINGUAL ANALYSIS:")
print(f"   Correlation (EN vs FR): {correlation:.3f}")
print(f"   Mean absolute difference: {np.abs(acc_per_class_en - acc_per_class_fr).mean():.2f}%")

print(f"\n MULTILINGUAL BENEFIT:")
gain_mean = gain.mean()
gain_positive = (gain > 0).sum()
gain_negative = (gain < 0).sum()
print(f"   Mean gain vs English: {gain_mean:+.2f}%")
print(f"   Classes improved: {gain_positive}/100")
print(f"   Classes degraded: {gain_negative}/100")

print(f"\n TOP 5 CLASSES BENEFITING FROM MULTILINGUAL:")
for i in sorted_indices[-5:][::-1]:
    print(f"   {cifar100_classes_en[i]:20s}: {gain[i]:+.2f}% (EN: {acc_per_class_en[i]:.1f}% → Multi: {acc_per_class_multi[i]:.1f}%)")

print(f"\n TOP 5 CLASSES HURT BY MULTILINGUAL:")
for i in sorted_indices[:5]:
    print(f"   {cifar100_classes_en[i]:20s}: {gain[i]:+.2f}% (EN: {acc_per_class_en[i]:.1f}% → Multi: {acc_per_class_multi[i]:.1f}%)")

print("\n" + "="*70)
print(f"\n Image saved in: {os.path.abspath(output_filename)}")
print("="*70)

plt.show()