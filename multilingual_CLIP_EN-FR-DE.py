import os
import clip
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================================
# Multilingual CLIP with 3 languages (EN + FR + DE)
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

model, preprocess = clip.load("ViT-L/14@336px", device=device) # or "ViT-B/32"

dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ============================================================================
# Translations : English, French, German
# ============================================================================
cifar100_classes_en = dataset.classes

# French translation
translation_fr = {
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

# German translation
translation_de = {
    'apple': 'Apfel', 'aquarium_fish': 'Aquarienfisch', 'baby': 'Baby',
    'bear': 'Bär', 'beaver': 'Biber', 'bed': 'Bett', 'bee': 'Biene',
    'beetle': 'Käfer', 'bicycle': 'Fahrrad', 'bottle': 'Flasche',
    'bowl': 'Schüssel', 'boy': 'Junge', 'bridge': 'Brücke', 'bus': 'Bus',
    'butterfly': 'Schmetterling', 'camel': 'Kamel', 'can': 'Dose',
    'castle': 'Schloss', 'caterpillar': 'Raupe', 'cattle': 'Vieh',
    'chair': 'Stuhl', 'chimpanzee': 'Schimpanse', 'clock': 'Uhr',
    'cloud': 'Wolke', 'cockroach': 'Kakerlake', 'couch': 'Sofa',
    'crab': 'Krabbe', 'crocodile': 'Krokodil', 'cup': 'Tasse',
    'dinosaur': 'Dinosaurier', 'dolphin': 'Delfin', 'elephant': 'Elefant',
    'flatfish': 'Plattfisch', 'forest': 'Wald', 'fox': 'Fuchs',
    'girl': 'Mädchen', 'hamster': 'Hamster', 'house': 'Haus',
    'kangaroo': 'Känguru', 'keyboard': 'Tastatur', 'lamp': 'Lampe',
    'lawn_mower': 'Rasenmäher', 'leopard': 'Leopard', 'lion': 'Löwe',
    'lizard': 'Eidechse', 'lobster': 'Hummer', 'man': 'Mann',
    'maple_tree': 'Ahornbaum', 'motorcycle': 'Motorrad', 'mountain': 'Berg',
    'mouse': 'Maus', 'mushroom': 'Pilz', 'oak_tree': 'Eiche',
    'orange': 'Orange', 'orchid': 'Orchidee', 'otter': 'Otter',
    'palm_tree': 'Palme', 'pear': 'Birne', 'pickup_truck': 'Pickup',
    'pine_tree': 'Kiefer', 'plain': 'Ebene', 'plate': 'Teller',
    'poppy': 'Mohn', 'porcupine': 'Stachelschwein', 'possum': 'Opossum',
    'rabbit': 'Kaninchen', 'raccoon': 'Waschbär', 'ray': 'Rochen',
    'road': 'Straße', 'rocket': 'Rakete', 'rose': 'Rose',
    'sea': 'Meer', 'seal': 'Robbe', 'shark': 'Hai',
    'shrew': 'Spitzmaus', 'skunk': 'Stinktier', 'skyscraper': 'Wolkenkratzer',
    'snail': 'Schnecke', 'snake': 'Schlange', 'spider': 'Spinne',
    'squirrel': 'Eichhörnchen', 'streetcar': 'Straßenbahn', 'sunflower': 'Sonnenblume',
    'sweet_pepper': 'Paprika', 'table': 'Tisch', 'tank': 'Panzer',
    'telephone': 'Telefon', 'television': 'Fernseher', 'tiger': 'Tiger',
    'tractor': 'Traktor', 'train': 'Zug', 'trout': 'Forelle',
    'tulip': 'Tulpe', 'turtle': 'Schildkröte', 'wardrobe': 'Kleiderschrank',
    'whale': 'Wal', 'willow_tree': 'Weide', 'wolf': 'Wolf',
    'woman': 'Frau', 'worm': 'Wurm'
}

cifar100_classes_fr = [translation_fr.get(cls, cls) for cls in cifar100_classes_en]
cifar100_classes_de = [translation_de.get(cls, cls) for cls in cifar100_classes_en]

# ============================================================================
# Multilingual templates
# ============================================================================
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

templates_de = [
    "ein Foto von einem {}.",
    "ein verschwommenes Foto von einem {}.",
    "ein Foto von vielen {}.",
]

# ============================================================================
# Utility functions
# ============================================================================
def compute_zeroshot_weights(classnames, templates, language):
    print(f"\nComputing zero-shot weights for {language.upper()}...")
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            texts_tokenized = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts_tokenized)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1).to(device)

def evaluate_per_class(zeroshot_weights, dataloader, description):
    print(f"Evaluating: {description}...")
    class_correct = torch.zeros(100).to(device)
    class_total = torch.zeros(100).to(device)
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
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

# ============================================================================
# Computing weights for each language
# ============================================================================
weights_en = compute_zeroshot_weights(cifar100_classes_en, templates_en, "English")
weights_fr = compute_zeroshot_weights(cifar100_classes_fr, templates_fr, "French")
weights_de = compute_zeroshot_weights(cifar100_classes_de, templates_de, "German")

# Multilingual combinations
print("\nComputing multilingual combinations...")
weights_en_fr = (weights_en + weights_fr) / 2
weights_en_fr /= weights_en_fr.norm(dim=0, keepdim=True)

weights_en_de = (weights_en + weights_de) / 2
weights_en_de /= weights_en_de.norm(dim=0, keepdim=True)

weights_fr_de = (weights_fr + weights_de) / 2
weights_fr_de /= weights_fr_de.norm(dim=0, keepdim=True)

weights_all = (weights_en + weights_fr + weights_de) / 3
weights_all /= weights_all.norm(dim=0, keepdim=True)

# ============================================================================
# Evaluation of all configurations
# ============================================================================
print("\n" + "="*70)
print("EVALUATING ALL CONFIGURATIONS")
print("="*70)

acc_en = evaluate_per_class(weights_en, dataloader, "English")
acc_fr = evaluate_per_class(weights_fr, dataloader, "French")
acc_de = evaluate_per_class(weights_de, dataloader, "German")
acc_en_fr = evaluate_per_class(weights_en_fr, dataloader, "EN+FR")
acc_en_de = evaluate_per_class(weights_en_de, dataloader, "EN+DE")
acc_fr_de = evaluate_per_class(weights_fr_de, dataloader, "FR+DE")
acc_all = evaluate_per_class(weights_all, dataloader, "EN+FR+DE")

# ============================================================================
# ADVANCED VISUALIZATIONS
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Comparison of the 3 individual languages
ax1 = fig.add_subplot(gs[0, 0])
languages = ['English', 'French', 'German']
accuracies_single = [acc_en.mean(), acc_fr.mean(), acc_de.mean()]
colors_single = ['#3498db', '#e74c3c', '#f39c12']
bars = ax1.bar(languages, accuracies_single, color=colors_single, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Single Language Performance', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 70])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, acc in zip(bars, accuracies_single):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# 2. Bilingual comparisons
ax2 = fig.add_subplot(gs[0, 1])
bilingual = ['EN+FR', 'EN+DE', 'FR+DE']
accuracies_bi = [acc_en_fr.mean(), acc_en_de.mean(), acc_fr_de.mean()]
colors_bi = ['#9b59b6', '#16a085', '#d35400']
bars2 = ax2.bar(bilingual, accuracies_bi, color=colors_bi, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Bilingual Combinations', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 70])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar, acc in zip(bars2, accuracies_bi):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# 3. Trilingual
ax3 = fig.add_subplot(gs[0, 2])
all_configs = ['English\n(best single)', 'EN+FR+DE\n(trilingual)']
acc_comparison = [acc_en.mean(), acc_all.mean()]
colors_tri = ['#3498db', '#2ecc71']
bars3 = ax3.bar(all_configs, acc_comparison, color=colors_tri, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Best Single vs Trilingual', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 70])
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar, acc in zip(bars3, acc_comparison):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Triangle plot: correlations between languages
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(acc_en, acc_fr, alpha=0.5, s=40, color='purple', edgecolor='black', linewidth=0.5, label='EN vs FR')
ax4.scatter(acc_en, acc_de, alpha=0.5, s=40, color='orange', edgecolor='black', linewidth=0.5, label='EN vs DE')
ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=2)
ax4.set_xlabel('English Accuracy (%)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Other Language Accuracy (%)', fontsize=10, fontweight='bold')
ax4.set_title('Cross-Lingual Correlations', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3, linestyle='--')
ax4.set_xlim([0, 100])
ax4.set_ylim([0, 100])

corr_en_fr = np.corrcoef(acc_en, acc_fr)[0, 1]
corr_en_de = np.corrcoef(acc_en, acc_de)[0, 1]
ax4.text(5, 90, f'Corr(EN,FR): {corr_en_fr:.3f}\nCorr(EN,DE): {corr_en_de:.3f}', 
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. French vs German
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(acc_fr, acc_de, alpha=0.6, s=40, color='green', edgecolor='black', linewidth=0.5)
ax5.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=2)
ax5.set_xlabel('French Accuracy (%)', fontsize=10, fontweight='bold')
ax5.set_ylabel('German Accuracy (%)', fontsize=10, fontweight='bold')
ax5.set_title('French vs German (per class)', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3, linestyle='--')
ax5.set_xlim([0, 100])
ax5.set_ylim([0, 100])

corr_fr_de = np.corrcoef(acc_fr, acc_de)[0, 1]
ax5.text(5, 90, f'Correlation: {corr_fr_de:.3f}', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 6. Distribution of performances
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist([acc_en, acc_fr, acc_de], bins=20, 
         label=['English', 'French', 'German'],
         color=colors_single, alpha=0.6, edgecolor='black')
ax6.set_xlabel('Accuracy per class (%)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Number of classes', fontsize=10, fontweight='bold')
ax6.set_title('Distribution of Per-Class Accuracies', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3, linestyle='--')

# 7. Top gainers : Trilingual vs English
ax7 = fig.add_subplot(gs[2, :])
gain_tri = acc_all - acc_en
sorted_indices = np.argsort(gain_tri)
top_10_gainers = sorted_indices[-10:]
top_10_losers = sorted_indices[:10]

x_pos = np.arange(20)
gains_to_plot = np.concatenate([gain_tri[top_10_losers], gain_tri[top_10_gainers]])
labels_to_plot = [cifar100_classes_en[i] for i in top_10_losers] + \
                 [cifar100_classes_en[i] for i in top_10_gainers]
colors_gain = ['red' if g < 0 else 'green' for g in gains_to_plot]

ax7.barh(x_pos, gains_to_plot, color=colors_gain, alpha=0.7, edgecolor='black')
ax7.set_yticks(x_pos)
ax7.set_yticklabels(labels_to_plot, fontsize=8)
ax7.set_xlabel('Accuracy Gain/Loss (%) - Trilingual vs English', fontsize=11, fontweight='bold')
ax7.set_title('Top 10 Losers (left) and Top 10 Gainers (right): Trilingual vs English', 
              fontsize=12, fontweight='bold')
ax7.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax7.grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('CLIP Trilingual Performance Analysis (EN + FR + DE) on CIFAR-100', 
             fontsize=16, fontweight='bold', y=0.995)

output_filename = 'multilingual_clip_3languages.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n Visualization saved as '{output_filename}'")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*70)
print("DETAILED STATISTICS - 3 LANGUAGES COMPARISON")
print("="*70)

print(f"\n SINGLE LANGUAGE PERFORMANCE:")
print(f"   English:  {acc_en.mean():.2f}% (±{acc_en.std():.2f}%)")
print(f"   French:   {acc_fr.mean():.2f}% (±{acc_fr.std():.2f}%)")
print(f"   German:   {acc_de.mean():.2f}% (±{acc_de.std():.2f}%)")

print(f"\n BILINGUAL COMBINATIONS:")
print(f"   EN+FR:    {acc_en_fr.mean():.2f}% (±{acc_en_fr.std():.2f}%)")
print(f"   EN+DE:    {acc_en_de.mean():.2f}% (±{acc_en_de.std():.2f}%)")
print(f"   FR+DE:    {acc_fr_de.mean():.2f}% (±{acc_fr_de.std():.2f}%)")

print(f"\n TRILINGUAL:")
print(f"   EN+FR+DE: {acc_all.mean():.2f}% (±{acc_all.std():.2f}%)")

print(f"\n CROSS-LINGUAL CORRELATIONS:")
print(f"   EN vs FR: {corr_en_fr:.3f}")
print(f"   EN vs DE: {corr_en_de:.3f}")
print(f"   FR vs DE: {corr_fr_de:.3f}")

print(f"\n LANGUAGE COMPARISON:")
print(f"   French gap vs English: {acc_en.mean() - acc_fr.mean():.2f}%")
print(f"   German gap vs English: {acc_en.mean() - acc_de.mean():.2f}%")
print(f"   French vs German:      {acc_fr.mean() - acc_de.mean():+.2f}%")

print(f"\n BEST CONFIGURATION:")
best_config = max([
    ('English', acc_en.mean()),
    ('EN+FR', acc_en_fr.mean()),
    ('EN+DE', acc_en_de.mean()),
    ('EN+FR+DE', acc_all.mean())
], key=lambda x: x[1])
print(f"   {best_config[0]}: {best_config[1]:.2f}%")

print(f"\n TOP 5 CLASSES WHERE GERMAN OUTPERFORMS FRENCH:")
diff_de_fr = acc_de - acc_fr
top_de_over_fr = np.argsort(diff_de_fr)[-5:][::-1]
for i in top_de_over_fr:
    print(f"   {cifar100_classes_en[i]:20s}: DE {acc_de[i]:.1f}% vs FR {acc_fr[i]:.1f}% (+{diff_de_fr[i]:.1f}%)")

print(f"\n TOP 5 CLASSES WHERE FRENCH OUTPERFORMS GERMAN:")
top_fr_over_de = np.argsort(diff_de_fr)[:5]
for i in top_fr_over_de:
    print(f"   {cifar100_classes_en[i]:20s}: FR {acc_fr[i]:.1f}% vs DE {acc_de[i]:.1f}% (+{-diff_de_fr[i]:.1f}%)")

print("\n" + "="*70)
print(f" Full results saved in: {os.path.abspath(output_filename)}")
print("="*70)

plt.show()