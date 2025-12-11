# CLIP - Multilingual Zero-Shot Classification

Implémentation et amélioration du modèle CLIP pour la classification zero-shot sur CIFAR-100, avec extension multilingue (Anglais, Français, Allemand).

## Structure du projet

```
├── Clip.py                              # Zero-shot basique
├── Clip_ensembling.py                   # Prompt ensembling  
├── multilingual_French_English_CLIP.py  # Amélioration: EN + FR
├── multilingual_french_german.py        # Amélioration: EN + FR + DE
├── requirements.txt                     # Dépendances
└── README.md                            
```

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

**Note**: CLIP sera installé depuis GitHub (inclus dans requirements.txt)

## Exécution

### 1. Reproduction du paper CLIP

```bash
# Zero-shot basique (~63%)
python Clip.py

# Prompt ensembling (~63.72%)
python Clip_ensembling.py
```

### 2. Amélioration multilingue

```bash
# Comparaison Anglais vs Français
python multilingual_French_English_CLIP.py

# Comparaison 3 langues (Anglais, Français, Allemand)
python multilingual_french_german.py
```

## Résultats obtenus

### Implémentations du paper

| Méthode | Accuracy CIFAR-100 |
|---------|-------------------|
| Zero-shot basique | 62.93% |
| Prompt ensembling | 63.72% |

### Amélioration multilingue

#### Langues individuelles
- **Anglais** : 62.93% (baseline)
- **Français** : 39.80% (×40 mieux que le hasard)
- **Allemand** : 37.20%

#### Combinaisons
- **EN+FR** : 57.95%
- **EN+DE** : 57.78%
- **FR+DE** : 42.09%
- **EN+FR+DE** : 55.86%



## Insights principaux

1. **CLIP possède des capacités multilingues implicites**
   - Fonctionne en français et allemand sans entraînement explicite
   - Corrélations cross-linguales ~0.34

2. **L'anglais reste optimal pour l'usage général**
   - Meilleure performance globale
   - CLIP a été entraîné principalement sur de l'anglais

3. **Le multilinguisme améliore certaines catégories**
   - Gains spectaculaires (+70%) sur des classes spécifiques
   - Chaque langue capture des aspects complémentaires


## Visualisations

Les scripts génèrent des graphiques PNG avec :
- Comparaison des performances par langue
- Corrélations cross-linguales
- Distribution par classe
- Top gainers/losers du multilinguisme

## Références

- **CLIP Paper** : Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision"
- **Dataset** : CIFAR-100 (Krizhevsky, 2009)


---

**Auteurs** : Marine VIEILLARD & Amine GRINI 
**Date** : Décembre 2025