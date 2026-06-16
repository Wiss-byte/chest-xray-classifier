# Détection de Pneumonie sur Radiographies

## ⚠️ IMPORTANT : Projet Éducatif Uniquement

**Ce projet est à but PÉDAGOGIQUE uniquement.** Il ne doit **jamais** être utilisé comme outil clinique ou de diagnostic médical. 

Pour un usage médical réel, consultez des professionnels de santé qualifiés.

---

## Objectif

Implémenter un projet de **vision par ordinateur** pour détecter la pneumonie sur des radiographies thoraciques en utilisant le deep learning avec PyTorch.

## Dataset

- **Source** : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Taille** : ~5863 images au total
- **Classes** : NORMAL vs PNEUMONIA (déséquilibrée 1:3)
- **Split** : 
  - Train : 4695 images
  - Validation : 521 images  
  - Test : 624 images

## Architecture

### 1. CNN Simple (Baseline)
Input [B, 3, 224, 224]
├─ Conv2d(3→16) + BatchNorm + ReLU + MaxPool2d
├─ Conv2d(16→32) + BatchNorm + ReLU + MaxPool2d
├─ Conv2d(32→64) + BatchNorm + ReLU + AdaptiveAvgPool2d
└─ FC(64→128) + ReLU + Dropout(0.5) + FC(128→1)
Output [B, 1] (logit binaire)

**Résultats CNN :**
- Accuracy : 67%
- Recall PNEUMONIA : 99%
- Precision PNEUMONIA : 66%
- AUC-ROC : 0.9007

### 2. ResNet18 Fine-tuning (Transfer Learning)

Modèle ResNet18 pré-entraîné sur ImageNet avec :
- Couches conv gelées (frozen)
- Dernière couche FC remplacée : `Linear(512→128) + ReLU + Dropout(0.3) + Linear(128→1)`

**Résultats ResNet18 :**
- **Accuracy : 90%** ✅
- Recall NORMAL : 81% ✅ (vs 15% CNN)
- Recall PNEUMONIA : 95% ✅ (vs 99% CNN)
- Precision NORMAL : 91% ✅ (vs 87% CNN)
- Precision PNEUMONIA : 89% ✅ (vs 66% CNN)
- **AUC-ROC : 0.9546** ✅

### Comparaison

ResNet18 **outperform** CNN sur tous les métriques grâce au transfer learning.

---

## Loss & Optimisation

- **Loss** : `BCEWithLogitsLoss` avec `pos_weight=1.5`
  - Gère le déséquilibre des classes
  
- **Optimiseur** : Adam
  - CNN : lr=1e-3
  - ResNet18 : lr=1e-4 (fine-tuning)

---

## Métriques

| Métrique | CNN | ResNet18 |
|----------|-----|----------|
| Accuracy | 67% | **90%** |
| Recall PNEUMONIA | 99% | 95% |
| Recall NORMAL | 15% | **81%** |
| Precision PNEUMONIA | 66% | **89%** |
| Precision NORMAL | 87% | **91%** |
| F1 PNEUMONIA | 0.79 | **0.92** |
| AUC-ROC | 0.9007 | **0.9546** |

---

## Utilisation

### Installation

```bash
uv add torch torchvision torchaudio matplotlib scikit-learn pillow tqdm kaggle
```

### Entraînement

```bash
# CNN
uv run python -m src.train

# ResNet18 seulement
uv run python train_resnet_only.py
```

### Évaluation

```bash
uv run python main.py evaluate
```

### Inférence (Test sur une image)

```python
from src.infer import predict

result = predict('path/to/image.jpg')
print(f"Prédiction : {result['prediction']}")
print(f"Confiance PNEUMONIA : {result['PNEUMONIA']:.2%}")
```

Ou exécute :
```bash
uv run python test_infer.py
```

### Visualisation Grad-CAM

```bash
uv run python gradcam_v2.py
```

Génère `gradcam_result.png` montrant où le modèle regarde dans l'image.

---

## Structure du Projet
pneumonia-detection/
├── src/
│   ├── init.py
│   ├── dataset.py          # Chargement & augmentation données
│   ├── model.py            # CNN simple
│   ├── model_resnet.py     # ResNet18 fine-tuning
│   ├── train.py            # Boucle d'entraînement
│   ├── evaluate.py         # Évaluation test set
│   └── infer.py            # Inférence sur nouvelles images
├── main.py                 # Interface train/evaluate
├── test_infer.py           # Test inférence
├── train_resnet_only.py    # Entraînement ResNet18
├── eval_resnet.py          # Évaluation ResNet18
├── gradcam_v2.py           # Visualisation Grad-CAM
├── best_cnn.pth            # Poids CNN entraîné
├── best_resnet18.pth       # Poids ResNet18 entraîné
├── gradcam_result.png      # Visualisation Grad-CAM
├── pyproject.toml          # Config uv
└── README.md

---

## Augmentation de Données

Pour réduire l'overfitting :
- RandomHorizontalFlip
- RandomRotation(15°)
- ColorJitter (brightness, contrast)
- RandomAffine (translation)

---

## Extensions & Améliorations

✅ **Complétées :**
- CNN simple entraîné
- ResNet18 fine-tuning  
- Augmentation de données
- Grad-CAM visualization
- Comparaison modèles

⚠️ **Possibles :**
- Ensemble de modèles (voting)
- Fine-tuning plus agressif
- Data balancing (oversampling)
- Batch normalization tuning
- Cross-validation

---

## Notes Importantes

1. **Pas d'outil clinique** : Ce modèle atteint 90% d'accuracy mais n'est pas assez fiable pour un usage médical réel.

2. **Faux négatifs critiques** : Rappel PNEUMONIA = 95% → 5% de cas manqués (dangereux médicalement).

3. **Contexte éducatif** : Le projet démontre les concepts de deep learning, transfer learning, et class imbalance.

4. **Données limitées** : Seulement 624 images de test. Un vrai système médical aurait besoin de milliers d'images.

---

## Résultats Finaux

✅ **ResNet18 est le meilleur modèle** avec :
- 90% d'accuracy globale
- 81% de recall sur NORMAL (moins de faux positifs)
- 95% de recall sur PNEUMONIA (détecte la plupart des cas)
- 0.9546 AUC-ROC (discrimination excellente)

