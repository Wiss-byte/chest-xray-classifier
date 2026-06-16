import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

class MLPWithBatchNorm(nn.Module):
    """MLP amélioré avec Batch Normalization"""
    
    def __init__(self, input_dim=13, hidden_dims=[128, 64, 32], num_classes=3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # ← BATCH NORM
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FeatureImportanceAnalyzer:
    """Calcule l'importance des features"""
    
    @staticmethod
    def permutation_importance_pytorch(model, X, y, device, n_repeats=10):
        """Importance par permutation"""
        print("\n" + "="*70)
        print("ANALYSE : IMPORTANCE DES FEATURES (PERMUTATION)")
        print("="*70)
        
        model.eval()
        
        # Baseline loss
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        with torch.no_grad():
            baseline_pred = model(X_tensor)
        
        baseline_loss = nn.CrossEntropyLoss()(baseline_pred, y_tensor).item()
        
        importances = []
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        for feature_idx in range(X.shape[1]):
            losses = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                
                X_perm_tensor = torch.FloatTensor(X_permuted).to(device)
                
                with torch.no_grad():
                    pred = model(X_perm_tensor)
                
                loss = nn.CrossEntropyLoss()(pred, y_tensor).item()
                losses.append(loss - baseline_loss)
            
            importance = np.mean(losses)
            importances.append(importance)
        
        # Trier par importance
        importance_df = [(feature_names[i], importances[i]) 
                        for i in range(len(feature_names))]
        importance_df.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n📊 Importance des features (n_repeats={n_repeats}):\n")
        print(f"{'Feature':<20} {'Importance':<15} {'Rang':<10}")
        print("-" * 50)
        
        for rank, (name, imp) in enumerate(importance_df, 1):
            print(f"{name:<20} {imp:>14.4f} {'#' + str(rank):<9}")
        
        print("\n" + "="*70 + "\n")
        
        return importance_df
    
    @staticmethod
    def sklearn_permutation_importance(model_rf, X, y, feature_names):
        """Importance avec scikit-learn (pour comparaison)"""
        print("\n" + "="*70)
        print("COMPARAISON : RANDOM FOREST FEATURE IMPORTANCE")
        print("="*70)
        
        result = permutation_importance(model_rf, X, y, n_repeats=10, random_state=42)
        
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]
        
        print(f"\n📊 Top 10 features (Random Forest):\n")
        print(f"{'Rank':<6} {'Feature':<20} {'Importance':<15}")
        print("-" * 45)
        
        for rank, idx in enumerate(indices[:10], 1):
            print(f"#{rank:<5} {feature_names[idx]:<20} {importances[idx]:>14.4f}")
        
        print("\n" + "="*70 + "\n")
        
        return indices, importances
    
    @staticmethod
    def plot_feature_importance(importance_df, save_path=None):
        """Visualise l'importance des features"""
        features = [x[0] for x in importance_df]
        scores = [x[1] for x in importance_df]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if s > 0 else 'red' for s in scores]
        ax.barh(features, scores, color=colors, alpha=0.7)
        
        ax.set_xlabel('Importance (Loss Increase)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Permutation Feature Importance - MLP', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Feature importance plot sauvegardé : {save_path}")
        
        plt.close()


def demonstrate_improved_mlp():
    """Démontre le MLP amélioré"""
    print("\n" + "="*70)
    print("MLP AMÉLIORÉ : BATCH NORMALIZATION + FEATURE IMPORTANCE")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Charger Wine
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    
    print(f"\n📊 Dataset Wine:")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Normaliser
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Modèle
    model = MLPWithBatchNorm(input_dim=13, hidden_dims=[128, 64, 32], num_classes=3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 Paramètres : {total_params:,}")
    print("\nArchitecture:")
    print(model)
    
    # Entraîner
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n⏳ Entraînement (30 epochs)...")
    
    for epoch in range(30):
        model.train()
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/30 - Loss: {loss.item():.4f}")
    
    # Évaluer
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_t)
        _, preds = torch.max(outputs, 1)
    
    accuracy = (preds == y_test_t).sum().item() / len(y_test_t)
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    
    # Feature Importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    analyzer = FeatureImportanceAnalyzer()
    importance_df = analyzer.permutation_importance_pytorch(model, X_test, y_test, device)
    
    # Visualiser
    analyzer.plot_feature_importance(importance_df, save_path="../results/feature_importance_mlp.png")
    
    # Comparaison avec Random Forest
    print("\n💡 Comparaison : Random Forest Feature Importance")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    indices, importances = analyzer.sklearn_permutation_importance(
        rf, X_test, y_test, feature_names
    )
    
    print("\n✅ Analyse MLP amélioré terminée !")


if __name__ == "__main__":
    demonstrate_improved_mlp()