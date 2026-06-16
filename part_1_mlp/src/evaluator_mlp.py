import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

class MLPEvaluator:
    """Évaluateur pour modèles MLP"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def get_predictions(self, test_loader):
        """Obtient les prédictions"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc="Eval", leave=False):
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def evaluate(self, test_loader):
        """Évaluation complète"""
        labels, preds, probs = self.get_predictions(test_loader)
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        auc_score = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        
        print("\n" + "="*50)
        print("📊 MÉTRIQUES D'ÉVALUATION")
        print("="*50)
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"AUC-ROC   : {auc_score:.4f}")
        print("="*50 + "\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'labels': labels,
            'preds': preds,
            'probs': probs
        }
    
    def plot_confusion_matrix(self, labels, preds, save_path=None):
        """Matrice de confusion"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Classe 0', 'Classe 1', 'Classe 2'],
                    yticklabels=['Classe 0', 'Classe 1', 'Classe 2'])
        plt.title('Matrice de Confusion')
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if save_path:
            print(f"✅ Matrice de confusion sauvegardée : {save_path}")
    
    def plot_roc_curve(self, labels, probs, save_path=None):
        """Courbe ROC (One-vs-Rest)"""
        from sklearn.preprocessing import label_binarize
        
        # Binariser les labels pour multiclass
        y_bin = label_binarize(labels, classes=[0, 1, 2])
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green']
        for i, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'Classe {i} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Courbe ROC (One-vs-Rest)')
        plt.legend(loc="lower right")
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if save_path:
            print(f"✅ Courbe ROC sauvegardée : {save_path}")
    
    def plot_training_history(self, history, save_path=None):
        """Historique d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Évolution de la Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Évolution de l\'Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if save_path:
            print(f"✅ Historique sauvegardé : {save_path}")