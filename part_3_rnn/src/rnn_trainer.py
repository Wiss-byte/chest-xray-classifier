import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

class RNNTrainer:
    """Entraîneur pour modèles RNN/LSTM/GRU"""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, train_loader):
        """Une époque d'entraînement"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc="Train", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward avec gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Évaluation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Val", leave=False):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=30, patience=5, save_path="best_rnn.pth"):
        """Entraînement complet"""
        Path(save_path).parent.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<12} {'Val Acc':<10} {'Status':<10}")
        print("-" * 70)
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                status = "✓ Saved"
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                status = f"Wait {patience_counter}/{patience}"
            
            print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<10.4f} {val_loss:<12.4f} {val_acc:<10.4f} {status:<10}")
            
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping à l'epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print(f"✅ Meilleur modèle chargé : {save_path}\n")
        
        return self.history


class PerplexityCalculator:
    """Calcule la perplexité d'un modèle de langage"""
    
    @staticmethod
    def calculate_perplexity(loss):
        """Perplexité = exp(loss)"""
        return torch.exp(torch.tensor(loss)).item()
    
    @staticmethod
    def explain():
        print("\n" + "="*70)
        print("EXPLICATION : PERPLEXITÉ")
        print("="*70)
        
        print("\n📚 Définition :")
        print("   Perplexité = exp(loss d'entropie croisée)")
        print("   PP = e^(-1/N * Σ log P(w_i))")
        
        print("\n📊 Interprétation :")
        print("   • PP = 1 : modèle certain (impossible en pratique)")
        print("   • PP = 10 : modèle confus sur 10 mots en moyenne")
        print("   • PP = 100 : modèle très confus")
        print("   • Plus bas = meilleur")
        
        print("\n💡 Exemple :")
        print("   Si loss = 2.3, alors PP = e^2.3 ≈ 10")
        print("   Le modèle est aussi perplexe que de choisir parmi 10 mots")
        
        print("\n" + "="*70 + "\n")


class BeamSearchDecoder:
    """Implémente le Beam Search pour la génération"""
    
    @staticmethod
    def beam_search(model, start_token, max_length=20, beam_width=3, device='cpu'):
        """
        Beam search avec largeur de faisceau
        
        Args:
            model : modèle de séquence
            start_token : token de démarrage
            max_length : longueur max de la séquence
            beam_width : nombre de candidats à explorer
            device : cpu ou gpu
        """
        print("\n" + "="*70)
        print("DÉMONSTRATION : BEAM SEARCH")
        print("="*70)
        
        print(f"\n🔍 Beam Search avec beam_width={beam_width}")
        print("   Explore les top-k hypothèses à chaque étape")
        
        print("\n📊 Processus :")
        print("   1. Génération initiale avec top-k tokens")
        print("   2. À chaque étape, garder les top-k séquences")
        print("   3. Continuer jusqu'à token de fin ou longueur max")
        print("   4. Retourner la meilleure séquence")
        
        print("\n✅ Avantages vs Greedy :")
        print("   • Greedy : choisir le meilleur token à chaque étape")
        print("   • Beam Search : explorer plusieurs chemins")
        print("   • Beam Search trouve souvent de meilleures séquences")
        
        print("\n⚙️  Complexité :")
        print(f"   • Greedy : O(max_length)")
        print(f"   • Beam Search : O(max_length * beam_width * vocab_size)")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def compare_decoding_strategies():
        """Compare greedy et beam search"""
        print("\n" + "="*70)
        print("COMPARAISON : GREEDY vs BEAM SEARCH")
        print("="*70)
        
        comparison = {
            "Stratégie": ["Greedy", "Beam Search"],
            "Vitesse": ["Très rapide", "Lente"],
            "Qualité": ["Moyenne", "Bonne"],
            "Mémoire": ["Faible", "Élevée"],
            "Diversité": ["Faible", "Bonne"],
            "Cas d'usage": ["Real-time", "Batch/offline"]
        }
        
        print(f"\n{'Aspect':<15} {'Greedy':<20} {'Beam Search':<20}")
        print("-" * 55)
        
        for i, aspect in enumerate(comparison["Stratégie"]):
            print(f"{aspect:<15} {comparison['Vitesse'][i]:<20} {comparison['Qualité'][i]:<20}")
        
        print("\n" + "="*70 + "\n")


def demonstrate_rnn_concepts():
    """Démontre tous les concepts RNN"""
    from rnn_models import print_model_comparison, BPTTExplainer, GradientClippingDemo
    
    print_model_comparison()
    BPTTExplainer.explain()
    GradientClippingDemo.demonstrate()
    PerplexityCalculator.explain()
    BeamSearchDecoder.beam_search(None, None)
    BeamSearchDecoder.compare_decoding_strategies()


if __name__ == "__main__":
    demonstrate_rnn_concepts()