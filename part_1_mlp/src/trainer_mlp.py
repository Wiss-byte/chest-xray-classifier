import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path

class MLPTrainer:
    """Entraîneur pour modèles MLP"""
    
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
            
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
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
    
    def train(self, train_loader, val_loader, epochs=50, patience=10, save_path="best_mlp.pth"):
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
    
    def save_checkpoint(self, path):
        """Sauvegarde complète"""
        Path(path).parent.mkdir(exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint sauvegardé : {path}")
    
    def load_checkpoint(self, path, device):
        """Charge un checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"✅ Checkpoint chargé : {path}")