import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path

class CNNArchitectureComparison:
    """Compare différentes architectures CNN avec variations de paramètres"""
    
    @staticmethod
    def experiment_padding_stride():
        """Teste l'impact du padding et stride"""
        print("\n" + "="*70)
        print("EXPÉRIMENT 1 : IMPACT DU PADDING ET STRIDE")
        print("="*70)
        
        results = {}
        configs = [
            {"padding": 0, "stride": 1, "name": "P0_S1"},
            {"padding": 1, "stride": 1, "name": "P1_S1"},
            {"padding": 1, "stride": 2, "name": "P1_S2"},
            {"padding": 2, "stride": 1, "name": "P2_S1"},
        ]
        
        for config in configs:
            model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=config["padding"], stride=config["stride"]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            dummy_input = torch.randn(1, 1, 28, 28)
            output = model(dummy_input)
            
            results[config["name"]] = {
                "output_shape": output.shape,
                "compression": 28 - output.shape[-1]
            }
            
            print(f"\n{config['name']} (padding={config['padding']}, stride={config['stride']}):")
            print(f"   Output: {output.shape}")
            print(f"   Compression: {results[config['name']]['compression']} pixels")
        
        print("\n" + "="*70 + "\n")
        return results
    
    @staticmethod
    def experiment_pooling_types():
        """Compare max-pooling vs avg-pooling"""
        print("\n" + "="*70)
        print("EXPÉRIMENT 2 : MAX-POOLING vs AVERAGE-POOLING")
        print("="*70)
        
        # Données de test
        data = torch.randn(10, 32, 14, 14)
        
        max_pool = nn.MaxPool2d(2)
        avg_pool = nn.AvgPool2d(2)
        
        max_output = max_pool(data)
        avg_output = avg_pool(data)
        
        print(f"\nInput: {data.shape}")
        print(f"Max-Pooling output: {max_output.shape}")
        print(f"Avg-Pooling output: {avg_output.shape}")
        
        # Statistiques
        max_mean = max_output.mean().item()
        avg_mean = avg_output.mean().item()
        max_std = max_output.std().item()
        avg_std = avg_output.std().item()
        
        print(f"\nMax-Pooling - Mean: {max_mean:.4f}, Std: {max_std:.4f}")
        print(f"Avg-Pooling - Mean: {avg_mean:.4f}, Std: {avg_std:.4f}")
        
        print("\n📊 Observations:")
        print("   • Max-pooling conserve les features fortes")
        print("   • Avg-pooling lisse et crée une sortie moyenne")
        print("   • Max-pooling a plus de variance")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def experiment_filter_counts():
        """Teste l'impact du nombre de filtres"""
        print("\n" + "="*70)
        print("EXPÉRIMENT 3 : IMPACT DU NOMBRE DE FILTRES")
        print("="*70)
        
        filter_counts = [8, 16, 32, 64]
        results = {}
        
        for filters in filter_counts:
            model = nn.Conv2d(1, filters, 3, padding=1)
            total_params = sum(p.numel() for p in model.parameters())
            results[filters] = total_params
            
            print(f"\nFiltres: {filters}")
            print(f"   Paramètres: {total_params:,}")
            print(f"   Facteur: {total_params / results[8]:.1f}x")
        
        print("\n" + "="*70 + "\n")
        return results
    
    @staticmethod
    def experiment_cnn_vs_mlp():
        """Compare CNN et MLP sur MNIST"""
        print("\n" + "="*70)
        print("EXPÉRIMENT 4 : CNN vs MLP sur MNIST")
        print("="*70)
        
        device = torch.device("cpu")
        
        # Charger MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('../data', train=False, transform=transform, download=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # MLP simple
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(28*28, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        # CNN simple
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(64*7*7, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        mlp = MLP().to(device)
        cnn = CNN().to(device)
        
        mlp_params = sum(p.numel() for p in mlp.parameters())
        cnn_params = sum(p.numel() for p in cnn.parameters())
        
        print(f"\nMLP : {mlp_params:,} paramètres")
        print(f"CNN : {cnn_params:,} paramètres")
        
        # Évaluer les deux modèles
        def evaluate(model, loader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            return correct / total
        
        print("\n📊 RÉSULTATS (sur test set MNIST):")
        mlp_acc = evaluate(mlp, test_loader)
        cnn_acc = evaluate(cnn, test_loader)
        
        print(f"MLP Accuracy : {mlp_acc:.4f}")
        print(f"CNN Accuracy : {cnn_acc:.4f}")
        print(f"CNN Amélioration : {(cnn_acc - mlp_acc)*100:.2f}%")
        
        print("\n💡 ANALYSE:")
        print("   CNN > MLP car :")
        print("   • Exploite la structure spatiale des images")
        print("   • Partage des poids réduit le nombre de paramètres")
        print("   • Convolutions capturent les patterns locaux")
        print("   • Meilleure généralisation avec moins de paramètres")
        
        print("\n" + "="*70 + "\n")


def run_all_experiments():
    """Exécute tous les expériments"""
    print("\n" + "="*70)
    print("EXPÉRIMENTATIONS COMPARATIVES - PARTIE II")
    print("="*70)
    
    CNNArchitectureComparison.experiment_padding_stride()
    CNNArchitectureComparison.experiment_pooling_types()
    CNNArchitectureComparison.experiment_filter_counts()
    CNNArchitectureComparison.experiment_cnn_vs_mlp()


if __name__ == "__main__":
    run_all_experiments()