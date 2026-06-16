import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn_manual import (
    ManualConvolution2D, ManualMaxPooling2D, ManualAvgPooling2D,
    ConvolutionDimensionCalculator, demonstrate_all
)
from feature_visualization import (
    FeatureMapVisualizer, FilterVisualization, ConvolutionAnalyzer
)

def main():
    print("="*70)
    print("PARTIE II - CNN ET VISION PAR ORDINATEUR")
    print("Classification d'images avec CNN (MNIST)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    
    # 1. Démonstration des concepts manuels
    print("\n" + "="*70)
    print("1. DÉMONSTRATION DES CONCEPTS FONDAMENTAUX")
    print("="*70)
    
    demonstrate_all()
    
    # 2. Analyse des paramètres
    print("\n" + "="*70)
    print("2. ANALYSE DES PARAMÈTRES CNN")
    print("="*70)
    
    ConvolutionAnalyzer.analyze_padding_stride_effects(input_size=28, kernel_size=3)
    ConvolutionAnalyzer.analyze_filter_count_effects()
    ConvolutionAnalyzer.print_recommendations()
    
    # 3. Charger MNIST
    print("\n" + "="*70)
    print("3. PRÉPARATION DU DATASET MNIST")
    print("="*70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"✅ MNIST chargé")
    print(f"   Train : {len(train_dataset)} images")
    print(f"   Test  : {len(test_dataset)} images")
    print(f"   Taille image : 28x28")
    print(f"   Canaux : 1 (grayscale)")
    
    # 4. Définir une simple CNN
    print("\n" + "="*70)
    print("4. ARCHITECTURE CNN SIMPLE")
    print("="*70)
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 10),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 Paramètres : {total_params:,}")
    print("\nArchitecture :")
    print(model)
    
    # 5. Entraînement simple
    print("\n" + "="*70)
    print("5. ENTRAÎNEMENT (5 EPOCHS)")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / len(train_dataset)
        
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # 6. Évaluation
    print("\n" + "="*70)
    print("6. ÉVALUATION TEST")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_accuracy = correct / total
    print(f"\n✅ Test Accuracy: {test_accuracy:.4f} ({correct}/{total})")
    
    # 7. Visualisation des feature maps
    print("\n" + "="*70)
    print("7. VISUALISATION DES FEATURE MAPS")
    print("="*70)
    
    sample_image, _ = test_dataset[0]
    sample_batch = sample_image.unsqueeze(0).to(device)
    
    visualizer = FeatureMapVisualizer(model, device)
    
    # Enregistrer hooks pour les couches de convolution
    visualizer.register_hook(model.features[0])  # Conv1
    visualizer.register_hook(model.features[3])  # Conv2
    
    # Visualiser
    Path("../results").mkdir(exist_ok=True)
    visualizer.visualize_feature_maps(sample_batch, layer_index=0, save_path="../results/feature_maps_layer1.png")
    visualizer.visualize_feature_maps(sample_batch, layer_index=1, save_path="../results/feature_maps_layer2.png")
    
    visualizer.remove_hooks()
    
    # 8. Visualisation des filtres
    print("\n" + "="*70)
    print("8. VISUALISATION DES FILTRES")
    print("="*70)
    
    FilterVisualization.visualize_filters(model.features[0], save_path="../results/filters_conv1.png")
    FilterVisualization.visualize_filters(model.features[3], save_path="../results/filters_conv2.png")
    
    # 9. Sauvegarder le modèle
    print("\n" + "="*70)
    print("9. SAUVEGARDE DU MODÈLE")
    print("="*70)
    
    Path("../models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "../models/cnn_mnist.pth")
    print("✅ Modèle sauvegardé : ../models/cnn_mnist.pth")
    
    print("\n✅ Partie II (CNN) - Concepts fondamentaux - Terminée !")

if __name__ == "__main__":
    main()