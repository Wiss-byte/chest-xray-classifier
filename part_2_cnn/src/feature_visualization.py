import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class FeatureMapVisualizer:
    """Visualise les cartes de caractéristiques (feature maps)"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.feature_maps = []
        self.hooks = []
    
    def register_hook(self, layer):
        """Enregistre un hook pour capturer les activations"""
        def hook(module, input, output):
            self.feature_maps.append(output.detach().cpu())
        
        handle = layer.register_forward_hook(hook)
        self.hooks.append(handle)
    
    def remove_hooks(self):
        """Supprime tous les hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_feature_maps(self, x, layer_index=0, num_filters=9, save_path=None):
        """
        Visualise les feature maps d'une couche
        
        Args:
            x : image d'entrée
            layer_index : index de la couche à visualiser
            num_filters : nombre de filtres à afficher
            save_path : chemin pour sauvegarder
        """
        self.feature_maps = []
        
        # Capturer les feature maps
        with torch.no_grad():
            _ = self.model(x)
        
        if layer_index >= len(self.feature_maps):
            print(f"⚠️  Layer index {layer_index} dépasse le nombre de couches")
            return
        
        feature_map = self.feature_maps[layer_index][0]  # Batch 0
        num_channels = min(num_filters, feature_map.shape[0])
        
        # Créer la grille de visualisation
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(9, num_channels)):
            fm = feature_map[i].numpy()
            axes[i].imshow(fm, cmap='viridis')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
        
        # Masquer les subplots inutilisés
        for i in range(num_channels, 9):
            axes[i].axis('off')
        
        plt.suptitle(f'Feature Maps - Layer {layer_index}')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Feature maps sauvegardées : {save_path}")
        
        plt.close()
    
    def compare_pooling_effects(self, x, model_max, model_avg, save_path=None):
        """Compare les effets du max-pooling vs avg-pooling"""
        with torch.no_grad():
            out_max = model_max(x)
            out_avg = model_avg(x)
        
        # Visualiser les résultats
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        out_max_np = out_max[0, 0].cpu().numpy()
        out_avg_np = out_avg[0, 0].cpu().numpy()
        
        axes[0].imshow(out_max_np, cmap='hot')
        axes[0].set_title('Max-Pooling')
        axes[0].axis('off')
        
        axes[1].imshow(out_avg_np, cmap='hot')
        axes[1].set_title('Average-Pooling')
        axes[1].axis('off')
        
        plt.suptitle('Comparaison Max-Pooling vs Average-Pooling')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Comparaison sauvegardée : {save_path}")
        
        plt.close()


class FilterVisualization:
    """Visualise les filtres (kernels) d'une couche de convolution"""
    
    @staticmethod
    def visualize_filters(conv_layer, num_filters=16, save_path=None):
        """Affiche les filtres d'une couche de convolution"""
        weights = conv_layer.weight.data.cpu().numpy()
        
        # Normaliser pour visualisation
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        num_to_show = min(num_filters, weights.shape[0])
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(16):
            if i < num_to_show:
                # Moyenne sur les canaux d'entrée
                kernel = weights[i].mean(axis=0)
                axes[i].imshow(kernel, cmap='gray')
                axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
        
        plt.suptitle('Visualisation des Filtres de Convolution')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Filtres sauvegardés : {save_path}")
        
        plt.close()


class ConvolutionAnalyzer:
    """Analyse les effets des paramètres de convolution"""
    
    @staticmethod
    def analyze_padding_stride_effects(input_size=224, kernel_size=3):
        """Analyse l'impact du padding et stride"""
        print("\n" + "="*70)
        print("ANALYSE : IMPACT DU PADDING ET STRIDE")
        print("="*70)
        
        print(f"\nInput size: {input_size}x{input_size}, Kernel: {kernel_size}x{kernel_size}\n")
        print(f"{'Padding':<10} {'Stride':<10} {'Output Size':<20} {'Réduction':<15}")
        print("-" * 60)
        
        for padding in [0, 1, 2]:
            for stride in [1, 2, 3]:
                output_size = ((input_size + 2*padding - kernel_size) // stride) + 1
                reduction = (input_size - output_size) / input_size * 100
                print(f"{padding:<10} {stride:<10} {output_size}x{output_size:<17} {reduction:.1f}%")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def analyze_filter_count_effects():
        """Analyse l'impact du nombre de filtres"""
        print("\n" + "="*70)
        print("ANALYSE : IMPACT DU NOMBRE DE FILTRES")
        print("="*70)
        
        print(f"\n{'Layer':<15} {'Input Channels':<20} {'Filters':<15} {'Parameters':<20}")
        print("-" * 70)
        
        layers = [
            ("Conv1", 3, 16),
            ("Conv2", 16, 32),
            ("Conv3", 32, 64),
            ("Conv4", 64, 128),
        ]
        
        for name, in_channels, filters in layers:
            # Kernel 3x3
            params = in_channels * filters * 3 * 3 + filters  # +filters pour les bias
            print(f"{name:<15} {in_channels:<20} {filters:<15} {params:,}")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def print_recommendations():
        """Affiche les recommandations pour les paramètres"""
        print("\n" + "="*70)
        print("RECOMMANDATIONS POUR LES PARAMÈTRES CNN")
        print("="*70)
        
        recommendations = {
            "Padding": [
                "• padding=0 : réduit la taille, perd info aux bords",
                "• padding=1 : préserve la taille avec kernel 3x3",
                "• padding=2 : augmente la taille, utile pour détails fins"
            ],
            "Stride": [
                "• stride=1 : préserve la résolution, calcul intensif",
                "• stride=2 : réduit par 2, plus rapide, perte d'info",
                "• stride=3+ : réduction drastique, risque de sous-sampling"
            ],
            "Pooling": [
                "• Max-pooling : conserve les features fortes, bruitées",
                "• Avg-pooling : lisse, perd détails locaux",
                "• 2x2 avec stride=2 : standard efficace"
            ],
            "Filtres": [
                "• Nombre de filtres augmente la capacité du modèle",
                "• Doubler les filtres ≈ 8x plus de paramètres",
                "• Équilibre entre expressivité et surapprentissage"
            ]
        }
        
        for category, tips in recommendations.items():
            print(f"\n📌 {category}:")
            for tip in tips:
                print(f"   {tip}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    ConvolutionAnalyzer.analyze_padding_stride_effects()
    ConvolutionAnalyzer.analyze_filter_count_effects()
    ConvolutionAnalyzer.print_recommendations()