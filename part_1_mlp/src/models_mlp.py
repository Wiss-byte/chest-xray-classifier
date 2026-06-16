import torch
import torch.nn as nn
import numpy as np

class MLPSequential(nn.Module):
    """MLP implémenté avec nn.Sequential"""
    
    def __init__(self, input_size=11, hidden_sizes=[128, 64], output_size=2, dropout=0.3):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class MLPCustom(nn.Module):
    """MLP implémenté avec classe personnalisée"""
    
    def __init__(self, input_size=11, hidden_sizes=[128, 64], output_size=2, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x)
        return x
    
    def get_parameters_info(self):
        """Affiche les paramètres"""
        print("\n📋 Paramètres du modèle MLPCustom:")
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"   {name:30s} : {param.shape} = {num_params:,} params")
        print(f"\n   Total : {total_params:,} paramètres\n")
        return total_params


class InitializationTester:
    """Test différentes stratégies d'initialisation"""
    
    @staticmethod
    def gaussian_init(model, std=0.01):
        """Initialisation gaussienne"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.normal_(param, mean=0.0, std=std)
            else:
                nn.init.constant_(param, 0.0)
        return model
    
    @staticmethod
    def xavier_init(model):
        """Initialisation Xavier"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)
        return model
    
    @staticmethod
    def constant_init(model, value=0.01):
        """Initialisation constante"""
        for param in model.parameters():
            nn.init.constant_(param, value)
        return model
    
    @staticmethod
    def he_init(model):
        """Initialisation He (Kaiming)"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.constant_(param, 0.0)
        return model
    
    @staticmethod
    def print_init_comparison(model, device):
        """Compare différentes initialisations"""
        print("\n" + "="*60)
        print("COMPARAISON DES STRATÉGIES D'INITIALISATION")
        print("="*60)
        
        strategies = {
            'Gaussienne (std=0.01)': lambda m: InitializationTester.gaussian_init(m, std=0.01),
            'Xavier (Glorot)': lambda m: InitializationTester.xavier_init(m),
            'Constante (0.01)': lambda m: InitializationTester.constant_init(m, value=0.01),
            'He (Kaiming)': lambda m: InitializationTester.he_init(m),
        }
        
        for strategy_name, init_fn in strategies.items():
            test_model = MLPCustom().to(device)
            init_fn(test_model)
            
            first_layer = list(test_model.parameters())[0]
            mean = first_layer.data.mean().item()
            std = first_layer.data.std().item()
            min_val = first_layer.data.min().item()
            max_val = first_layer.data.max().item()
            
            print(f"\n{strategy_name}:")
            print(f"   Moyenne : {mean:+.6f}")
            print(f"   Écart-type : {std:.6f}")
            print(f"   Min/Max : [{min_val:+.6f}, {max_val:+.6f}]")
        
        print("\n" + "="*60 + "\n")