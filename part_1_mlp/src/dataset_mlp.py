import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
from pathlib import Path

DATA_DIR = Path("../data")

class WineQualityDataset:
    """Préparation du dataset Wine"""
    
    def __init__(self, test_size=0.15, val_size=0.1, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def download_and_prepare(self):
        """Charge le dataset Wine"""
        DATA_DIR.mkdir(exist_ok=True)
        
        print("📥 Chargement du Wine dataset (sklearn)...")
        wine = datasets.load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['quality'] = wine.target
        
        print(f"\n📊 Dataset Wine (sklearn)")
        print(f"   Forme : {df.shape}")
        print(f"   Colonnes : {list(df.columns)[:5]}...")
        
        return df
    
    def prepare(self, df):
        """Préparation des données"""
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        
        print(f"\n📈 Distribution des classes:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"   Classe {u}: {c} samples ({100*c/len(y):.1f}%)")
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=self.random_state, stratify=y_train_val
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n📊 Split : Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """Crée les DataLoaders"""
        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        X_test_t = torch.FloatTensor(X_test)
        
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)
        y_test_t = torch.LongTensor(y_test)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader