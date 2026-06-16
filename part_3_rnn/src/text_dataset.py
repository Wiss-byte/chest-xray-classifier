import torch
import numpy as np
from collections import Counter
from pathlib import Path
import pickle

class TextDataset:
    """Gère le dataset de texte pour RNN/LSTM"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=50):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    
    def build_vocab(self, texts):
        """Construit le vocabulaire à partir des textes"""
        print("📚 Construction du vocabulaire...")
        
        # Compter les mots
        counter = Counter()
        for text in texts:
            words = text.lower().split()
            counter.update(words)
        
        # Sélectionner les top mots
        most_common = counter.most_common(self.max_vocab_size - 4)
        
        # Tokens spéciaux
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # Créer mappings
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        for idx, (word, _) in enumerate(most_common, start=len(special_tokens)):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"✅ Vocabulaire : {self.vocab_size} mots uniques")
        
        return self.word2idx
    
    def encode_text(self, text):
        """Encode un texte en indices"""
        words = text.lower().split()
        indices = []
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx['<UNK>'])
        
        return indices
    
    def decode_indices(self, indices):
        """Décode des indices en texte"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def pad_sequence(self, sequence, pad_length=None):
        """Ajoute du padding à une séquence"""
        if pad_length is None:
            pad_length = self.max_seq_length
        
        if len(sequence) >= pad_length:
            return sequence[:pad_length]
        else:
            return sequence + [self.word2idx['<PAD>']] * (pad_length - len(sequence))
    
    def create_sequences(self, texts, labels=None):
        """Crée des séquences avec padding"""
        print("📝 Création des séquences...")
        
        sequences = []
        for text in texts:
            encoded = self.encode_text(text)
            padded = self.pad_sequence(encoded)
            sequences.append(padded)
        
        X = torch.LongTensor(sequences)
        
        if labels is not None:
            y = torch.LongTensor(labels)
            return X, y
        else:
            return X
    
    def save_vocab(self, path):
        """Sauvegarde le vocabulaire"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"✅ Vocabulaire sauvegardé : {path}")
    
    def load_vocab(self, path):
        """Charge le vocabulaire"""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = vocab_data['vocab_size']
        
        print(f"✅ Vocabulaire chargé : {path}")


class SentimentDataset:
    """Dataset pour l'analyse de sentiment"""
    
    @staticmethod
    def get_sample_data():
        """Retourne des données d'exemple (sentiment positif/négatif)"""
        texts = [
            "this movie is absolutely amazing and wonderful",
            "i love this film it is fantastic",
            "great acting and excellent plot",
            "worst movie ever made really bad",
            "terrible waste of time very boring",
            "horrible film i hated every minute",
            "beautiful cinematography great performances",
            "boring and predictable very disappointed",
            "outstanding film masterpiece truly",
            "dreadful movie absolutely awful",
        ]
        
        labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1=positif, 0=négatif
        
        return texts, labels
    
    @staticmethod
    def get_extended_data():
        """Données étendues pour meilleur entraînement"""
        positive = [
            "this movie is absolutely amazing and wonderful",
            "i love this film it is fantastic",
            "great acting and excellent plot",
            "beautiful cinematography great performances",
            "outstanding film masterpiece truly",
            "loved every second of this movie",
            "brilliant acting outstanding direction",
            "incredibly well done amazing story",
            "best movie i have seen this year",
            "absolutely brilliant performance",
        ]
        
        negative = [
            "worst movie ever made really bad",
            "terrible waste of time very boring",
            "horrible film i hated every minute",
            "dreadful movie absolutely awful",
            "waste of time bad acting",
            "boring and predictable very disappointed",
            "awful story terrible acting",
            "worst experience ever very bad",
            "unwatchable complete disaster",
            "terrible screenplay horrible movie",
        ]
        
        texts = positive + negative
        labels = [1] * len(positive) + [0] * len(negative)
        
        return texts, labels


class LanguageModelDataset:
    """Dataset pour language modeling (prédiction du prochain token)"""
    
    @staticmethod
    def create_training_pairs(sequences, vocab_size):
        """Crée des paires (input, target) pour language modeling"""
        X = []
        y = []
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                X.append(seq[:i+1])
                y.append(seq[i+1])
        
        return X, y
    
    @staticmethod
    def get_sample_text():
        """Retourne un texte d'exemple"""
        return """
        deep learning is a subset of machine learning.
        machine learning is a subset of artificial intelligence.
        neural networks are inspired by biological neurons.
        recurrent neural networks can process sequences.
        lstm networks solve the vanishing gradient problem.
        transformers use attention mechanisms.
        attention is all you need.
        """


def demonstrate_text_processing():
    """Démontre le traitement de texte"""
    print("\n" + "="*70)
    print("DÉMONSTRATION : TRAITEMENT DE TEXTE")
    print("="*70)
    
    # Créer le dataset
    dataset = TextDataset(max_vocab_size=100, max_seq_length=20)
    
    # Données d'exemple
    texts, labels = SentimentDataset.get_sample_data()
    
    print(f"\n📊 Dataset sentiment :")
    print(f"   {len(texts)} textes")
    print(f"   {sum(labels)} positifs, {len(labels) - sum(labels)} négatifs")
    
    # Construire le vocabulaire
    dataset.build_vocab(texts)
    
    # Encoder un exemple
    example_text = texts[0]
    print(f"\n📝 Texte original : {example_text}")
    
    encoded = dataset.encode_text(example_text)
    print(f"   Encodé : {encoded[:10]}...")
    
    padded = dataset.pad_sequence(encoded)
    print(f"   Avec padding : {padded}")
    
    decoded = dataset.decode_indices(padded)
    print(f"   Décodé : {decoded}")
    
    # Créer les séquences
    X, y = dataset.create_sequences(texts, labels)
    print(f"\n✅ Séquences créées : {X.shape}")
    print(f"   Labels : {y.shape}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demonstrate_text_processing()