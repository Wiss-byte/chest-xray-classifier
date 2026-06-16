import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    """RNN simple pour traitement de séquences"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=1, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, 
                         dropout=dropout if n_layers > 1 else 0,
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        
        # Prendre la dernière sortie
        output = output[:, -1, :]
        output = self.fc(output)
        
        return output


class LSTMNetwork(nn.Module):
    """LSTM pour traitement de séquences"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=1, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        # Prendre la dernière sortie
        output = output[:, -1, :]
        output = self.fc(output)
        
        return output


class GRUNetwork(nn.Module):
    """GRU pour traitement de séquences"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2, n_layers=1, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers,
                         dropout=dropout if n_layers > 1 else 0,
                         batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths=None):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        
        # Prendre la dernière sortie
        output = output[:, -1, :]
        output = self.fc(output)
        
        return output


class Seq2SeqEncoder(nn.Module):
    """Encodeur pour architecture Seq2Seq"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Seq2SeqDecoder(nn.Module):
    """Décodeur pour architecture Seq2Seq"""
    
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Architecture complète Seq2Seq"""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        
        hidden, cell = self.encoder(src)
        
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = trg[:, t] if teacher_force else top1
        
        return outputs


class GradientClippingDemo:
    """Démontre le gradient clipping"""
    
    @staticmethod
    def demonstrate():
        print("\n" + "="*70)
        print("DÉMONSTRATION : GRADIENT CLIPPING")
        print("="*70)
        
        print("\n📌 Problème du vanishing/exploding gradient :")
        print("   • Vanishing: gradients → 0, apprentissage stagne")
        print("   • Exploding: gradients → ∞, instabilité numérique")
        
        print("\n🔧 Solutions :")
        print("   1. Gradient Clipping par norme (clip_grad_norm_)")
        print("   2. Gradient Clipping par valeur (clip_grad_value_)")
        print("   3. Initialisation He/Xavier")
        print("   4. Batch Normalization")
        
        print("\n💡 Exemple avec PyTorch :")
        print("""
    # Clip par norme
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Clip par valeur
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        """)
        
        print("\n" + "="*70 + "\n")


class BPTTExplainer:
    """Explique la Backpropagation Through Time"""
    
    @staticmethod
    def explain():
        print("\n" + "="*70)
        print("EXPLICATION : BACKPROPAGATION THROUGH TIME (BPTT)")
        print("="*70)
        
        print("\n📚 Concept :")
        print("   BPTT = Backpropagation appliquée à travers les timesteps")
        
        print("\n🔄 Processus :")
        print("   1. Forward pass : calculer sortie à chaque timestep")
        print("   2. Calculer la loss totale")
        print("   3. Backward pass : propager l'erreur en arrière dans le temps")
        print("   4. Accumulation des gradients à travers les timesteps")
        
        print("\n⚠️  Problèmes :")
        print("   • Vanishing gradient : ∂L/∂h diminue exponentiellement")
        print("   • Exploding gradient : ∂L/∂h augmente exponentiellement")
        print("   • Difficile d'apprendre les dépendances à long terme")
        
        print("\n✅ Solutions :")
        print("   • LSTM : gate mechanisms pour contrôler l'information")
        print("   • GRU : version simplifiée du LSTM")
        print("   • Gradient clipping : limiter la norme des gradients")
        
        print("\n" + "="*70 + "\n")


def print_model_comparison():
    """Compare les architectures RNN, LSTM et GRU"""
    print("\n" + "="*70)
    print("COMPARAISON : RNN vs LSTM vs GRU")
    print("="*70)
    
    comparison = {
        "RNN": {
            "Gating": "Non",
            "Mémoire": "Courte seulement",
            "Gradient": "Vanishing/Exploding",
            "Paramètres": "3h²",
            "Vitesse": "Rapide",
            "Performance": "Faible pour LT"
        },
        "LSTM": {
            "Gating": "Oui (3 gates)",
            "Mémoire": "Court + Long terme",
            "Gradient": "Contrôlé",
            "Paramètres": "4h² (plus lourd)",
            "Vitesse": "Lente",
            "Performance": "Excellente"
        },
        "GRU": {
            "Gating": "Oui (2 gates)",
            "Mémoire": "Court + Long terme",
            "Gradient": "Contrôlé",
            "Paramètres": "3h² (intermédiaire)",
            "Vitesse": "Intermédiaire",
            "Performance": "Très bonne"
        }
    }
    
    print(f"\n{'Propriété':<20} {'RNN':<20} {'LSTM':<20} {'GRU':<20}")
    print("-" * 80)
    
    for prop in comparison["RNN"].keys():
        rnn_val = comparison["RNN"][prop]
        lstm_val = comparison["LSTM"][prop]
        gru_val = comparison["GRU"][prop]
        print(f"{prop:<20} {rnn_val:<20} {lstm_val:<20} {gru_val:<20}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print_model_comparison()
    BPTTExplainer.explain()
    GradientClippingDemo.demonstrate()