import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Seq2SeqEncoder(nn.Module):
    """Encodeur pour architecture Seq2Seq"""
    
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Seq2SeqDecoder(nn.Module):
    """Décodeur pour architecture Seq2Seq"""
    
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """Architecture complète Seq2Seq"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        decoder_input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = trg[:, t] if teacher_force else top1
        
        return outputs


class BeamSearchDecoder:
    """Implémente le Beam Search pour décoder les séquences"""
    
    def __init__(self, model, vocab_size, max_length=20, beam_width=3, device='cpu'):
        self.model = model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.beam_width = beam_width
        self.device = device
    
    def decode(self, src, start_token, end_token):
        """
        Decode une séquence source avec beam search
        
        Args:
            src : séquence d'entrée
            start_token : token de démarrage
            end_token : token de fin
        
        Returns:
            best_sequence : meilleure séquence trouvée
            best_score : score de la meilleure séquence
        """
        with torch.no_grad():
            hidden, cell = self.model.encoder(src)
        
        # Initialiser les hypothèses (séquence, score, hidden, cell)
        hypotheses = [
            {
                'sequence': [start_token],
                'score': 0.0,
                'hidden': hidden,
                'cell': cell
            }
        ]
        
        for step in range(self.max_length):
            all_candidates = []
            
            for hypothesis in hypotheses:
                if hypothesis['sequence'][-1] == end_token:
                    all_candidates.append(hypothesis)
                    continue
                
                # Décoder un token
                with torch.no_grad():
                    decoder_input = torch.tensor([hypothesis['sequence'][-1]]).to(self.device)
                    output, hidden, cell = self.model.decoder(
                        decoder_input, 
                        hypothesis['hidden'], 
                        hypothesis['cell']
                    )
                
                # Top-k tokens
                log_probs = torch.log_softmax(output, dim=1)[0]
                top_k_probs, top_k_indices = torch.topk(log_probs, self.beam_width)
                
                for prob, idx in zip(top_k_probs, top_k_indices):
                    candidate = {
                        'sequence': hypothesis['sequence'] + [idx.item()],
                        'score': hypothesis['score'] + prob.item(),
                        'hidden': hidden,
                        'cell': cell
                    }
                    all_candidates.append(candidate)
            
            # Garder les top-k hypothèses
            hypotheses = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:self.beam_width]
            
            # Vérifier si tous les chemins sont terminés
            if all(h['sequence'][-1] == end_token for h in hypotheses):
                break
        
        # Retourner la meilleure hypothèse
        best = hypotheses[0]
        return best['sequence'], best['score']
    
    @staticmethod
    def compare_with_greedy(model, src, start_token, end_token, device='cpu'):
        """Compare Beam Search avec Greedy Decoding"""
        print("\n" + "="*70)
        print("COMPARAISON : BEAM SEARCH vs GREEDY DECODING")
        print("="*70)
        
        # Greedy
        greedy_seq = []
        with torch.no_grad():
            hidden, cell = model.encoder(src)
            decoder_input = torch.tensor([start_token]).to(device)
            
            for _ in range(20):
                output, hidden, cell = model.decoder(decoder_input, hidden, cell)
                top_token = output.argmax(1)
                greedy_seq.append(top_token.item())
                decoder_input = top_token
                
                if top_token.item() == end_token:
                    break
        
        # Beam Search
        beam_decoder = BeamSearchDecoder(model, vocab_size=100, beam_width=3, device=device)
        beam_seq, beam_score = beam_decoder.decode(src, start_token, end_token)
        
        print(f"\n🎯 Greedy Sequence: {greedy_seq}")
        print(f"   Longueur: {len(greedy_seq)}")
        
        print(f"\n🔍 Beam Search Sequence: {beam_seq}")
        print(f"   Score: {beam_score:.4f}")
        print(f"   Longueur: {len(beam_seq)}")
        
        print("\n📊 Avantages de Beam Search:")
        print("   ✓ Explore plusieurs chemins")
        print("   ✓ Moins de risque de se bloquer localement")
        print("   ✓ Meilleure qualité globale")
        print("   ✗ Plus lent computationnellement")
        
        print("\n" + "="*70 + "\n")


class Seq2SeqTrainer:
    """Entraîne un modèle Seq2Seq"""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """Une époque d'entraînement"""
        self.model.train()
        total_loss = 0
        
        for src, trg in train_loader:
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            outputs = self.model(src, trg, teacher_forcing_ratio=0.5)
            
            # Reshape pour loss
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs, trg)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader, val_loader, epochs=10):
        """Entraînement complet"""
        print("\n" + "="*70)
        print("ENTRAÎNEMENT SEQ2SEQ")
        print("="*70)
        
        print(f"\n{'Epoch':<6} {'Train Loss':<15} {'Val Loss':<15}")
        print("-" * 40)
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            print(f"{epoch:<6} {train_loss:<15.4f}")
        
        print("\n✅ Entraînement terminé\n")
        print("=" * 70 + "\n")


def demonstrate_seq2seq():
    """Démontre Seq2Seq"""
    print("\n" + "="*70)
    print("DÉMONSTRATION : SEQ2SEQ AVEC BEAM SEARCH")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Hyperparamètres
    input_dim = 100
    output_dim = 100
    emb_dim = 128
    hidden_dim = 256
    n_layers = 1
    dropout = 0.3
    
    # Créer le modèle
    encoder = Seq2SeqEncoder(input_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)
    decoder = Seq2SeqDecoder(output_dim, emb_dim, hidden_dim, n_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📋 Paramètres Seq2Seq : {total_params:,}")
    
    # Données de test
    src = torch.randint(1, input_dim, (1, 10)).to(device)
    trg = torch.randint(1, output_dim, (1, 10)).to(device)
    
    # Forward pass
    outputs = model(src, trg)
    print(f"\nInput shape: {src.shape}")
    print(f"Target shape: {trg.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # Démonstration Beam Search
    print("\n" + "="*70)
    print("BEAM SEARCH")
    print("="*70)
    
    beam_decoder = BeamSearchDecoder(model, vocab_size=output_dim, beam_width=3, device=device)
    sequence, score = beam_decoder.decode(src, start_token=1, end_token=2)
    
    print(f"\n✅ Séquence générée : {sequence}")
    print(f"   Score: {score:.4f}")
    print(f"   Longueur: {len(sequence)}")
    
    # Comparaison
    BeamSearchDecoder.compare_with_greedy(model, src, start_token=1, end_token=2, device=device)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demonstrate_seq2seq()