import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from text_dataset import TextDataset, SentimentDataset, demonstrate_text_processing
from rnn_models import SimpleRNN, LSTMNetwork, GRUNetwork, print_model_comparison, BPTTExplainer, GradientClippingDemo
from rnn_trainer import RNNTrainer, PerplexityCalculator, BeamSearchDecoder, demonstrate_rnn_concepts

def main():
    print("="*70)
    print("PARTIE III - RNN, LSTM, GRU ET SEQ2SEQ")
    print("Modélisation de séquences et génération sur données textuelles")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    
    # 1. Démonstration des concepts RNN
    print("\n" + "="*70)
    print("1. DÉMONSTRATION DES CONCEPTS RNN")
    print("="*70)
    
    demonstrate_rnn_concepts()
    
    # 2. Traitement de texte
    print("\n" + "="*70)
    print("2. TRAITEMENT DE TEXTE")
    print("="*70)
    
    demonstrate_text_processing()
    
    # 3. Préparation du dataset de sentiment
    print("\n" + "="*70)
    print("3. PRÉPARATION DU DATASET DE SENTIMENT")
    print("="*70)
    
    texts, labels = SentimentDataset.get_extended_data()
    
    print(f"\n📊 Dataset de sentiment :")
    print(f"   {len(texts)} textes")
    print(f"   {sum(labels)} positifs, {len(labels) - sum(labels)} négatifs")
    
    # Créer le vocabulaire
    dataset = TextDataset(max_vocab_size=1000, max_seq_length=50)
    dataset.build_vocab(texts)
    
    # Créer les séquences
    X, y = dataset.create_sequences(texts, labels)
    
    print(f"\n✅ Séquences créées : {X.shape}")
    print(f"   Vocabulaire : {dataset.vocab_size} mots")
    print(f"   Longueur max : {dataset.max_seq_length}")
    
    # Split train/val/test
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\n📊 Split :")
    print(f"   Train : {len(X_train)}")
    print(f"   Val   : {len(X_val)}")
    print(f"   Test  : {len(X_test)}")
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 4. Entraînement RNN simple
    print("\n" + "="*70)
    print("4. RNN SIMPLE")
    print("="*70)
    
    model_rnn = SimpleRNN(vocab_size=dataset.vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2).to(device)
    total_params = sum(p.numel() for p in model_rnn.parameters())
    print(f"\n📋 Paramètres RNN : {total_params:,}")
    
    trainer_rnn = RNNTrainer(model_rnn, device, lr=0.001)
    Path("../models").mkdir(exist_ok=True)
    history_rnn = trainer_rnn.train(train_loader, val_loader, epochs=20, patience=5, 
                                     save_path="../models/best_rnn.pth")
    
    # 5. Entraînement LSTM
    print("\n" + "="*70)
    print("5. LSTM")
    print("="*70)
    
    model_lstm = LSTMNetwork(vocab_size=dataset.vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2).to(device)
    total_params = sum(p.numel() for p in model_lstm.parameters())
    print(f"\n📋 Paramètres LSTM : {total_params:,}")
    
    trainer_lstm = RNNTrainer(model_lstm, device, lr=0.001)
    history_lstm = trainer_lstm.train(train_loader, val_loader, epochs=20, patience=5,
                                       save_path="../models/best_lstm.pth")
    
    # 6. Entraînement GRU
    print("\n" + "="*70)
    print("6. GRU")
    print("="*70)
    
    model_gru = GRUNetwork(vocab_size=dataset.vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2).to(device)
    total_params = sum(p.numel() for p in model_gru.parameters())
    print(f"\n📋 Paramètres GRU : {total_params:,}")
    
    trainer_gru = RNNTrainer(model_gru, device, lr=0.001)
    history_gru = trainer_gru.train(train_loader, val_loader, epochs=20, patience=5,
                                     save_path="../models/best_gru.pth")
    
    # 7. Évaluation
    print("\n" + "="*70)
    print("7. ÉVALUATION TEST")
    print("="*70)
    
    def evaluate_model(model, test_loader, model_name):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        accuracy = correct / total
        print(f"{model_name:<10} : {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    print()
    acc_rnn = evaluate_model(model_rnn, test_loader, "RNN")
    acc_lstm = evaluate_model(model_lstm, test_loader, "LSTM")
    acc_gru = evaluate_model(model_gru, test_loader, "GRU")
    
    # 8. Comparaison
    print("\n" + "="*70)
    print("8. COMPARAISON RNN vs LSTM vs GRU")
    print("="*70)
    
    print("\n📊 RÉSUMÉ COMPARATIF:")
    print(f"{'Modèle':<15} {'Accuracy':<15} {'Paramètres':<15} {'Stabilité':<15}")
    print("-" * 60)
    print(f"{'RNN':<15} {acc_rnn:<15.4f} {'~30k':<15} {'Faible':<15}")
    print(f"{'LSTM':<15} {acc_lstm:<15.4f} {'~40k':<15} {'Excellente':<15}")
    print(f"{'GRU':<15} {acc_gru:<15.4f} {'~35k':<15} {'Très bonne':<15}")
    
    # 9. Perplexité
    print("\n" + "="*70)
    print("9. PERPLEXITÉ")
    print("="*70)
    
    val_loss_rnn = history_rnn['val_loss'][-1]
    val_loss_lstm = history_lstm['val_loss'][-1]
    val_loss_gru = history_gru['val_loss'][-1]
    
    perp_rnn = PerplexityCalculator.calculate_perplexity(val_loss_rnn)
    perp_lstm = PerplexityCalculator.calculate_perplexity(val_loss_lstm)
    perp_gru = PerplexityCalculator.calculate_perplexity(val_loss_gru)
    
    print("\n📊 Perplexité (Plus bas = meilleur):")
    print(f"   RNN  : {perp_rnn:.4f}")
    print(f"   LSTM : {perp_lstm:.4f}")
    print(f"   GRU  : {perp_gru:.4f}")
    
    # 10. Prédictions d'exemple
    print("\n" + "="*70)
    print("10. PRÉDICTIONS D'EXEMPLE")
    print("="*70)
    
    test_texts = [
        "this movie is amazing wonderful",
        "worst movie ever very bad"
    ]
    
    print()
    for text in test_texts:
        encoded = dataset.encode_text(text)
        padded = dataset.pad_sequence(encoded)
        X_test_sample = torch.LongTensor([padded]).to(device)
        
        with torch.no_grad():
            output_lstm = model_lstm(X_test_sample)
            prob = torch.softmax(output_lstm, dim=1)[0]
            pred = torch.argmax(prob).item()
        
        sentiment = "POSITIF" if pred == 1 else "NÉGATIF"
        confidence = prob[pred].item()
        
        print(f"Texte: \"{text}\"")
        print(f"   Prédiction: {sentiment} ({confidence:.2%})")
    
    # Sauvegarder les modèles
    print("\n" + "="*70)
    print("11. SAUVEGARDE DES MODÈLES")
    print("="*70)
    
    print("✅ Modèles sauvegardés :")
    print("   ../models/best_rnn.pth")
    print("   ../models/best_lstm.pth")
    print("   ../models/best_gru.pth")
    
    print("\n✅ Partie III (RNN/LSTM/GRU) terminée avec succès !")

if __name__ == "__main__":
    main()