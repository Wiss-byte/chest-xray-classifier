import sys
import torch
from pathlib import Path

from dataset_mlp import WineQualityDataset
from models_mlp import MLPSequential, MLPCustom, InitializationTester
from trainer_mlp import MLPTrainer
from evaluator_mlp import MLPEvaluator

def main():
    print("="*70)
    print("PARTIE I - MLP ET INGÉNIERIE PYTORCH")
    print("Classification supervisée sur données tabulaires (Wine)")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device : {device}")
    
    # 1. Préparation des données
    print("\n" + "="*70)
    print("1. PRÉPARATION DES DONNÉES")
    print("="*70)
    
    dataset_manager = WineQualityDataset()
    df = dataset_manager.download_and_prepare()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_manager.prepare(df)
    
    batch_size = 32
    train_loader, val_loader, test_loader = dataset_manager.get_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=batch_size
    )
    
    input_size = X_train.shape[1]
    print(f"\n✅ Taille d'entrée : {input_size}")
    
    # 2. Test des stratégies d'initialisation
    print("\n" + "="*70)
    print("2. TEST DES STRATÉGIES D'INITIALISATION")
    print("="*70)
    
    dummy_model = MLPCustom(input_size=input_size, output_size=3)
    InitializationTester.print_init_comparison(dummy_model, device)
    
    # 3. MLP avec nn.Sequential
    print("="*70)
    print("3. MLP AVEC nn.Sequential")
    print("="*70)
    
    model_seq = MLPSequential(input_size=input_size, hidden_sizes=[128, 64], output_size=3).to(device)
    print("\n📋 Paramètres du modèle Sequential:")
    total_params = sum(p.numel() for p in model_seq.parameters())
    print(f"   Total : {total_params:,} paramètres")
    
    trainer_seq = MLPTrainer(model_seq, device, lr=0.001)
    Path("../models").mkdir(exist_ok=True)
    history_seq = trainer_seq.train(train_loader, val_loader, epochs=50, patience=10, 
                                     save_path="../models/best_mlp_sequential.pth")
    
    # 4. MLP avec classe personnalisée
    print("\n" + "="*70)
    print("4. MLP AVEC CLASSE PERSONNALISÉE")
    print("="*70)
    
    model_custom = MLPCustom(input_size=input_size, hidden_sizes=[128, 64], output_size=3).to(device)
    model_custom.get_parameters_info()
    
    model_custom = InitializationTester.xavier_init(model_custom)
    
    trainer_custom = MLPTrainer(model_custom, device, lr=0.001)
    history_custom = trainer_custom.train(train_loader, val_loader, epochs=50, patience=10,
                                          save_path="../models/best_mlp_custom.pth")
    
    # 5. Inspection du modèle (state_dict)
    print("\n" + "="*70)
    print("5. INSPECTION DU MODÈLE (state_dict)")
    print("="*70)
    
    print("\n📋 Named Parameters du modèle personnalisé:")
    for name, param in model_custom.named_parameters():
        print(f"   {name:30s} : {param.shape} = {param.numel():,} params")
    
    state_dict = model_custom.state_dict()
    print(f"\n📋 Clés du state_dict : {list(state_dict.keys())}")
    
    # 6. Sauvegarde et rechargement
    print("\n" + "="*70)
    print("6. SAUVEGARDE ET RECHARGEMENT")
    print("="*70)
    
    trainer_custom.save_checkpoint("../models/checkpoint_mlp.pt")
    
    model_loaded = MLPCustom(input_size=input_size, output_size=3).to(device)
    trainer_loaded = MLPTrainer(model_loaded, device)
    trainer_loaded.load_checkpoint("../models/checkpoint_mlp.pt", device)
    print("✅ Modèle rechargé avec succès")
    
    # 7. Évaluation
    print("\n" + "="*70)
    print("7. ÉVALUATION SUR LE TEST SET")
    print("="*70)
    
    evaluator = MLPEvaluator(model_custom, device)
    results = evaluator.evaluate(test_loader)
    
    # 8. Visualisations
    print("\n" + "="*70)
    print("8. VISUALISATIONS")
    print("="*70)
    
    Path("../results").mkdir(exist_ok=True)
    
    evaluator.plot_training_history(history_custom, save_path="../results/training_history.png")
    evaluator.plot_confusion_matrix(results['labels'], results['preds'], 
                                     save_path="../results/confusion_matrix.png")
    evaluator.plot_roc_curve(results['labels'], results['probs'], 
                             save_path="../results/roc_curve.png")
    
    # 9. Comparaison
    print("\n" + "="*70)
    print("9. COMPARAISON SEQUENTIAL vs PERSONNALISÉ")
    print("="*70)
    
    evaluator_seq = MLPEvaluator(model_seq, device)
    results_seq = evaluator_seq.evaluate(test_loader)
    
    print("\n📊 RÉSUMÉ COMPARATIF:")
    print(f"{'Métrique':<15} {'Sequential':<15} {'Personnalisé':<15}")
    print("-" * 45)
    print(f"{'Accuracy':<15} {results_seq['accuracy']:<15.4f} {results['accuracy']:<15.4f}")
    print(f"{'Precision':<15} {results_seq['precision']:<15.4f} {results['precision']:<15.4f}")
    print(f"{'Recall':<15} {results_seq['recall']:<15.4f} {results['recall']:<15.4f}")
    print(f"{'F1-score':<15} {results_seq['f1']:<15.4f} {results['f1']:<15.4f}")
    print(f"{'AUC-ROC':<15} {results_seq['auc']:<15.4f} {results['auc']:<15.4f}")
    
    print("\n✅ Partie I (MLP) terminée avec succès !")

if __name__ == "__main__":
    main()