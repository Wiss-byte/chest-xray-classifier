import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import SimpleCNN


def get_predictions(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).int().cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=32)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    print("Modèle chargé : best_model.pth")

    labels, preds, probs = get_predictions(model, test_loader, device)

    print("\n=== Rapport de classification ===")
    print(classification_report(labels, preds,
                                target_names=["NORMAL", "PNEUMONIA"]))

    print("=== Matrice de confusion ===")
    print(confusion_matrix(labels, preds))

    auc = roc_auc_score(labels, probs)
    print(f"\nAUC-ROC : {auc:.4f}")


if __name__ == "__main__":
    main()