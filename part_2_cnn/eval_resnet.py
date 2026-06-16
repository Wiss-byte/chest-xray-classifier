from src.model_resnet import get_resnet18
from src.evaluate import get_predictions
from src.dataset import get_dataloaders
from sklearn.metrics import classification_report, roc_auc_score
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_dataloaders(batch_size=32)

model = get_resnet18().to(device)
model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))

labels, preds, probs = get_predictions(model, test_loader, device)

print("\n=== RÉSULTATS RESNET18 ===")
print(classification_report(labels, preds, target_names=["NORMAL", "PNEUMONIA"]))
auc = roc_auc_score(labels, probs)
print(f"AUC-ROC : {auc:.4f}")