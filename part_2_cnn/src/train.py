import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.model_resnet import get_resnet18


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  "):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_model(model, model_name="model", epochs=10, lr=1e-3, pos_weight=1.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Entraînement : {model_name} | Device : {device} ===")

    train_loader, val_loader, _ = get_dataloaders(batch_size=32, augment=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"  --> Meilleur {model_name} sauvegardé !")


def main():
    # CNN simple
    print("\n" + "="*60)
    print("ENTRAÎNEMENT CNN")
    print("="*60)
    cnn = SimpleCNN()
    train_model(cnn, model_name="cnn", epochs=10, lr=1e-3, pos_weight=1.5)
    
    # ResNet18 fine-tuning
    print("\n" + "="*60)
    print("ENTRAÎNEMENT RESNET18")
    print("="*60)
    resnet = get_resnet18()
    train_model(resnet, model_name="resnet18", epochs=10, lr=1e-4, pos_weight=1.5)


if __name__ == "__main__":
    main()