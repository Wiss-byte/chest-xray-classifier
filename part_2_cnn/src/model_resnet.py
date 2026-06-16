import torch.nn as nn
from torchvision import models


def get_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")
    # Geler toutes les couches sauf la dernière
    for param in model.parameters():
        param.requires_grad = False
    # Remplacer la dernière couche
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    return model


if __name__ == "__main__":
    import torch
    model = get_resnet18()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")