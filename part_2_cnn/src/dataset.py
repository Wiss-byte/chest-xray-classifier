import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

DATA_DIR = Path("data/data/chest_xray")


def get_transforms(augment=False):
    if augment:
        train_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        train_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm


def get_dataloaders(batch_size=32, augment=False):
    train_tfm, val_tfm = get_transforms(augment=augment)

    # Chargement complet du train
    full_train = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfm)
    test_ds    = datasets.ImageFolder(DATA_DIR / "test",  transform=val_tfm)

    # Split 90/10 sur le train → nouveau val set
    n_total = len(full_train)
    n_val   = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Classes : {full_train.classes}")
    print(f"Train   : {len(train_ds)} images")
    print(f"Val     : {len(val_ds)} images")
    print(f"Test    : {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader