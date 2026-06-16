from src.model_resnet import get_resnet18
from src.train import train_model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENTRAÎNEMENT RESNET18 FINE-TUNING")
    print("="*60)
    resnet = get_resnet18()
    train_model(resnet, model_name="resnet18", epochs=10, lr=1e-4, pos_weight=1.5)