import torch
from PIL import Image
from torchvision import transforms
from src.model import SimpleCNN

def predict(image_path, model_path="best_cnn.pth"):
    """Prédit si une radio contient une pneumonie"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le modèle
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Préparer l'image
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = tfm(img).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()
    
    return {
        "NORMAL": 1 - prob,
        "PNEUMONIA": prob,
        "prediction": "PNEUMONIA" if prob > 0.5 else "NORMAL"
    }

if __name__ == "__main__":
    # Exemple d'usage
    result = predict("path/to/image.jpg")
    print(f"Prédiction : {result['prediction']}")
    print(f"Confiance PNEUMONIA : {result['PNEUMONIA']:.2%}")