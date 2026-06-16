import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
sys.path.insert(0, '.')

from src.model_resnet import get_resnet18


def gradcam_simple(image_path):
    """Grad-CAM simplifié sans hooks"""
    device = torch.device("cpu")
    
    # Modèle
    model = get_resnet18().to(device)
    model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))
    model.eval()
    
    # Image
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = tfm(img_pil).unsqueeze(0).to(device)
    
    # Forward avec requires_grad
    img_tensor.requires_grad = True
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()
    
    # Backward
    output.backward()
    
    # Gradient de l'input
    gradients = img_tensor.grad[0]  # [3, 224, 224]
    gradients = gradients.abs().mean(dim=0)  # [224, 224]
    
    # Normaliser
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
    
    pred_class = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    
    # Visualiser
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_pil)
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    
    axes[1].imshow(gradients.cpu().numpy(), cmap="hot")
    axes[1].set_title("Gradient (attention)")
    axes[1].axis("off")
    
    axes[2].imshow(img_pil)
    axes[2].imshow(gradients.cpu().numpy(), cmap="hot", alpha=0.6)
    axes[2].set_title(f"Prédiction: {pred_class} ({prob:.1%})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=100, bbox_inches="tight")
    print("✅ Sauvegardé : gradcam_result.png")
    plt.show()


if __name__ == "__main__":
    gradcam_simple("data/data/chest_xray/test/PNEUMONIA/person1_virus_8.jpeg")