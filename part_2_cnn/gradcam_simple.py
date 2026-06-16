import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
sys.path.insert(0, '.')

from src.model_resnet import get_resnet18


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_backward_hook(self._save_gradient)
        self.target_layer.register_forward_hook(self._save_activation)
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(self, input_tensor):
        """Génère la heatmap Grad-CAM"""
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        score = output[0, 0]
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2))
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def visualize(image_path):
    device = torch.device("cpu")
    
    # Charger modèle
    model = get_resnet18().to(device)
    model.load_state_dict(torch.load("best_resnet18.pth", map_location=device))
    
    # GradCAM
    gradcam = GradCAM(model, model.layer4[-1])
    
    # Image
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = tfm(img_pil).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    
    pred_class = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    
    # CAM
    cam = gradcam.generate_cam(img_tensor)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(cam, cmap="hot")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    
    axes[2].imshow(img_pil)
    axes[2].imshow(cam, cmap="hot", alpha=0.5)
    axes[2].set_title(f"Pred: {pred_class} ({prob:.1%})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=100, bbox_inches="tight")
    print("✅ Sauvegardé : gradcam_result.png")
    plt.close()


if __name__ == "__main__":
    # Test PNEUMONIA
    visualize("data/data/chest_xray/test/PNEUMONIA/person1_virus_8.jpeg")
    print("✅ Grad-CAM généré !")