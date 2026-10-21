import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.anomaly_detector import PatchAnomalyDetector

def generate_anomaly_map(detector: PatchAnomalyDetector, image_tensor: torch.Tensor) -> np.ndarray:
    """
    Génère une carte thermique bidimensionnelle indiquant la déviation spatiale locale.
    """
    detector.extractor.eval()
    with torch.no_grad():
        feat1, feat2 = detector.extractor(image_tensor.unsqueeze(0))
        feat2_upsampled = torch.nn.functional.interpolate(
            feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat([feat1, feat2_upsampled], dim=1)
        
        _, c, h, w = features.shape
        patches = features[0].permute(1, 2, 0).reshape(-1, c).numpy()
        
        # Calcul des distances aux patchs sains de référence
        distances, _ = detector.nn_index.kneighbors(patches)
        anomaly_map = distances.reshape(h, w)

        # Normalisation min-max locale
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        
        # Redimensionnement vers la résolution de l'image d'entrée
        anomaly_map = cv2.resize(anomaly_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        # Lissage gaussien pour adoucir les transitions
        anomaly_map = cv2.GaussianBlur(anomaly_map, (9, 9), sigmaX=4)
        
        return anomaly_map

def run_inspection_visualizer():
    print("[INFO] Chargement des données et construction de la référence...")
    train_loader, test_loader = get_dataloaders(batch_size=16)
    
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.20)

    # Récupération d'un échantillon défectueux dans le test loader
    target_img, target_label, target_path = None, None, None
    for images, labels, paths in test_loader:
        for img, lbl, p in zip(images, labels, paths):
            if lbl.item() == 1 and ("thread_side" in p or "scratch_head" in p):
                target_img, target_label, target_path = img, lbl, p
                break
        if target_img is not None:
            break

    print(f"[INFO] Analyse de l'échantillon : {Path(target_path).name} (Classe : {Path(target_path).parent.name})")
    
    # 1. Calcul de la carte d'anomalie
    heatmap = generate_anomaly_map(detector, target_img)
    
    # 2. Dé-normalisation de l'image d'entrée (ImageNet -> [0, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    raw_img = target_img.permute(1, 2, 0).numpy()
    raw_img = np.clip((raw_img * std + mean) * 255.0, 0, 255).astype(np.uint8)

    # 3. Colorimétrie Jet pour l'anomalie et superposition
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(raw_img, 0.65, heatmap_colored, 0.35, 0)

    # 4. Affichage côte à côte
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Image Optique Brute\n{Path(target_path).parent.name}")
    axes[0].axis("off")

    im_hm = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Carte de Déviation (Heatmap)")
    axes[1].axis("off")
    plt.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Localisation Industrielle Superposée")
    axes[2].axis("off")

    plt.suptitle("Système d'Inspection Optique - Localisation de Défaut", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inspection_visualizer()