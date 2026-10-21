import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

from src.feature_extractor import FeatureExtractor
from src.data_loader import get_dataloaders

class PatchAnomalyDetector:
    """
    Détecteur d'anomalies basé sur l'extraction de patchs multi-échelles
    et la comparaison par plus proches voisins (k-NN).
    """
    def __init__(self, n_neighbors: int = 1):
        self.extractor = FeatureExtractor()
        self.n_neighbors = n_neighbors
        self.memory_bank = None
        self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")

    def _extract_patches(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Aligne layer3 sur la résolution spatiale de layer2 et concatène les canaux.
        Retourne un tenseur de descripteurs locaux de dimension [N_patches, 384].
        """
        # Interpolation bilinéaire de layer3 (14x14 -> 28x28)
        feat2_upsampled = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        # Concaténation sur l'axe des canaux : [B, 128+256, 28, 28] = [B, 384, 28, 28]
        features = torch.cat([feat1, feat2_upsampled], dim=1)

        b, c, h, w = features.shape
        # Permutation et reshape pour obtenir un vecteur par position spatiale : [B * H * W, C]
        patches = features.permute(0, 2, 3, 1).reshape(-1, c)
        return patches

    def fit(self, train_loader, subsample_ratio: float = 0.1):
        """
        Construit la banque de référence à partir des images saines (train/good).
        subsample_ratio permet de sous-échantillonner les patchs pour optimiser la mémoire CPU.
        """
        print("[INFO] Extraction des descripteurs nominaux (Train)...")
        collected_patches = []

        for i, (images, _, _) in enumerate(train_loader):
            feat1, feat2 = self.extractor(images)
            patches = self._extract_patches(feat1, feat2)

            # Sous-échantillonnage aléatoire pour limiter l'empreinte mémoire
            n_select = int(patches.shape[0] * subsample_ratio)
            indices = torch.randperm(patches.shape[0])[:n_select]
            collected_patches.append(patches[indices])

        self.memory_bank = torch.cat(collected_patches, dim=0).numpy()
        print(f"[INFO] Banque de mémoire construite : {self.memory_bank.shape[0]} patchs de dimension {self.memory_bank.shape[1]}")
        print("[INFO] Indexation k-NN en cours...")
        self.nn_index.fit(self.memory_bank)
        print("[SUCCESS] Modèle prêt pour l'inspection.")

    def predict_score(self, images: torch.Tensor) -> np.ndarray:
        """
        Calcule un score d'anomalie global pour chaque image d'un batch.
        Le score correspond à la distance maximale détectée sur la pièce.
        """
        feat1, feat2 = self.extractor(images)
        feat2_upsampled = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([feat1, feat2_upsampled], dim=1)

        b, c, h, w = features.shape
        scores = []

        for i in range(b):
            img_patches = features[i].permute(1, 2, 0).reshape(-1, c).numpy()
            # Distance au plus proche voisin nominal
            distances, _ = self.nn_index.kneighbors(img_patches)
            # Le score de l'image est la déviation maximale observée sur la surface
            max_dist = np.max(distances)
            scores.append(max_dist)

        return np.array(scores)

if __name__ == "__main__":
    train_loader, _ = get_dataloaders(batch_size=32)
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.05)