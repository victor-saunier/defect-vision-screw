import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

    def _fuse(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Aligne layer3 sur la résolution spatiale de layer2 et concatène les canaux : [B, 384, 28, 28].
        """
        # Interpolation bilinéaire de layer3 (14x14 -> 28x28)
        feat2_upsampled = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([feat1, feat2_upsampled], dim=1)

    def _extract_patches(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Retourne un descripteur local par position spatiale : [N_patches, 384]."""
        features = self._fuse(feat1, feat2)
        # Permutation et reshape pour obtenir un vecteur par position spatiale : [B * H * W, C]
        return features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])

    def fit(self, train_loader, subsample_ratio: float = 0.1, seed: int = 0):
        """
        Construit la banque de référence à partir des images saines (train/good).
        subsample_ratio permet de sous-échantillonner les patchs pour optimiser la mémoire CPU.
        """
        print("[INFO] Extraction des descripteurs nominaux (Train)...")
        torch.manual_seed(seed)  # sous-échantillonnage reproductible
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

    def distance_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Carte de distances [28, 28] entre chaque patch d'une image [3, 224, 224]
        et son plus proche voisin dans la banque nominale.
        """
        feat1, feat2 = self.extractor(image_tensor.unsqueeze(0))
        features = self._fuse(feat1, feat2)
        _, c, h, w = features.shape
        patches = features[0].permute(1, 2, 0).reshape(-1, c).numpy()
        distances, _ = self.nn_index.kneighbors(patches)
        return distances.reshape(h, w)

    def predict_score(self, images: torch.Tensor) -> np.ndarray:
        """
        Score d'anomalie global pour chaque image d'un batch :
        la déviation maximale observée sur la pièce.
        """
        return np.array([self.distance_map(img).max() for img in images])

if __name__ == "__main__":
    train_loader, _ = get_dataloaders(batch_size=32)
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.05)