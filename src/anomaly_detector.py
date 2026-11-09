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
    def __init__(self, n_neighbors: int = 1, pool: bool = True):
        self.extractor = FeatureExtractor(pool=pool)
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

    def _describe(self, images: torch.Tensor) -> torch.Tensor:
        """Descripteurs locaux d'un lot d'images [B, 3, 224, 224] : un vecteur par position, [B, 784, 384]."""
        feat1, feat2 = self.extractor(images)
        features = self._fuse(feat1, feat2)
        return features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])

    def extract_patches(self, loader) -> torch.Tensor:
        """
        Calcule les descripteurs locaux de toutes les images d'un loader : [N_images, 784, 384].
        Le backbone est gelé : le résultat est toujours le même pour une image donnée, on peut donc
        le calculer une seule fois et le réutiliser (validation croisée, plusieurs banques).
        """
        return torch.cat([self._describe(images) for images, _, _ in loader], dim=0)

    def fit_patches(self, patches: torch.Tensor, subsample_ratio: float = 0.1, seed: int = 0, batch_size: int = 16):
        """
        Construit la banque de référence à partir des descripteurs d'images saines [N_images, 784, 384].
        subsample_ratio permet de sous-échantillonner les patchs pour optimiser la mémoire CPU.
        Le tirage se fait par lots de batch_size images.
        """
        torch.manual_seed(seed)  # sous-échantillonnage reproductible
        collected_patches = []

        for start in range(0, patches.shape[0], batch_size):
            batch = patches[start:start + batch_size].reshape(-1, patches.shape[-1])

            # Sous-échantillonnage aléatoire pour limiter l'empreinte mémoire
            n_select = int(batch.shape[0] * subsample_ratio)
            indices = torch.randperm(batch.shape[0])[:n_select]
            collected_patches.append(batch[indices])

        self.memory_bank = torch.cat(collected_patches, dim=0).numpy()
        print(f"[INFO] Banque de mémoire construite : {self.memory_bank.shape[0]} patchs de dimension {self.memory_bank.shape[1]}")
        print("[INFO] Indexation k-NN en cours...")
        self.nn_index.fit(self.memory_bank)
        print("[SUCCESS] Modèle prêt pour l'inspection.")

    def fit(self, train_loader, subsample_ratio: float = 0.1, seed: int = 0):
        """Construit la banque de référence à partir des images saines (train/good)."""
        print("[INFO] Extraction des descripteurs nominaux (Train)...")
        self.fit_patches(self.extract_patches(train_loader), subsample_ratio, seed)

    def distance_map_from_patches(self, patches: torch.Tensor) -> np.ndarray:
        """
        Carte de distances [28, 28] à partir des descripteurs [784, 384] d'une image :
        distance de chaque patch à son plus proche voisin dans la banque nominale.
        """
        distances, _ = self.nn_index.kneighbors(patches.numpy())
        side = int(patches.shape[0] ** 0.5)
        return distances.reshape(side, side)

    def distance_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Carte de distances [28, 28] pour une image [3, 224, 224]."""
        return self.distance_map_from_patches(self._describe(image_tensor.unsqueeze(0))[0])

if __name__ == "__main__":
    train_loader, _ = get_dataloaders(batch_size=32)
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.05)
