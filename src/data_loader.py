import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ScrewDataset(Dataset):
    """
    Dataset PyTorch pour l'inspection optique de vis industrielles.
    Gère le chargement, le redimensionnement et la normalisation ImageNet.
    """
    def __init__(self, root_dir: str, split: str = "train", img_size: int = 224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_paths = []
        self.labels = []  # 0 pour nominal (good), 1 pour défectueux

        # Normalisation standard requise par les modèles pré-entraînés PyTorch
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._load_dataset()

    def _load_dataset(self):
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dossier introuvable : {split_dir}")

        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                is_nominal = (category_dir.name == "good")
                label = 0 if is_nominal else 1

                for img_path in category_dir.glob("*.png"):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        label = self.labels[idx]
        return tensor, label, str(path)

def get_dataloaders(data_dir: str = "data/screw", batch_size: int = 16):
    train_dataset = ScrewDataset(root_dir=data_dir, split="train")
    test_dataset = ScrewDataset(root_dir=data_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader