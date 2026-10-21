import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """
    Extracteur de descripteurs visuels multi-échelles avec agrégation locale (PatchCore-like).
    Intègre un lissage par voisinage pour immuniser la détection contre le bruit optique.
    """
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Lissage par moyenne locale : capture le contexte spatial immédiat du défaut
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.stem(x)
            f1 = self.layer2(x)
            f2 = self.layer3(f1)
            
            # Lissage spatial des descripteurs
            f1 = self.avg_pool(f1)
            f2 = self.avg_pool(f2)
            
        return f1, f2