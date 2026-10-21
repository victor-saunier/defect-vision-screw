import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """
    Extracteur de descripteurs visuels basé sur ResNet-18.
    Capture les représentations intermédiaires (layer2 et layer3) 
    adaptées à l'inspection de textures et d'anomalies de surface.
    """
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # On isole les blocs convolutionnels en amont de layer2 et layer3
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Mode évaluation strict : désactivation du Dropout et du BatchNorm dynamique
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.stem(x)
            f1 = self.layer2(x) # Forme : [Batch, 128, H/8, W/8]
            f2 = self.layer3(f1) # Forme : [Batch, 256, H/16, W/16]
        return f1, f2

if __name__ == "__main__":
    extractor = FeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    feat1, feat2 = extractor(dummy_input)
    print("[TEST] Extracteur initialisé avec succès :")
    print(f"• Layer2 feature map shape : {feat1.shape}")
    print(f"• Layer3 feature map shape : {feat2.shape}")