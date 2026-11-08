import torch

from src.anomaly_detector import PatchAnomalyDetector


def test_fusion_gives_384_channels_at_layer2_resolution():
    detector = PatchAnomalyDetector.__new__(PatchAnomalyDetector)  # sans charger ResNet
    feat1 = torch.rand(2, 128, 28, 28)  # layer2
    feat2 = torch.rand(2, 256, 14, 14)  # layer3
    assert detector._fuse(feat1, feat2).shape == (2, 384, 28, 28)
    assert detector._extract_patches(feat1, feat2).shape == (2 * 28 * 28, 384)
