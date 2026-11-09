import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.anomaly_detector import PatchAnomalyDetector


def make_detector():
    detector = PatchAnomalyDetector.__new__(PatchAnomalyDetector)  # sans charger ResNet
    detector.nn_index = NearestNeighbors(n_neighbors=1)
    return detector


def test_fusion_gives_384_channels_at_layer2_resolution():
    detector = make_detector()
    feat1 = torch.rand(2, 128, 28, 28)  # layer2
    feat2 = torch.rand(2, 256, 14, 14)  # layer3
    assert detector._fuse(feat1, feat2).shape == (2, 384, 28, 28)


def test_fit_patches_keeps_requested_share_of_patches():
    detector = make_detector()
    patches = torch.rand(10, 784, 384)
    detector.fit_patches(patches, subsample_ratio=0.5, batch_size=4)  # lots de 4, 4 et 2 images
    assert detector.memory_bank.shape == (10 * 784 // 2, 384)


def test_distance_map_separates_known_and_unknown_patches():
    detector = make_detector()
    patches = torch.rand(1, 784, 384)
    detector.fit_patches(patches, subsample_ratio=1.0)

    known = detector.distance_map_from_patches(patches[0])
    unknown = detector.distance_map_from_patches(patches[0] + 5)  # très éloignés de la banque
    assert known.shape == (28, 28)
    assert np.allclose(known, 0, atol=1e-4)  # chaque patch est présent dans la banque
    assert unknown.min() > 1
