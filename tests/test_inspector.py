import numpy as np
import torch

from src.inspector import OpticalInspector


class FakeDetector:
    """Détecteur factice : renvoie une carte de distances connue, sans réseau ni modèle."""
    def __init__(self, distance_map):
        self._map = distance_map

    def distance_map(self, image_tensor):
        return self._map


def make_map(defect: bool) -> np.ndarray:
    m = np.full((28, 28), 0.5, dtype=np.float32)
    if defect:
        m[10:20, 10:20] = 5.0  # zone très éloignée de la banque nominale
    return m


def test_nominal_part_passes():
    inspector = OpticalInspector(FakeDetector(make_map(defect=False)), score_threshold=1.7)
    result = inspector.inspect(torch.zeros(3, 224, 224))
    assert result.decision == "PASS"
    assert result.defects == []


def test_defective_part_fails_with_area_in_mm2():
    inspector = OpticalInspector(FakeDetector(make_map(defect=True)), score_threshold=1.7)
    result = inspector.inspect(torch.zeros(3, 224, 224))
    assert result.decision == "FAIL"
    assert len(result.defects) == 1

    defect = result.defects[0]
    # 1 px de l'image 224x224 = 1024/224 px d'origine, à 0,05 mm chacun
    mm_per_px = 0.05 * 1024 / 224
    assert defect.area_mm2 == defect.area_px * mm_per_px ** 2


def test_score_is_max_of_map():
    inspector = OpticalInspector(FakeDetector(make_map(defect=True)))
    assert 4.0 < inspector.score(torch.zeros(3, 224, 224)) <= 5.0
