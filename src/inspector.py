import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

from src.anomaly_detector import PatchAnomalyDetector

@dataclass
class DefectMetric:
    defect_id: int
    bbox: Tuple[int, int, int, int]
    area_px: int
    area_mm2: float

@dataclass
class InspectionResult:
    decision: str
    anomaly_score: float
    defects: List[DefectMetric]
    annotated_image: np.ndarray

class OpticalInspector:
    def __init__(
        self,
        detector: PatchAnomalyDetector,
        score_threshold: float = 1.70,     # Seuil d'anomalie brut issu de evaluate.py
        min_defect_area_px: int = 40,
        pixel_to_mm_ratio: float = 0.05
    ):
        self.detector = detector
        self.score_threshold = score_threshold
        self.min_defect_area_px = min_defect_area_px
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def _compute_raw_anomaly_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        self.detector.extractor.eval()
        with torch.no_grad():
            feat1, feat2 = self.detector.extractor(image_tensor.unsqueeze(0))
            feat2_up = torch.nn.functional.interpolate(
                feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=False
            )
            features = torch.cat([feat1, feat2_up], dim=1)

            _, c, h, w = features.shape
            patches = features[0].permute(1, 2, 0).reshape(-1, c).numpy()

            distances, _ = self.detector.nn_index.kneighbors(patches)
            raw_map = distances.reshape(h, w)

            # Redimensionnement direct sans écrasement Min-Max artificiel
            raw_map = cv2.resize(raw_map, (224, 224), interpolation=cv2.INTER_LINEAR)
            raw_map = cv2.GaussianBlur(raw_map, (9, 9), sigmaX=4)
            return raw_map

    def inspect(self, image_tensor: torch.Tensor) -> InspectionResult:
        raw_map = self._compute_raw_anomaly_map(image_tensor)
        global_score = float(np.max(raw_map))

        # 1. Binarisation basée sur la distance euclidienne réelle de référence
        binary_mask = (raw_map >= self.score_threshold).astype(np.uint8) * 255

        # 2. Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 3. Composantes connexes
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        defects: List[DefectMetric] = []
        defect_idx = 1

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_defect_area_px:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])

                area_mm2 = area * (self.pixel_to_mm_ratio ** 2)
                defects.append(DefectMetric(defect_id=defect_idx, bbox=(x, y, w, h), area_px=area, area_mm2=area_mm2))
                defect_idx += 1

        # 4. Décision industrielle
        decision = "FAIL" if (global_score >= self.score_threshold and len(defects) > 0) else "PASS"

        # 5. Dé-normalisation pour affichage
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        raw_img = image_tensor.permute(1, 2, 0).numpy()
        annotated = np.clip((raw_img * std + mean) * 255.0, 0, 255).astype(np.uint8)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # 6. Rendu UI industriel
        status_color = (0, 0, 255) if decision == "FAIL" else (0, 255, 0)
        cv2.putText(annotated, f"STATUT: {decision}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Score: {global_score:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        if decision == "FAIL":
            for d in defects:
                x, y, w, h = d.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, f"#{d.defect_id}: {d.area_mm2:.2f}mm2", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        return InspectionResult(
            decision=decision,
            anomaly_score=global_score,
            defects=defects if decision == "FAIL" else [],
            annotated_image=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        )