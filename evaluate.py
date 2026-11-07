import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from src.data_loader import get_dataloaders_with_val
from src.inspector import OpticalInspector, THRESHOLD_PATH
from src.anomaly_detector import PatchAnomalyDetector

def evaluate_pipeline():
    print("=======================================================")
    print("   ÉVALUATION DE L'INSPECTION OPTIQUE (MVTec Screw)    ")
    print("=======================================================")

    # 1. Chargement des données (20 % des pièces saines réservées au calibrage du seuil)
    train_loader, val_loader, test_loader = get_dataloaders_with_val(batch_size=16)

    # 2. Construction de la mémoire de référence
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.20)
    inspector = OpticalInspector(detector=detector)

    def score_loader(loader):
        scores, labels = [], []
        for images, lbls, _ in loader:
            scores.extend(inspector.score(img) for img in images)
            labels.extend(lbls.numpy())
        return np.array(scores), np.array(labels)

    # 3. Seuil calibré sur les pièces saines de validation (95e percentile), jamais sur le test
    print("\n[INFO] Calibrage du seuil sur les pièces saines de validation...")
    val_scores, _ = score_loader(val_loader)
    threshold = float(np.percentile(val_scores, 95))
    Path("models").mkdir(exist_ok=True)
    Path(THRESHOLD_PATH).write_text(json.dumps({"score_threshold": threshold}))

    # 4. Inférence sur le jeu de test complet
    print("[INFO] Inférence sur les pièces de test (saines et défectueuses)...")
    y_scores, y_true = score_loader(test_loader)
    auroc = roc_auc_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    # 6. Métriques de classification
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 7. Rapport d'inspection
    print("\n================== RÉSULTATS D'INSPECTION ================")
    print(f"• AUROC Globale               : {auroc * 100:.2f} %")
    print(f"• F1-Score                    : {f1 * 100:.2f} %")
    print(f"• Sensibilité (Recall Défauts): {recall * 100:.2f} %")
    print(f"• Précision                   : {precision * 100:.2f} %")
    print(f"• Seuil calibré               : {threshold:.4f}")
    print("----------------------------------------------------------")
    print(f"• Vrais Négatifs (Sains OK)   : {tn} / {tn + fp}")
    print(f"• Vrais Positifs (Défauts OK) : {tp} / {tp + fn}")
    print(f"• Faux Positifs (Fausse alerte: {fp}")
    print(f"• Faux Négatifs (Défauts omis): {fn}")
    print("==========================================================")

if __name__ == "__main__":
    evaluate_pipeline()