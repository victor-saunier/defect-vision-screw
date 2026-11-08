import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from src.data_loader import get_dataloaders, get_cv_folds
from src.inspector import OpticalInspector, THRESHOLD_PATH
from src.anomaly_detector import PatchAnomalyDetector

def evaluate_pipeline():
    print("=======================================================")
    print("   ÉVALUATION DE L'INSPECTION OPTIQUE (MVTec Screw)    ")
    print("=======================================================")

    # 1. Seuil par validation croisée (5 blocs) sur les pièces saines de train/good :
    #    chaque pièce est scorée par une banque qui ne l'a jamais vue, le test n'est pas utilisé
    val_scores = []
    for fold, (fold_train, fold_val) in enumerate(get_cv_folds(), start=1):
        print(f"\n[INFO] Validation croisée : bloc {fold}/5")
        fold_detector = PatchAnomalyDetector()
        fold_detector.fit(fold_train, subsample_ratio=0.20)
        fold_inspector = OpticalInspector(detector=fold_detector)
        for images, _, _ in fold_val:
            val_scores.extend(fold_inspector.score(img) for img in images)
    threshold = float(np.percentile(val_scores, 95))
    print(f"\n[INFO] Seuil calibré sur {len(val_scores)} pièces saines : {threshold:.4f}")
    Path("models").mkdir(exist_ok=True)
    Path(THRESHOLD_PATH).write_text(json.dumps({"score_threshold": threshold}))

    # 2. Modèle final : banque de référence sur toutes les pièces saines
    train_loader, test_loader = get_dataloaders(batch_size=16)
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.20)
    inspector = OpticalInspector(detector=detector)

    # 3. Inférence sur le jeu de test complet
    print("\n[INFO] Inférence sur les pièces de test (saines et défectueuses)...")
    y_true, y_scores = [], []
    for images, labels, _ in test_loader:
        y_scores.extend(inspector.score(img) for img in images)
        y_true.extend(labels.numpy())
    y_true, y_scores = np.array(y_true), np.array(y_scores)
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