import sys
from pathlib import Path

# Résolution automatique du chemin racine
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

from src.data_loader import get_dataloaders
from src.anomaly_detector import PatchAnomalyDetector

def evaluate_pipeline():
    print("=======================================================")
    print("   ÉVALUATION DE L'INSPECTION OPTIQUE (MVTec Screw)    ")
    print("=======================================================")

    # 1. Chargement des données
    train_loader, test_loader = get_dataloaders(batch_size=16)

    # 2. Construction de la mémoire de référence
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.20)

    # 3. Inférence sur le jeu de test complet
    print("\n[INFO] Inférence sur les pièces de test (saines et défectueuses)...")
    y_true = []
    y_scores = []

    for images, labels, _ in test_loader:
        scores = detector.predict_score(images)
        y_scores.extend(scores)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 4. Calcul de l'AUROC globale
    auroc = roc_auc_score(y_true, y_scores)

    # 5. Détermination du seuil de décision optimal (seuil séparateur)
    # On calibre le seuil au 95e pourcentage des pièces nominales de test
    nominal_scores = y_scores[y_true == 0]
    threshold = np.percentile(nominal_scores, 95)
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