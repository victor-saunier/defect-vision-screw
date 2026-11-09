import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve

from src.data_loader import get_dataloaders, get_cv_folds
from src.inspector import OpticalInspector, THRESHOLD_PATH
from src.anomaly_detector import PatchAnomalyDetector

def report_operating_points(y_true, y_scores, val_scores, save_plot):
    """
    Compare plusieurs seuils. Chacun est choisi uniquement à partir des scores des pièces saines
    de validation (jamais du test) : le test sert seulement à constater ce que donne chaque réglage.
    """
    print("\n======== POINTS DE FONCTIONNEMENT (seuil choisi sans le test) ========")
    print("Percentile | Seuil  | Rappel  | Pièces saines rejetées | Précision (test)")
    points = []
    for p in (99, 95, 90, 80):
        threshold = np.percentile(val_scores, p)
        rejected = y_scores >= threshold
        recall = rejected[y_true == 1].mean()          # part des défauts détectés
        false_alarm = rejected[y_true == 0].mean()     # part des pièces saines rejetées
        precision = rejected[y_true == 1].sum() / rejected.sum()
        print(f"{p:>10} | {threshold:.3f}  | {recall*100:5.1f} % | {false_alarm*100:15.1f} %        | {precision*100:11.1f} %")
        points.append((p, false_alarm, recall))

    if save_plot:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        n_nominal = int((y_true == 0).sum())
        plt.figure(figsize=(6, 4.5))
        plt.plot(fpr, tpr, label=f"Courbe ROC (AUROC {roc_auc_score(y_true, y_scores)*100:.1f} %)")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
        for p, false_alarm, recall in points:
            plt.scatter(false_alarm, recall, color="red", zorder=3)
            plt.annotate(f"P{p}", (false_alarm, recall), textcoords="offset points", xytext=(8, -14))
        plt.xlabel(f"Pièces saines rejetées (sur {n_nominal} pièces saines : marches de {100 / n_nominal:.1f} points)")
        plt.ylabel("Rappel (défauts détectés)")
        plt.title("Compromis détection / pièces saines rejetées")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/roc_curve.png", dpi=100)
        plt.close()

def evaluate_pipeline(baseline: bool = False, percentile: float = 95):
    # Baseline = version initiale du projet : sans lissage local et avec 10 % des patchs en mémoire
    pool, ratio = (False, 0.10) if baseline else (True, 0.20)

    print("=======================================================")
    print("   ÉVALUATION DE L'INSPECTION OPTIQUE (MVTec Screw)    ")
    print("=======================================================")

    # 0. Descripteurs de toutes les images calculés une seule fois : le backbone est gelé, le résultat
    #    est identique à chaque fois, on le réutilise pour chaque banque de la validation croisée
    train_loader, test_loader = get_dataloaders(batch_size=16)
    detector = PatchAnomalyDetector(pool=pool)
    inspector = OpticalInspector(detector=detector)
    print("[INFO] Extraction des descripteurs (train et test)...")
    train_patches = detector.extract_patches(train_loader)
    test_patches = detector.extract_patches(test_loader)
    y_true = np.array(test_loader.dataset.labels)

    # 1. Seuil par validation croisée (5 blocs) sur les pièces saines de train/good :
    #    chaque pièce est scorée par une banque qui ne l'a jamais vue, le test n'est pas utilisé
    val_scores = []
    for fold, (train_idx, val_idx) in enumerate(get_cv_folds(len(train_patches)), start=1):
        print(f"\n[INFO] Validation croisée : bloc {fold}/5")
        detector.fit_patches(train_patches[train_idx], subsample_ratio=ratio)
        val_scores.extend(inspector.score_patches(p) for p in train_patches[val_idx])
    threshold = float(np.percentile(val_scores, percentile))
    print(f"\n[INFO] Seuil calibré sur {len(val_scores)} pièces saines (percentile {percentile:g}) : {threshold:.4f}")
    if not baseline:  # la baseline ne doit pas écraser le seuil utilisé par run_inspection.py
        Path("models").mkdir(exist_ok=True)
        Path(THRESHOLD_PATH).write_text(json.dumps({"score_threshold": threshold}))

    # 2. Modèle final : banque de référence sur toutes les pièces saines
    detector.fit_patches(train_patches, subsample_ratio=ratio)

    # 3. Inférence sur le jeu de test complet
    print("\n[INFO] Inférence sur les pièces de test (saines et défectueuses)...")
    y_scores = np.array([inspector.score_patches(p) for p in test_patches])
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

    report_operating_points(y_true, y_scores, np.array(val_scores), save_plot=not baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="évalue la version initiale (sans lissage, 10 %% des patchs)")
    parser.add_argument("--percentile", type=float, default=95, help="percentile des scores sains retenu comme seuil (plus bas = plus de défauts détectés, plus de fausses alertes)")
    args = parser.parse_args()
    evaluate_pipeline(baseline=args.baseline, percentile=args.percentile)