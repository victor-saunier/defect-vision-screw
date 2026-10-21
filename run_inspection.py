import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import get_dataloaders
from src.anomaly_detector import PatchAnomalyDetector
from src.inspector import OpticalInspector

def main():
    print("[INFO] Initialisation du système de contrôle qualité...")
    train_loader, test_loader = get_dataloaders(batch_size=16)

    # Entraînement de la référence nominale
    detector = PatchAnomalyDetector()
    detector.fit(train_loader, subsample_ratio=0.20)

    inspector = OpticalInspector(
        detector=detector,
        score_threshold=1.70,     # Seuil d'anomalie calibré lors de l'évaluation
        min_defect_area_px=50,
        pixel_to_mm_ratio=0.05    # Hypothèse optique : 50 µm / pixel
    )

    # Récupérer une pièce saine et une pièce avec défaut dans le test loader
    good_sample = None
    defect_sample = None

    for images, labels, paths in test_loader:
        for img, lbl, p in zip(images, labels, paths):
            if lbl.item() == 0 and good_sample is None:
                good_sample = (img, p)
            elif lbl.item() == 1 and ("scratch_head" in p or "thread_side" in p) and defect_sample is None:
                defect_sample = (img, p)
            if good_sample is not None and defect_sample is not None:
                break
        if good_sample is not None and defect_sample is not None:
            break

    # Exécution des inspections
    print("\n--- Analyse de la Pièce Conforme ---")
    res_good = inspector.inspect(good_sample[0])
    print(f"• Fichier  : {Path(good_sample[1]).name}")
    print(f"• Décision : {res_good.decision}")
    print(f"• Défauts  : {len(res_good.defects)}")

    print("\n--- Analyse de la Pièce Défectueuse ---")
    res_defect = inspector.inspect(defect_sample[0])
    print(f"• Fichier  : {Path(defect_sample[1]).name}")
    print(f"• Décision : {res_defect.decision}")
    print(f"• Défauts détectés : {len(res_defect.defects)}")
    for d in res_defect.defects:
        print(f"   -> Défaut #{d.defect_id}: Surface = {d.area_px} px ({d.area_mm2:.3f} mm²), Bounding Box = {d.bbox}")

    # Rendu visuel
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(res_good.annotated_image)
    axes[0].set_title(f"Contrôle Optique : {res_good.decision}\n{Path(good_sample[1]).name}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(res_defect.annotated_image)
    axes[1].set_title(f"Contrôle Optique : {res_defect.decision}\n{Path(defect_sample[1]).name}", fontsize=11)
    axes[1].axis("off")

    plt.suptitle("Station de Métrologie & Décision de Tri Industrielle", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()