import os
import shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image

DATA_DIR = Path("data") / "screw"
CATEGORY = "screw"

def get_pil_image(item, key="image_path"):
    val = item[key]
    if isinstance(val, Image.Image):
        return val
    return Image.open(val)

def setup_dataset():
    train_good_dir = DATA_DIR / "train" / "good"
    if train_good_dir.exists() and len(list(train_good_dir.glob("*.png"))) > 0:
        print(f"[INFO] Les images sont déjà en place dans {DATA_DIR}")
        return

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    print(f"[INFO] Chargement du split train '{CATEGORY}'...")
    train_ds = load_dataset("TheoM55/mvtec_all_objects_split", split=f"{CATEGORY}.train")

    print(f"[INFO] Chargement du split test '{CATEGORY}'...")
    test_ds = load_dataset("TheoM55/mvtec_all_objects_split", split=f"{CATEGORY}.test")

    # 1. Sauvegarde des vis saines pour l'entraînement
    print("[INFO] Sauvegarde du dossier train/good...")
    train_good_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(train_ds):
        img = get_pil_image(item, "image_path")
        img.save(train_good_dir / f"{i:03d}.png")

    # 2. Sauvegarde des vis de test réparties par défaut
    print("[INFO] Sauvegarde et tri du dossier test...")
    test_counts = {}
    for i, item in enumerate(test_ds):
        img = get_pil_image(item, "image_path")
        defect_type = item.get("defect", "unknown")

        target_subfolder = DATA_DIR / "test" / str(defect_type)
        target_subfolder.mkdir(parents=True, exist_ok=True)

        img.save(target_subfolder / f"{i:03d}.png")
        test_counts[defect_type] = test_counts.get(defect_type, 0) + 1

    train_count = len(list(train_good_dir.glob("*.png")))
    print("\n================ RÉCAPITULATIF DATASET ================")
    print(f"• Images conformes (train/good)    : {train_count}")
    print("• Répartition du jeu de test :")
    for defect_name, count in sorted(test_counts.items()):
        print(f"   - {defect_name:<18} : {count} images")
    print("=======================================================")

if __name__ == "__main__":
    setup_dataset()