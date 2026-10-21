import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data") / "screw"

def load_rgb(image_path: Path):
    """Charge une image via OpenCV en conservant le repère RGB pour l'affichage."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def inspect_dataset_samples():
    good_sample = list((DATA_DIR / "train" / "good").glob("*.png"))[0]
    thread_sample = list((DATA_DIR / "test" / "thread_side").glob("*.png"))[0]
    scratch_sample = list((DATA_DIR / "test" / "scratch_head").glob("*.png"))[0]
    front_sample = list((DATA_DIR / "test" / "manipulated_front").glob("*.png"))[0]

    samples = [
        ("Nominale (Conforme)", good_sample),
        ("Filetage abîmé (Thread Side)", thread_sample),
        ("Rayure tête (Scratch Head)", scratch_sample),
        ("Pointe altérée (Manipulated)", front_sample),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (title, path) in zip(axes, samples):
        img = load_rgb(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Inspection Optique MVTec AD - Catégorie 'Screw'", fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Affichage des propriétés d'une trame brute
    h, w, c = load_rgb(good_sample).shape
    print(f"\n[PROPRIÉTÉS MATRICIELLES]")
    print(f"• Résolution native : {w}x{h} pixels")
    print(f"• Canaux couleur    : {c} (RGB)")
    print(f"• Type de données   : uint8 [0, 255]")

    plt.show()

if __name__ == "__main__":
    inspect_dataset_samples()