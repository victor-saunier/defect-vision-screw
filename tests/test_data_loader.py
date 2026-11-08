import numpy as np
from PIL import Image

from src.data_loader import ScrewDataset, get_cv_folds


def make_split(root, split, folders):
    for folder, n in folders.items():
        d = root / split / folder
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(d / f"{i:03d}.png")


def test_labels_follow_folder_names(tmp_path):
    make_split(tmp_path, "test", {"good": 3, "scratch_head": 2})
    dataset = ScrewDataset(tmp_path, split="test")
    assert len(dataset) == 5
    assert sorted(dataset.labels) == [0, 0, 0, 1, 1]
    image, label, _ = dataset[0]
    assert image.shape == (3, 224, 224)


def test_cv_folds_are_disjoint_and_cover_all_parts(tmp_path):
    make_split(tmp_path, "train", {"good": 10})
    folds = list(get_cv_folds(str(tmp_path), batch_size=4, n_splits=5))
    assert len(folds) == 5
    val_paths = []
    for train_loader, val_loader in folds:
        val = {p for _, _, paths in val_loader for p in paths}
        train = {p for _, _, paths in train_loader for p in paths}
        assert val.isdisjoint(train)  # une pièce de validation n'est jamais dans la banque
        val_paths += list(val)
    assert len(set(val_paths)) == 10  # chaque pièce est validée exactement une fois
