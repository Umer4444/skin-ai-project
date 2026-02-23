import os
import random
import shutil

# source images
SOURCE_DIR = "data/backup_images"

# output directories
OUTPUT_DIR = "data"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = ["benign", "melanoma"]

random.seed(42)

for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(class_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)
    total = len(images)

    train_end = int(SPLITS["train"] * total)
    val_end = train_end + int(SPLITS["val"] * total)

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_files.items():
        split_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for file in files:
            src = os.path.join(class_dir, file)
            dst = os.path.join(split_dir, file)
            shutil.copy(src, dst)

    print(f"{cls} split done: {total} images")
