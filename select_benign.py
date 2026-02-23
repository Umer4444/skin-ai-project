import os
import random
import shutil

source_dir = "data/images/benign"
target_dir = "data/images/benign_15k"

os.makedirs(target_dir, exist_ok=True)

images = [f for f in os.listdir(source_dir)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print("Total benign images:", len(images))

selected_images = random.sample(images, 15000)

for img in selected_images:
    shutil.copy(
        os.path.join(source_dir, img),
        os.path.join(target_dir, img)
    )

print("Copied 15,000 benign images successfully")




