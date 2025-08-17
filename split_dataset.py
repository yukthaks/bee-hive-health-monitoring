import os
import shutil
import random

#input folders
images_dir = "dataset/images"
labels_dir = "dataset/annotations"

#target folders
base_dir = "data"
image_out = os.path.join(base_dir, "images")
label_out = os.path.join(base_dir, "labels")
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
image_files = [f for f in image_files if os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))]
random.shuffle(image_files)
n_total = len(image_files)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)

train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

def copy_pair(files, split):
    for file in files:
        base = os.path.splitext(file)[0]
        img_src = os.path.join(images_dir, file)
        lbl_src = os.path.join(labels_dir, base + ".txt")

        img_dst = os.path.join(image_out, split, file)
        lbl_dst = os.path.join(label_out, split, base + ".txt")

        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)

copy_pair(train_files, "train")
copy_pair(val_files, "val")
copy_pair(test_files, "test")

print(f"Copied: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test images + annotations.")
