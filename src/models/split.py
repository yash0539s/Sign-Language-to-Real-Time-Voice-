import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        cls_source = os.path.join(source_dir, cls)
        cls_train = os.path.join(train_dir, cls)
        cls_val = os.path.join(val_dir, cls)

        os.makedirs(cls_train, exist_ok=True)
        os.makedirs(cls_val, exist_ok=True)

        images = os.listdir(cls_source)
        random.shuffle(images)

        train_count = int(len(images) * split_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:]

        for img in train_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_train, img))

        for img in val_images:
            shutil.copy(os.path.join(cls_source, img), os.path.join(cls_val, img))

    print("✅ Dataset split complete!")

# Usage example:
source_folder = "data/asl_alphabet_train"
train_folder = "data/asl_alphabet_train_split/train"
val_folder = "data/asl_alphabet_train_split/val"

split_data(source_folder, train_folder, val_folder)

