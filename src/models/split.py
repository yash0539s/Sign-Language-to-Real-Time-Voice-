import os
import shutil
import random

SRC_DIR = 'dataset'
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
SPLIT_RATIO = 0.8

for label in os.listdir(SRC_DIR):
    files = os.listdir(os.path.join(SRC_DIR, label))
    random.shuffle(files)

    train_len = int(len(files) * SPLIT_RATIO)
    train_files = files[:train_len]
    test_files = files[train_len:]

    os.makedirs(os.path.join(TRAIN_DIR, label), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, label), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(SRC_DIR, label, f), os.path.join(TRAIN_DIR, label, f))

    for f in test_files:
        shutil.copy(os.path.join(SRC_DIR, label, f), os.path.join(TEST_DIR, label, f))

print("[✅] Dataset split complete!")
