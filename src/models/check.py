import os

def count_classes_in_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"❌ Path not found: {dataset_path}")
        return 0
    
    class_folders = [folder for folder in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, folder))]
    
    print(f"📁 Found {len(class_folders)} classes in '{dataset_path}':")
    for cls in class_folders:
        print(f" - {cls}")
    
    return len(class_folders)

# Replace these with your actual paths (from config.yaml)
train_path = "data/train"
test_path = "data/test"

num_train_classes = count_classes_in_dataset(train_path)
num_test_classes = count_classes_in_dataset(test_path)

print(f"\n✅ Total Train Classes: {num_train_classes}")
print(f"✅ Total Test Classes: {num_test_classes}")
