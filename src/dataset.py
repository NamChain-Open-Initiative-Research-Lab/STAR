import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class NudeMultiLabelDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_file = label_file

        # Load labels
        with open(label_file, "r") as f:
            self.labels = json.load(f)

        self.image_paths = list(self.labels.keys())
        self.classes = sorted(set(tag for tags in self.labels.values() for tag in tags))
        self.class_to_idx = {tag: idx for idx, tag in enumerate(self.classes)}

        # Print dataset info
        print(f"📂 Dataset loaded from: {data_dir}")
        print(f"📄 Labels loaded from: {label_file}")
        print(f"🖼️ Total images: {len(self.image_paths)}")
        print(f"🏷️ Unique labels: {len(self.classes)}")
        print(f"🔹 Label-to-Index Mapping: {self.class_to_idx}")

        # Print example data
        if self.image_paths:
            example_img, example_label = self.__getitem__(0)
            print(f"✅ Example Image Shape: {example_img.shape}")
            print(f"✅ Example Label: {example_label}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Convert labels to multi-hot encoding
        labels = self.labels[img_name]
        label_tensor = torch.zeros(len(self.classes))
        for tag in labels:
            if tag in self.class_to_idx:
                label_tensor[self.class_to_idx[tag]] = 1  # Multi-label

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# 🔹 Main function to test the dataset independently
if __name__ == "__main__":
    # Set paths
    DATA_DIR = "../data/images"   # Change to actual path
    LABEL_FILE = "../data/labels.json"   # Change to actual path

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = NudeMultiLabelDataset(DATA_DIR, LABEL_FILE, transform=transform)
    
    # Create DataLoader for testing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Fetch one batch and print information
    for images, labels in dataloader:
        print(f"🖼️ Batch Image Shape: {images.shape}")  # Should be [batch_size, 3, 224, 224]
        print(f"🏷️ Batch Labels: {labels}")
        break  # Stop after one batch
