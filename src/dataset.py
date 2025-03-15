import os
import json
import torch
from torch.utils.data import Dataset
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
