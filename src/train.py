import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import NudeDataset
from model import SwinTransformerModel
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Train a Transformer-based nude detection model")
parser.add_argument("--data", type=str, required=True, help="Path to dataset")
parser.add_argument("--save", type=str, required=True, help="Path to save model")
args = parser.parse_args()

# Load dataset with transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
])

dataset = NudeDataset(args.data, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinTransformerModel(num_classes=2).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save trained model
os.makedirs(args.save, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.save, "nude_detector.pth"))
print(f"Model saved at {args.save}/nude_detector.pth")
