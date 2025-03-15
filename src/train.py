import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import NudeMultiLabelDataset
from model import SwinTransformerMultiLabel
import argparse
import os
import time  

# Parse arguments
parser = argparse.ArgumentParser(description="Train a Transformer-based nude classification model")
parser.add_argument("--data", type=str, required=True, help="Path to dataset")
parser.add_argument("--labels", type=str, required=True, help="Path to labels.json")
parser.add_argument("--save", type=str, required=True, help="Path to save model")
args = parser.parse_args()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = NudeMultiLabelDataset(args.data, args.labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinTransformerMultiLabel(num_classes=len(dataset.classes)).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Start measuring training time
start_time = time.time()

# Training loop
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_start = time.time()  # Track epoch time
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # Ensure correct shape

        # Debugging: Print tensor shapes
        print(f"🔹 Outputs shape: {outputs.shape}")  # Expected: [batch_size, num_classes]
        print(f"🔹 Labels shape: {labels.shape}")  # Expected: [batch_size, num_classes]

        # Ensure output is (batch_size, num_classes)
        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)  # Flatten spatial dimensions if present

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_end = time.time()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}, Time: {epoch_end - epoch_start:.2f} sec")
    
# End measuring training time
end_time = time.time()
total_time = end_time - start_time    

# Save trained model
os.makedirs(args.save, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.save, "multi_nude_detector.pth"))
print(f"✅ Model saved at {args.save}/multi_nude_detector.pth")
print(f"⏳ Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
