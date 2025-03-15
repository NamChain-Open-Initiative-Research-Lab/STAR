from model import SwinTransformerMultiLabel
import torch
from PIL import Image
import torchvision.transforms as transforms
import json

# Load trained model
model = SwinTransformerMultiLabel(num_classes=12)  # Adjust based on total categories
model.load_state_dict(torch.load("models/multi_nude_detector.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class names
with open("data/labels.json", "r") as f:
    classes = sorted(set(tag for tags in json.load(f).values() for tag in tags))

# Test image
img_path = "test_image.jpg"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(image)
    predicted_labels = [classes[i] for i in range(len(classes)) if output[0][i] > 0.5]  # Threshold = 0.5

print("Predicted Tags:", predicted_labels)
