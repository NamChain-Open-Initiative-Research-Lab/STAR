from model import SwinTransformerMultiLabel
import torch
from PIL import Image
import torchvision.transforms as transforms
import json

# ✅ Define the correct number of classes
NUM_CLASSES = 16  

# ✅ Load model with the correct classifier head
model = SwinTransformerMultiLabel(num_classes=NUM_CLASSES)

# ✅ Load weights while ignoring mismatched layers
checkpoint = torch.load("../models/multi_nude_detector.pth", map_location="cpu")
model_dict = model.state_dict()
filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(filtered_checkpoint)
model.load_state_dict(model_dict, strict=False)

# ✅ Set model to evaluation mode
model.eval()

# ✅ Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ✅ Load class labels
with open("../data/labels.json", "r") as f:
    classes = sorted(set(tag for tags in json.load(f).values() for tag in tags))

# ✅ Ensure classes length matches model output
if len(classes) != NUM_CLASSES:
    raise ValueError(f"❌ Mismatch: Model expects {NUM_CLASSES} classes, but labels.json has {len(classes)} labels!")

# ✅ Load test image
img_path = "C:\\Users\\RamaguruRadhakrishna\\Videos\\STAR-main\\STAR-main\\data\\images\\410.jpeg"
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# ✅ Perform inference
with torch.no_grad():
    output = model(image)
    print(f"🔹 Model Output Shape: {output.shape}")  # Debugging

    # ✅ Ensure index range does not exceed the output size
    predicted_labels = [classes[i] for i in range(min(len(classes), output.shape[1])) if output[0][i] > 0.5]

print("✅ Predicted Tags:", predicted_labels)
