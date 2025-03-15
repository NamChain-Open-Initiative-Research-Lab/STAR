import torch
import torch.nn as nn
from torchvision.models import swin_t

class SwinTransformerMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerMultiLabel, self).__init__()
        self.model = swin_t(weights="IMAGENET1K_V1")

        # Adjust final classification layer
        in_features = self.model.head.in_features  # Should be 768
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.features(x)  # Extract features
        
        print(f"🔹 Feature map shape before flattening: {x.shape}")  # Debugging output

        # ✅ Correctly apply GAP over height & width
        x = x.mean(dim=[1, 2])  # Now shape is (batch_size, 768)
        print(f"🔹 Feature shape after GAP: {x.shape}")  

        x = self.model.head(x)  # Classification layer
        return x

        
def main():
    # Define number of classes
    num_classes = 2  

    # Create the model
    model = SwinTransformerMultiLabel(num_classes)

    # Set the model to evaluation mode
    model.eval()

    # Generate a dummy input tensor (batch_size=5, channels=3, height=224, width=224)
    dummy_input = torch.randn(5, 3, 224, 224)

    # Forward pass through the model
    output = model(dummy_input)

    # Print output shape
    print(f"✅ Model output shape: {output.shape}")  # Expected: (5, 2)

    # Check model parameters (classification head)
    print(f"✅ Model classification head: {model.model.head}")

    # Check with different batch sizes
    for batch_size in [1, 8, 16]:
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        output = model(dummy_input)
        print(f"✅ Batch Size {batch_size} -> Output Shape: {output.shape}")

if __name__ == "__main__":
    main()        
