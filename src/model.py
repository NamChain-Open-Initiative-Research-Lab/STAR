import torch
import torch.nn as nn
import timm

class SwinTransformerMultiLabel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerMultiLabel, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)
        self.model.head = nn.Sequential(
            nn.Linear(self.model.head.in_features, num_classes),  # Adjust final layer
            nn.Sigmoid()  # Multi-label classification
        )

    def forward(self, x):
        return self.model(x)
