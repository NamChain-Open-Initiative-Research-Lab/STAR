import torch
import torch.nn as nn
import timm  # Hugging Face Pretrained Model

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinTransformerModel, self).__init__()
        
        # Load Swin Transformer (pretrained)
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
