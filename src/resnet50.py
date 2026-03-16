import torch.nn as nn
from torchvision import models
 
 
def build_resnet50(num_classes: int = 22, dropout: float = 0.4) -> nn.Module:
    """
    ResNet-50 pretrained on ImageNet with a custom classification head.
    Upgraded from ResNet-34 to match the project outline.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
 
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
 