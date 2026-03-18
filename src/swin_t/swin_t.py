import torch.nn as nn
import timm


def build_swin_tiny(num_classes: int = 7, dropout: float = 0.4) -> nn.Module:
    model = timm.create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=True,
        num_classes=num_classes,
    )
    model.head.drop = nn.Dropout(dropout)
    return model