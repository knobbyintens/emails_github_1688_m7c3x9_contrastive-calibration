"""Long-tailed classifier backbone."""

import torch
import torch.nn as nn
import torchvision.models as models


class LongTailClassifier(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_dim = backbone.fc.in_features
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = torch.flatten(self.features(images), 1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits
