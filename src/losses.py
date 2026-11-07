"""Contrastive calibration objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveCalibrationLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, calibration_weight: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.calibration_weight = calibration_weight

    def forward(self, features, logits, labels, class_counts):
        features = F.normalize(features, dim=-1)
        similarity = features @ features.T / self.temperature
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)
        positive = similarity[mask].mean()
        negative = similarity[~mask].mean()
        contrastive = -positive + negative

        prior = torch.log(class_counts / class_counts.sum())
        calibrated = logits + self.calibration_weight * prior.unsqueeze(0)
        ce = F.cross_entropy(calibrated, labels)
        return ce + 0.1 * contrastive
