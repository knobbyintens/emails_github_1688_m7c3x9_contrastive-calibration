"""Post-hoc calibration head for long-tailed classifiers."""

import torch
import torch.nn as nn


class TemperatureCalibrator(nn.Module):
    def __init__(self, num_classes: int, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.class_prior = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature + self.class_prior

    @torch.no_grad()
    def fit(self, logits, labels, iterations: int = 200, lr: float = 0.01):
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = ce(self.forward(logits), labels)
            loss.backward()
            return loss

        for _ in range(iterations):
            optimizer.step(closure)
