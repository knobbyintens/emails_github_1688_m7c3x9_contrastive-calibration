"""Long-tailed dataset helpers."""

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LongTailDataset(Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        self.root = Path(root) / split
        self.samples = sorted(self.root.glob("*.pt"))
        self.labels = [int(p.stem.split("_")[0]) for p in self.samples]
        self.class_counts = [Counter(self.labels)[i] for i in range(max(self.labels) + 1)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        payload = torch.load(self.samples[index])
        return payload["image"], int(payload["label"])
