"""Training script for contrastive calibration."""

import argparse

import torch
from torch.utils.data import DataLoader

from src.losses import ContrastiveCalibrationLoss
from src.model import LongTailClassifier
from src.datasets import LongTailDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wacv.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = LongTailDataset("data/train", split="train")
    loader = DataLoader(train_set, batch_size=128, shuffle=True)
    class_counts = torch.tensor(train_set.class_counts, dtype=torch.float32, device=device)
    model = LongTailClassifier(num_classes=len(class_counts)).to(device)
    criterion = ContrastiveCalibrationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(30):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, features = model(images, return_features=True)
            loss = criterion(features, logits, labels, class_counts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")
    torch.save(model.state_dict(), "outputs/contrastive_calibration.pt")


if __name__ == "__main__":
    main()
