"""Stage-1 contrastive pretraining for speech translation."""

import argparse

import torch
from torch.utils.data import DataLoader

from contrastive.data import PairedSpeechTextDataset
from contrastive.model import ContrastiveSpeechEncoder


def pretrain(args):
    dataset = PairedSpeechTextDataset(args.manifest, max_examples=args.max_examples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ContrastiveSpeechEncoder(
        speech_dim=args.speech_dim,
        text_dim=args.text_dim,
        embed_dim=args.embed_dim,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        for step, (speech, text) in enumerate(loader):
            loss = model(speech.cuda(), text.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"epoch={epoch} step={step} contrastive_loss={loss.item():.4f}")
    torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/contrastive.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--speech-dim", type=int, default=768)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--embed-dim", type=int, default=256)
    pretrain(parser.parse_args())
