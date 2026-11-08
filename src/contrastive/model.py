"""Dual-encoder speech/text model with a contrastive objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveSpeechEncoder(nn.Module):
    """Shared contrastive embedding space for speech and text."""

    def __init__(self, speech_dim=768, text_dim=768, embed_dim=256, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.speech_proj = nn.Sequential(
            nn.Linear(speech_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, speech_features, text_features):
        speech_emb = F.normalize(self.speech_proj(speech_features), dim=-1)
        text_emb = F.normalize(self.text_proj(text_features), dim=-1)
        logits = torch.matmul(speech_emb, text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        speech_loss = F.cross_entropy(logits, labels)
        text_loss = F.cross_entropy(logits.t(), labels)
        return 0.5 * (speech_loss + text_loss)


class TranslationDecoder(nn.Module):
    """Autoregressive decoder initialised from the pretrained text projection."""

    def __init__(self, embed_dim=256, vocab_size=32000, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_states, target_ids):
        hidden = self.embedding(target_ids)
        hidden = self.encoder(hidden)
        return self.lm_head(hidden)
