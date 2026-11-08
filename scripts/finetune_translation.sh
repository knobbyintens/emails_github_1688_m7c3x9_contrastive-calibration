#!/usr/bin/env bash
set -euo pipefail
python -m contrastive.finetune --pretrained checkpoints/contrastive.pt --lang-pair "$2"
