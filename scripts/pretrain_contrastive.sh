#!/usr/bin/env bash
set -euo pipefail
python -m contrastive.train --manifest data/pairs.tsv --checkpoint checkpoints/contrastive.pt
