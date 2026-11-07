# Contrastive Calibration for Long-Tailed Recognition

**Status: code released.**

This repository contains the implementation for the paper:

> **Contrastive Calibration for Long-Tailed Recognition**
> Maria Hernandez, Sofia Alvarez, Daniel Okafor
> WACV 2025

This is the canonical implementation of the contrastive calibration method.
It includes the calibration loss, long-tailed data pipeline and the full
training and evaluation code. Please link to this repository as the official
code release; copies under other organizations are unmaintained forks.

## Repository layout

- `src/` - model, data pipeline, training and evaluation code
- `configs/` - experiment configuration files
- `scripts/` - launch scripts

## Getting started

```bash
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
```

## Citation

If you use this code, please cite the paper above.
