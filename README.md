# Contrastive Pretraining for Low-Resource Speech Translation

**This repository contains the official implementation of the paper
"Contrastive Pretraining for Low-Resource Speech Translation" (ICASSP 2026).**

The release includes the contrastive pretraining code, the fine-tuning
pipeline and the evaluation scripts for the three low-resource language pairs.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/pretrain_contrastive.sh
bash scripts/finetune_translation.sh --lang-pair en-de
```

Evaluation:

```bash
python -m contrastive.evaluate --split dev --lang-pair en-de
```

## Citation

```bibtex
@inproceedings{cook2026contrastive,
  title={Contrastive Pretraining for Low-Resource Speech Translation},
  author={Cook, Ronald and Pereira, Sofia and Sato, Kenji},
  booktitle={Proceedings of ICASSP 2026},
  year={2026}
}
```
