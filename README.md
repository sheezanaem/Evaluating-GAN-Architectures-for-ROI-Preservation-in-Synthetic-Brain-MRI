# ROI-Preserving GAN Synthesis for Brain MRI — BraTS 2025

> Comparative study of four GAN architectures for synthesizing high-fidelity brain MRI
> while preserving clinically meaningful Regions of Interest (ROIs).

## Quick Start

```bash
pip install -r requirements.txt

# 1. Preprocess BraTS data
python -m preprocess.hd_bet_strip   --input data/raw --output data/stripped
python -m preprocess.register_sri24 --input data/stripped --output data/registered
python -m preprocess.normalize      --input data/registered --output data/normalized
python -m preprocess.clahe_enhance  --input data/normalized --output data/enhanced
python -m preprocess.slice_extractor   --input data/enhanced --output data/processed/slices
python -m preprocess.roi_mask_builder  --input data/raw --output data/processed/masks

# 2. Train a model
python training/run_experiment.py --model vanilla_gan --config training/config.yaml

# 3. Evaluate all models
python evaluation/evaluate_all.py --results_dir results/

# 4. Downstream U-Net validation
python downstream/train_segmentation.py --synthetic_dir results/best_model/
python downstream/evaluate_segmentation.py
```

## Project Structure

```
├── data/              Dataset loading utilities
├── preprocess/        HD-BET, SRI24, Z-score, CLAHE, slice extraction, ROI masks
├── models/            Vanilla GAN, cGAN, SRGAN, DnGAN + shared building blocks
├── training/          Config, base trainer, experiment runner
├── evaluation/        ROI-SSIM, MSE, HD95, DSC, KL safety
├── downstream/        U-Net segmentation validation
```

## Architectures

| Model | Key Feature |
|---|---|
| Vanilla GAN | Baseline adversarial framework |
| Conditional GAN | ROI-mask + modality-label conditioned |
| SRGAN | Super-resolution with perceptual loss |
| DnGAN | Denoising with gradient preservation |

## Evaluation Metrics (ROI-Centric)

- **Localized SSIM** (≥ 0.85 target)
- **MSE** (≤ 0.01 target)
- **HD95** — boundary alignment
- **DSC** — volumetric overlap
- **KL Divergence** — safety constraint
