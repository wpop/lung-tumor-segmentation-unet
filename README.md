# Lung Tumor Segmentation U-Net

## Overview

Lung Tumor Segmentation U-Net is a research prototype for lung tumor segmentation from CT volumes. It is intended as an AI-assisted segmentation demo for exploring medical imaging workflows, model training, and volumetric inference.

This project is not a diagnostic system and should not be used for clinical decision-making.

## Dataset

This project uses the Medical Segmentation Decathlon Task06 Lung dataset.

- NIfTI `.nii.gz` volumes
- CT image volumes
- Binary tumor masks

## Pipeline

The project follows a practical 2D segmentation pipeline:

1. Load NIfTI CT volume
2. Preprocess HU values
3. Extract 2D slices
4. Train 2D U-Net
5. Validate with patient-level split
6. Run full-volume inference
7. Export results for viewer

## Model

- 2D U-Net
- BCE + Dice loss
- PyTorch implementation
- CUDA support when available

## Current Results

Patient-level validation with negative slices:

| Metric | Value |
| --- | ---: |
| Dice | 0.5166 |
| IoU | 0.4171 |
| Precision | 0.5769 |
| Recall | 0.5899 |
| Best threshold | 0.5 |

## How to Run

Run the dataset and dataloader checks:

```bash
python scripts/test_dataset.py
python scripts/test_dataloader.py
```

Train the model:

```bash
python src/training/train.py
```

Search prediction thresholds:

```bash
python scripts/search_threshold.py
```

Run full-volume inference and export viewer files:

```bash
python scripts/predict_full_volume.py
python scripts/export_for_viewer.py
```

## Output

Generated outputs are ignored by Git, including:

- `outputs/`
- `*.pt`
- `*.nii.gz`

These files can include trained checkpoints, predictions, exported volumes, and other generated artifacts.

## Future Work

- Data augmentation
- 3D U-Net
- FastAPI inference service
- React web viewer
- C++ Qt/OpenGL volume viewer
