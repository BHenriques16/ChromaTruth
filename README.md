# Chroma Truth: Deepfake Detection via Chromatic Reconstruction

## Overview
**Chroma Truth** is a forensic research project designed to detect facial manipulations (Deepfakes) by analyzing the physical coherence between luminance and color. Based on the hypothesis that generative models struggle to maintain the natural relationship between structure ($L$) and chrominance ($ab$), this project implements a **self-supervised U-Net** detective.

The project evaluates the viability of this forensic technique in a **2026 scenario**, where generative AI has reached "superhuman" levels of chromatic perfection.



## Project Structure
```text
ChromaTruth/
├── data/
│   ├── 01_Real/             # Authentic images for training and validation
│   └── 03_Test_Pairs/       # Paired Real/Fake images for forensic testing
├── results/
│   ├── checkpoints/         # Saved model weights (.pth)
│   ├── detections/          # Generated heatmaps and forensic reports
│   └── learning_curve.png   # Training progress visualization
├── source/
│   ├── model.py             # U-Net Architecture
│   ├── utils.py             # Lab color space conversion and normalization
│   ├── train.py             # Self-supervised training script with Early Stopping
│   └── detect.py            # Forensic analysis, Heatmap generation, and ROC/AUC
└── report.pdf               # IEEE-style scientific paper
