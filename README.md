# An Interpretable Machine Learning Framework for Predicting the Popularity of YouTube News Shorts

This repository contains the code and curated experiment outputs for a study on publication-time popularity prediction for Taiwanese YouTube News Shorts.

## Overview

The project examines whether the relative popularity of YouTube news shorts is driven primarily by:

- semantic content
- framing cues
- contextual or channel-level conditions

Rather than predicting absolute engagement counts, the study formulates the task as a channel-normalized binary classification problem. The goal is to predict whether a short video will outperform its channel-specific baseline using only information available at publication time.

## Research Focus

The study is built around three theory-informed feature groups:

- `SEM`: semantic content
- `FRM`: framing cues
- `CTX`: contextual and channel signals

Multiple model classes are compared under a shared feature representation, with additional ablation, explainability, and robustness analyses.

## Main Findings

- Semantic information provides the strongest predictive signal.
- Framing cues add smaller but consistent improvements.
- Contextual and channel-level information is weaker in aggregate, although several channel-scale variables remain important at the individual-feature level.
- Once the shared representation is fixed, stronger nonlinear models provide only marginal gains over Logistic Regression.

## Repository Contents

- `results/paper/`: curated tables and figures that directly support the paper
- `experiments/`: experiment scripts used in the project
- `models/`: model implementations used by the experiments
- `feature_blocks.py`, `load_data.py`, `labels.py`, `classification.py`, `ranking.py`: core data-processing and evaluation utilities
- `pixi.toml`, `pixi.lock`: reproducible environment configuration

## Data Access

The local `data/` directory is not tracked in this repository because the metadata and OCR files are hosted separately.

- Google Drive data folder: https://drive.google.com/drive/folders/1W81s704NR78hcI0s52dV3Cbd4dgGxnI1?usp=sharing

After downloading the files, place them under `data/` in the repository root.

## Notes

- The LaTeX manuscript and bibliography are maintained separately and are not part of this GitHub package.
- Large local caches, exploratory artifacts, raw thumbnail assets, and unrelated experimental branches are excluded from version control.
- The curated paper outputs included in this repository are documented in results/paper/README.md.

## Environment

This project uses `pixi` for environment management.

Typical setup:

```bash
pixi install
pixi run python --version
```

You can then run the relevant experiment scripts from the repository root.
