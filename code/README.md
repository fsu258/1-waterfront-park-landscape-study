# Dynamic Landscape Preference Analysis Toolkit

This repository contains the method-side code for the study *Integrating Spatial Structure and Soundscape to Explain Dynamic Landscape Preference in Waterfront Parks: A DAG-informed Machine Learning Approach*.

The codebase has been reorganized into a clean, GitHub-ready structure with English filenames, English outputs, and command-line interfaces for the three core analytical workflows:

- `src/spatial_landscape_analysis.py`: semantic segmentation, monocular depth estimation, spatial zoning, and landscape feature extraction from images
- `src/soundscape_classifier.py`: one-second soundscape classification based on YAMNet
- `src/psychoacoustic_metrics.py`: psychoacoustic descriptor extraction based on `moSQITo`

## Repository Structure

```text
code/
├── README.md
├── .gitignore
├── docs/
│   ├── environment_setup.md
│   └── methodology_overview.md
├── requirements/
│   ├── audio.txt
│   ├── psychoacoustics.txt
│   ├── vision.txt
│   └── all.txt
└── src/
    ├── psychoacoustic_metrics.py
    ├── soundscape_classifier.py
    └── spatial_landscape_analysis.py
```

## Workflows

### 1. Spatial landscape analysis

This workflow extracts image-based variables related to spatial structure and prospect-refuge interpretation.

Main outputs:

- corrected depth maps in `.npy`
- segmentation masks in `.png`
- visualization panels in `.jpg`
- a summary table in `.csv`

Example:

```bash
python src/spatial_landscape_analysis.py data/images \
  --output-dir outputs/landscape_spatial_analysis \
  --depth-model-path /path/to/depth-anything-v2-metric \
  --seg-config /path/to/mask2former_config.py \
  --seg-checkpoint /path/to/mask2former_checkpoint.pth
```

### 2. Temporal soundscape classification

This workflow classifies each audio file at one-second resolution into five broad soundscape categories:

- `speech`
- `nature`
- `traffic_mechanical`
- `music`
- `silence`

Main outputs:

- one-second classification table in `.csv`
- temporal analysis figures in `.png`

Example:

```bash
python src/soundscape_classifier.py data/audio \
  --output-csv outputs/soundscape_per_second.csv \
  --visualization-dir outputs/soundscape_figures
```

### 3. Psychoacoustic metrics extraction

This workflow extracts calibrated psychoacoustic indicators for each audio file.

Main outputs:

- psychoacoustic metrics table in `.csv`

Example:

```bash
python src/psychoacoustic_metrics.py data/audio \
  --output outputs/psychoacoustic_metrics.csv \
  --field-type free \
  --include-tonality
```

## Key Output Variables

### Spatial analysis

The spatial workflow exports:

- semantic composition proportions by category
- corrected depth statistics
- foreground, midground, and background layer shares
- derived feature indices such as naturalness, disturbance, complexity, intentionality, enclosure, and openness
- layer-specific color metrics

### Soundscape classification

The soundscape workflow exports:

- file-level one-second records
- probability values for the five soundscape categories
- a dominant category per second

### Psychoacoustics

The psychoacoustic workflow exports:

- `laeq_db`
- `loudness_sone`
- `sharpness_acum`
- `roughness_asper`
- `fluctuation_vacil`
- `annoyance`
- `modulation_depth`
- `modulation_frequency_hz`
- `tonality_pr_db` when available

## Environment Setup

Dependency lists are provided in `requirements/`.

- `requirements/vision.txt`: image analysis stack
- `requirements/audio.txt`: YAMNet soundscape classification stack
- `requirements/psychoacoustics.txt`: psychoacoustic analysis stack
- `requirements/all.txt`: combined installation file

Detailed setup notes are available in `docs/environment_setup.md`.

## Method Documentation

The analytical rationale and variable definitions are documented in `docs/methodology_overview.md`.

## Notes for Publication and Reproducibility

- Model weights and large raw datasets are not included in this repository.
- Paths must be provided through CLI arguments instead of being edited directly in source files.
- Generated outputs are excluded by `.gitignore`.
- If the repository is intended for public release, consider adding a license file and a data availability statement.
