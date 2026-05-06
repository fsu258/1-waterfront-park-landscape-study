# Methodology Overview

## Purpose

This repository operationalizes the computational part of a landscape-preference workflow that integrates image-based spatial structure and audio-based soundscape information. The code was organized to support transparent reporting, reproducible processing, and eventual repository publication.

## Analytical Modules

### 1. Spatial landscape analysis from images

The image pipeline combines semantic segmentation and monocular metric depth estimation.

#### Semantic segmentation

The segmentation branch uses an ADE20K-trained MMSegmentation model to assign a semantic label to each pixel. The original ADE20K classes are aggregated into a smaller set of landscape-oriented categories:

- `trees`
- `shrubs`
- `grass`
- `bare_land`
- `terrain`
- `people`
- `vehicles`
- `buildings`
- `structures`
- `trash_bins`
- `signboards`
- `roads_paving`
- `sky`
- `water`
- `other`

#### Depth estimation and correction

The depth branch uses a monocular metric depth-estimation model. The predicted depth map is then corrected in two ways:

1. Non-sky pixels are rescaled so that the nearest valid depth matches a user-defined minimum distance, with a default value of `0.5 m`.
2. Pixels classified as `sky` are assigned a large depth value so they remain in the farthest spatial layer.

#### Spatial zoning

The corrected depth map is divided into three layers:

- foreground: `< 10 m`
- midground: `10-30 m`
- background: `>= 30 m`

These thresholds are configurable through command-line arguments.

#### Derived indicators

The spatial workflow exports:

- semantic composition proportions
- corrected depth statistics
- layer-specific category shares
- derived indicators including naturalness, disturbance, complexity, intentionality, enclosure, and openness
- layer-specific color features including brightness, contrast, saturation, color count, and color entropy

### 2. Temporal soundscape classification

The audio classification workflow uses YAMNet to classify sound content at approximately frame level and then aggregates predictions into one-second intervals.

The original YAMNet classes are mapped into five broad soundscape categories:

- `speech`
- `nature`
- `traffic_mechanical`
- `music`
- `silence`

For each second of each file, the pipeline exports:

- category probabilities
- dominant category
- optional visualization panels showing waveform, spectrogram, temporal composition, and average composition

### 3. Psychoacoustic metrics extraction

The psychoacoustic workflow computes calibrated acoustic descriptors from audio files using `moSQITo` and additional signal-processing steps.

The exported variables include:

- A-weighted equivalent sound level
- loudness
- sharpness
- roughness
- fluctuation strength
- psychoacoustic annoyance
- modulation depth
- dominant low-frequency modulation rate
- prominence ratio when available in the installed `moSQITo` version

## Suggested Workflow in the Research Pipeline

1. Process landscape images with `src/spatial_landscape_analysis.py`.
2. Process segmented audio files with `src/soundscape_classifier.py`.
3. Extract calibrated psychoacoustic metrics with `src/psychoacoustic_metrics.py`.
4. Merge exported tables in the downstream statistical workflow used for modeling and inference.

## Reproducibility Notes

- All scripts now use command-line arguments instead of hard-coded local paths.
- Output variable names are in English for easier downstream analysis and public sharing.
- Generated artifacts are separated from source code and intended to be written into `outputs/`.
- Model checkpoints, pretrained weights, and raw data should be documented separately when the repository is published.
