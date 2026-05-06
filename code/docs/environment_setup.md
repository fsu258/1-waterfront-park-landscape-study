# Environment Setup

## Python Version

Recommended Python versions:

- `3.9` for audio and psychoacoustic workflows
- `3.8` or `3.9` for the MMSegmentation-based vision workflow, depending on the local CUDA and OpenMMLab stack

## Installation Strategy

Because the repository combines TensorFlow, PyTorch, MMSegmentation, and `moSQITo`, environment management is clearer if the workflows are installed either:

- in separate environments, or
- in one carefully managed environment after confirming version compatibility

The `requirements/` directory provides dependency lists for both approaches.

## Option A: Separate Environments

### Vision environment

```bash
python -m venv .venv-vision
source .venv-vision/bin/activate
pip install -r requirements/vision.txt
```

If CUDA is required, install a PyTorch build compatible with the local GPU and CUDA toolkit before installing the rest of the vision stack.

### Audio classification environment

```bash
python -m venv .venv-audio
source .venv-audio/bin/activate
pip install -r requirements/audio.txt
```

### Psychoacoustic environment

```bash
python -m venv .venv-psychoacoustics
source .venv-psychoacoustics/bin/activate
pip install -r requirements/psychoacoustics.txt
```

## Option B: Unified Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/all.txt
```

Use this option only after checking compatibility among:

- PyTorch
- MMCV
- MMSegmentation
- TensorFlow

## External Assets Required

The repository does not include large models or checkpoints. You must provide:

- a depth-estimation model path or compatible Hugging Face model identifier
- an MMSegmentation config file
- an MMSegmentation checkpoint file
- input image and audio data

## Quick Verification

### Soundscape classification

```bash
python src/soundscape_classifier.py sample_audio --output-csv outputs/test_soundscape.csv
```

### Psychoacoustic metrics

```bash
python src/psychoacoustic_metrics.py sample_audio --output outputs/test_psychoacoustics.csv
```

### Spatial landscape analysis

```bash
python src/spatial_landscape_analysis.py sample_images \
  --output-dir outputs/test_spatial \
  --depth-model-path /path/to/depth-model \
  --seg-config /path/to/config.py \
  --seg-checkpoint /path/to/checkpoint.pth
```

## Practical Notes

- `ffmpeg` may be needed by some audio backends when handling compressed audio formats.
- GPU acceleration is strongly recommended for the vision workflow.
- The YAMNet workflow downloads model resources at runtime.
- `moSQITo` should be checked carefully if the published paper requires exact standard compliance for every metric.
