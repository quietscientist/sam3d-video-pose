# SAM3D Integration with CAMT

## Overview

The SAM3D pipeline has been integrated into the CAMT (Cross-Age Motion Transfer) project structure.

## Directory Mapping

| Component | Location | Description |
|-----------|----------|-------------|
| **Segmentation** | `camt/segmentation/` | SAM3 video segmentation |
| **Reconstruction** | `camt/reconstruction/` | SAM 3D Body mesh & keypoints |
| **Processing** | `camt/processing/` | Quality analysis, smoothing, metrics |
| **Utilities** | `camt/utils/` | Shared utilities |
| **Scripts** | `scripts/` | Executable scripts |
| **Configs** | `configs/sam3d/` | Configuration files |
| **Docs** | `docs/sam3d/` | Documentation |
| **Notebooks** | `notebooks/sam3d/` | Example notebooks |

## Module Structure

### Segmentation Module
```python
from sam3dvideo.segmentation import SAM3Segmenter

segmenter = SAM3Segmenter()
```

### Reconstruction Module
```python
from sam3dvideo.reconstruction import MeshEstimator, KeypointExtractor

mesh_estimator = MeshEstimator()
keypoint_extractor = KeypointExtractor()
```

### Processing Module
```python
from sam3dvideo.processing import (
    QualityAnalyzer,
    TemporalSmoother,
    BundleAdjuster,
    MetricsLogger
)
```

## Usage

### Using Scripts

```bash
# Extract 3D poses
python scripts/extract_3d_pose.py --video input.mp4

# Full processing pipeline
python scripts/process_video.py input.mp4 --config configs/sam3d/default.yaml

# Batch processing
python scripts/batch_process.py videos_dir/
```

### Using as Python Module

```python
from sam3dvideo.segmentation import SAM3Segmenter
from sam3dvideo.reconstruction import MeshEstimator
from sam3dvideo.processing import QualityAnalyzer

# Your CAMT pipeline code here
segmenter = SAM3Segmenter()
mesh_estimator = MeshEstimator()
# ... integrate with your cross-age motion transfer pipeline
```

## External Dependencies

Install separately (see main README):

1. **PyTorch** with CUDA
2. **SAM3** (Segment Anything 2): `uv pip install sam2`
3. **SAM 3D Body**: Clone and install from GitHub
4. **MHR**: Clone and install from GitHub
5. **Detectron2**: Install from source

See main README for detailed installation instructions.

## Configuration

Configuration files are in `configs/sam3d/`:
- `default.yaml` - Complete configuration
- `quick_test.yaml` - Fast testing
- `batch_folder.yaml` - Batch processing

## Next Steps

1. Update main README with SAM3D integration info
2. Install external dependencies
3. Test integration with sample video
4. Integrate into your CAMT pipeline

## CAMT Pipeline Integration

To integrate SAM3D into your cross-age motion transfer pipeline:

```python
from sam3dvideo.segmentation import SAM3Segmenter
from sam3dvideo.reconstruction import MeshEstimator, KeypointExtractor
from sam3dvideo.processing import TemporalSmoother

# 1. Segment infant in video
segmenter = SAM3Segmenter()
for frame_idx, outputs, frame in segmenter.segment_video_chunks(video_path, "a baby"):
    # 2. Reconstruct 3D mesh
    mesh_estimator = MeshEstimator()
    mesh_outputs = mesh_estimator.estimate_mesh(frame, outputs['masks'][0])

    # 3. Extract keypoints
    keypoint_extractor = KeypointExtractor()
    keypoints_3d = keypoint_extractor.extract_mhr_parameters(mesh_outputs)

    # 4. Your CAMT cross-age transfer here
    # Transfer infant motion to adult body representation
    # ...
```
