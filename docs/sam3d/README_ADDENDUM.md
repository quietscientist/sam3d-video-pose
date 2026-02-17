# README Addendum: SAM3D Integration

Add this section to your main README.md:

---

## SAM3D Pipeline

This project includes SAM3D pipeline components for infant video segmentation and 3D reconstruction.

### Components

- **SAM3 Segmentation** (`camt/segmentation/`) - Video segmentation for tracking infants
- **SAM 3D Body** (`camt/reconstruction/`) - 3D mesh and keypoint extraction
- **Processing Pipeline** (`camt/processing/`) - Quality analysis and temporal smoothing

### Quick Start

```bash
# Process a video
python scripts/process_video.py input.mp4 --config configs/sam3d/default.yaml

# Extract 3D poses
python scripts/extract_3d_pose.py --video input.mp4
```

### Documentation

See `docs/sam3d/` for complete documentation:
- [CAMT_INTEGRATION.md](docs/sam3d/CAMT_INTEGRATION.md) - Integration guide
- [USAGE_MODES.md](docs/sam3d/USAGE_MODES.md) - Usage modes
- [QUALITY_FILTER_TUNING.md](docs/sam3d/QUALITY_FILTER_TUNING.md) - Quality tuning

### External Dependencies

SAM3D requires these external dependencies:

```bash
# SAM3 (video segmentation)
uv pip install sam2

# SAM 3D Body (3D reconstruction)
git clone https://github.com/facebookresearch/sam-3d-body.git external/sam-3d-body
cd external/sam-3d-body && uv pip install -e .

# MHR (body model)
git clone https://github.com/facebookresearch/MHR.git external/MHR
cd external/MHR && uv pip install -e .

# Detectron2
uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps
```

---
