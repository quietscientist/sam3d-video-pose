# SAM3D Video Pose

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

3D human pose estimation from video using [SAM3](https://github.com/facebookresearch/segment-anything-3) segmentation and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) reconstruction. Extracts full body meshes, 70-point MHR keypoints, and exports to COCO format with quality filtering and temporal smoothing.

## Features

- 🎥 **Video Processing**: Extract 3D body meshes and keypoints from any human video
- 🎯 **Text-Prompted Segmentation**: Use natural language to segment subjects (e.g., "a person", "a baby", "an athlete")
- 🌐 **URL Support**: Download and convert videos directly from URLs (including Wikimedia)
- 🦴 **3D Reconstruction**: Full body mesh and 70-point MHR keypoint extraction
- ⚡ **Multi-GPU Support**: Distribute models across multiple GPUs for memory efficiency
- 📊 **Quality Analysis**: Automatic detection of problematic frames (movement artifacts, camera motion)
- 🎯 **Temporal Smoothing**: Gaussian smoothing and interpolation for smooth motion sequences
- 📈 **Comprehensive Metrics**: Sample size tracking, quality statistics, keypoint jitter analysis
- 📄 **COCO Export**: Export 17-point COCO keypoints with bundle adjustment
- 📊 **3D Visualization**: Render skeleton time series with configurable bone connections

## Use Cases

- **Infant Motion Analysis**: Capture crawling, rolling, and reaching movements (use `--text-prompt "a baby"`)
- **Sports Performance**: Analyze athletic movements and technique
- **Rehabilitation**: Track patient movement patterns over time
- **Animation Reference**: Extract motion data for character animation
- **Research**: Study human movement across age groups and conditions

## Requirements

- Python 3.10
- CUDA-capable GPU (tested on 2x RTX 4090, 24GB VRAM each)
- Ubuntu/Linux (tested on Ubuntu 22.04)

## Installation

### Prerequisites
- Python 3.10
- CUDA-capable GPU with 16GB+ VRAM (or 2 GPUs for memory distribution)
- Git with submodule support

### Quick Install

1. **Clone the repository with submodules:**
```bash
git clone --recursive https://github.com/yourusername/sam3d-video-pose.git
cd sam3d-video-pose
```

2. **Create and activate virtual environment:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Or with `uv` (recommended):
```bash
uv venv --python 3.10
source .venv/bin/activate
```

3. **Install the package with all dependencies:**
```bash
# Basic installation (includes all core dependencies)
pip install -e .

# Or with uv (faster)
uv pip install -e .

# With optional dependencies
pip install -e ".[dev,notebooks]"  # Development tools and Jupyter support
pip install -e ".[viz]"            # Visualization tools
pip install -e ".[all]"            # Everything
```

4. **Set up HuggingFace token (for model downloads):**

Create a `.env` file in the project root:
```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

**Note:** All core dependencies (torch, transformers, opencv, moviepy, etc.) are included in `pyproject.toml` and installed automatically.

## Quick Start

### Process a Local Video

```bash
# Generic human subject
sam3dvideo path/to/video.mp4 --max-frames 100

# Infant analysis
sam3dvideo baby_video.mp4 --text-prompt "a baby" --max-frames 100

# Or using the script directly
python scripts/process_video.py path/to/video.mp4 --max-frames 100
```

### Download and Process from URL

```bash
# Download and process Wikimedia video
sam3dvideo "https://upload.wikimedia.org/wikipedia/commons/8/84/Infant_babbling_in_crib.ogv" \
    --text-prompt "a baby" --max-frames 100

# The tool will automatically:
# 1. Detect the URL
# 2. Download the video to data/
# 3. Convert to MP4 (using moviepy, no system ffmpeg needed!)
# 4. Process the video
```

### Process with Quality Filtering

```bash
# Enable quality filtering for videos with camera movement
sam3dvideo video.mp4 --enable-quality-filter --max-frames 300

# Adjust sensitivity thresholds
sam3dvideo video.mp4 --enable-quality-filter \
    --quality-z-velocity 0.10 \
    --quality-total-velocity 0.20
```

### Export COCO Keypoints

```bash
# Export to COCO 17-point format with bundle adjustment
sam3dvideo video.mp4 --export-coco-csv --max-frames 300
```

> **Note:** You can use either `sam3dvideo` (CLI command after installation) or `python scripts/process_video.py` (direct script). Both accept the same arguments.

## Usage Examples

### Basic Video Processing

```bash
# Process first 50 frames
sam3dvideo data/video.mp4 --max-frames 50

# Process all frames
sam3dvideo data/video.mp4 --max-frames -1

# Specify output directory
sam3dvideo data/video.mp4 --output-dir results/experiment1
```

### Advanced Options

```bash
# Full pipeline with quality filtering and COCO export
sam3dvideo data/video.mp4 \
    --max-frames 300 \
    --enable-quality-filter \
    --export-coco-csv \
    --smoothing-sigma 2.0 \
    --output-dir output/analysis

# Custom quality thresholds (lower = more sensitive)
sam3dvideo data/video.mp4 \
    --quality-z-velocity 0.10 \
    --quality-total-velocity 0.20 \
    --quality-vertex-displacement 0.05
```

### Infant-Specific Analysis

For infant motion capture, use these recommended settings:

```bash
sam3dvideo baby_crawling.mp4 \
    --text-prompt "a baby" \
    --enable-quality-filter \
    --export-coco-csv \
    --max-frames 300
```

**Why these settings:**
- `--text-prompt "a baby"`: Segments infant-sized humans
- `--enable-quality-filter`: Filters rolling, being picked up, camera movement
- Bundle adjustment excludes torso constraints (allows spinal flexibility)

### Download Only

If you just want to download and convert a video:

```bash
python -m sam3dvideo.utils.video_download "https://example.com/video.ogv" data
```

### Visualize 3D Keypoints

Visualize the 3D COCO keypoints with skeleton bones (excludes face keypoints by default):

```bash
# Install visualization dependencies first
pip install -e ".[viz]"

# Static multi-panel view (4 frames)
python scripts/visualize_3d_keypoints.py output/video_name/video_name_coco_keypoints_adjusted.csv

# Animated view
python scripts/visualize_3d_keypoints.py output/video_name/video_name_all_keypoints.json \
    --mode animation --fps 10

# Save output
python scripts/visualize_3d_keypoints.py output/video_name/video_name_coco_keypoints_adjusted.csv \
    --output visualization.png

# Include face keypoints (nose, eyes, ears)
python scripts/visualize_3d_keypoints.py output/video_name/video_name_coco_keypoints_adjusted.csv \
    --show-face
```

**Features:**
- 📊 Static multi-panel or animated views
- 🦴 COCO skeleton bones (limbs only, no flexible torso)
- 👤 Excludes face keypoints by default (nose, eyes, ears)
- 💾 Export to PNG (static) or GIF/MP4 (animation)

## Output Structure

```
output/
└── video_name/
    ├── video_name_masks/              # Segmentation masks
    ├── video_name_meshes/             # 3D meshes per frame
    │   ├── frame_0000_obj_0/
    │   │   ├── frame_0000_obj_0_mesh_000.ply
    │   │   ├── keypoints.json
    │   │   └── mhr_parameters.npz
    │   └── ...
    ├── video_name_quality_log.json    # Quality analysis
    ├── video_name_segments.json       # Continuous segments
    ├── video_name_all_keypoints.json  # All keypoints
    ├── video_name_coco_keypoints.csv  # COCO format (if --export-coco-csv)
    ├── video_name_coco_keypoints_adjusted.csv  # With bundle adjustment
    ├── video_name_smoothed/           # Temporally smoothed meshes
    └── video_name_metrics/            # Comprehensive metrics and plots
        ├── metrics.json
        ├── sample_sizes.csv
        └── *.png                       # Quality and jitter plots
```

## Project Structure

```
├── sam3dvideo/                 # Main package
│   ├── processing/             # Quality analysis, smoothing, metrics
│   ├── reconstruction/         # 3D mesh and keypoint extraction
│   ├── segmentation/           # SAM3 video segmentation
│   └── utils/                  # Utilities (config, video download, patches)
│
├── scripts/
│   ├── process_video.py        # Main processing script
│   └── visualize_3d_keypoints.py  # 3D visualization tool
│
├── external/                   # External dependencies
│   ├── sam-3d-body/            # SAM 3D Body (git submodule)
│   └── sam3/                   # SAM3 video model (git submodule)
│
├── data/                       # Downloaded/input videos
├── output/                     # Processing results
├── experiments/                # Experiment logs
└── configs/                    # Configuration files (YAML)
```

## GPU Memory Management

The pipeline uses two GPUs to manage memory:
- **GPU 0**: SAM3 segmentation (~15GB)
- **GPU 1**: SAM-3D-Body mesh estimation (~8GB)

For single GPU systems, reduce batch size or process fewer frames at once.

## Architecture

### Pipeline Stages

1. **Segmentation (SAM3)**: Segments subject in each frame using text prompts
2. **3D Reconstruction (SAM-3D-Body)**: Estimates 3D mesh and keypoints
3. **Quality Analysis**: Flags problematic frames (movement artifacts, camera motion)
4. **Temporal Smoothing**: Gaussian smoothing and interpolation
5. **Export**: Save meshes, keypoints, metrics, and visualizations

### Key Components

- **SAM3Segmenter**: Video segmentation with text prompts
- **MeshEstimator**: 3D body mesh estimation (70-point MHR skeleton)
- **KeypointExtractor**: 2D/3D keypoint extraction and COCO conversion
- **QualityAnalyzer**: Frame quality assessment
- **TemporalSmoother**: Temporal filtering and interpolation
- **BundleAdjuster**: Bone length constraints and temporal smoothing
- **MetricsLogger**: Comprehensive metrics and visualization

## Configuration

Create a YAML config file for batch processing:

```yaml
# config.yaml
experiment_name: "motion_analysis_experiment1"
input: "data/videos/"
output_dir: "output"
text_prompt: "a person"  # or "a baby", "an athlete", etc.

processing:
  max_frames: 300
  export_coco_csv: true
  bundle_adjustment: true
  temporal_smooth_keypoints: true

quality:
  enable_filter: true
  z_velocity_threshold: 0.15
  total_velocity_threshold: 0.25

metrics:
  enable_metrics: true
  enable_plots: true
```

Run with config:
```bash
sam3dvideo --config config.yaml
```

## Text Prompts

The pipeline uses SAM3's text-based segmentation. Use descriptive prompts for your subject:

- General: `"a person"`, `"a human"`
- Age-specific: `"a baby"`, `"a child"`, `"an adult"`
- Activity: `"a person walking"`, `"an athlete jumping"`
- Specific: `"a soccer player"`, `"a dancer"`

## Quality Filtering

The quality analyzer detects problematic frames:

- **Camera movement**: Rapid Z-axis or total velocity changes
- **Subject occlusion**: Partial visibility
- **Pose transitions**: Being picked up, rolling over
- **Mesh artifacts**: Unusual vertex displacements

Adjust sensitivity with threshold flags (lower = more sensitive):
- `--quality-z-velocity`: Camera depth movement (default: 0.15)
- `--quality-total-velocity`: Total camera movement (default: 0.25)
- `--quality-vertex-displacement`: Mesh deformation (default: 0.08)

## Bundle Adjustment

When exporting COCO keypoints, bundle adjustment:
- Enforces fixed bone lengths (realistic limb proportions)
- Applies temporal smoothing (Savitzky-Golay filter)
- **Excludes torso by default** (allows spinal flexibility - important for infants)

Include torso constraints with `--constrain-torso` (not recommended for infants).

## Troubleshooting

### CUDA Out of Memory
- Use `--skip-mesh-saving` to avoid saving PLY files
- Process fewer frames: `--max-frames 50`
- Ensure models are on different GPUs (check `sam3dvideo/reconstruction/mesh_estimator.py`)

### Module Import Errors
```bash
# Ensure package is installed
pip install -e .

# Verify submodules are initialized
git submodule update --init --recursive
```

### HuggingFace Authentication
```bash
# Set token in .env file
echo "HF_TOKEN=your_token" > .env

# Or use huggingface-cli
huggingface-cli login
```

## Citation

If you use this tool, please cite the underlying models:

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything in 3D Scenes},
  author={...},
  year={2024}
}

@article{sam3dbody,
  title={SAM-3D-Body: Single-Image 3D Human Body Estimation},
  author={...},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) file.

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/segment-anything-3) for video segmentation
- [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) for 3D body reconstruction
- [moviepy](https://github.com/Zulko/moviepy) for video processing

---

Built with ❤️ for human motion analysis research
