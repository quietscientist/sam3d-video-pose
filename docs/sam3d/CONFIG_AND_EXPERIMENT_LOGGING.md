# Configuration Files & Experiment Logging

## 🎯 Overview

The pipeline now supports **YAML configuration files** and **automatic experiment tracking**. This enables:
- ✅ Reproducible experiments with version-controlled configs
- ✅ Batch processing of video folders
- ✅ Automatic logging of all runs with parameters and results
- ✅ No need to remember complex command-line arguments

---

## 📋 Quick Start

### Using Config Files

```bash
# 1. Create or copy a config file
cp config_examples/default.yaml my_config.yaml

# 2. Edit the config (set your video path, parameters)
nano my_config.yaml

# 3. Run with config
python process_video_refactored.py --config my_config.yaml
```

### Command-Line Override

```bash
# Config sets defaults, CLI overrides specific parameters
python process_video_refactored.py --config my_config.yaml \
  --max-frames 100 \
  --experiment-name "quick_test"
```

### Traditional CLI (No Config)

```bash
# Still works exactly as before
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --export-coco-csv
```

---

## 📁 Config File Format

### Complete Example: `default.yaml`

```yaml
# Experiment name (optional, auto-generated if not provided)
experiment_name: null

# Input: single video file or folder of videos
input: "path/to/video.mp4"

# Text prompt for segmentation
text_prompt: "a baby"

# Output directory
output_dir: "output"

# Processing parameters
processing:
  max_frames: 300                      # -1 for all frames
  skip_mesh_generation: false          # Skip 3D mesh generation
  skip_mesh_saving: false              # Generate meshes but don't save PLY
  export_coco_csv: true                # Export COCO 17-point CSV
  bundle_adjustment: true              # Fix bone lengths
  constrain_torso: false               # Include torso constraints
  temporal_smooth_keypoints: true      # Apply SG smoothing
  temporal_smooth_window: 11           # SG window (must be odd)
  temporal_smooth_polyorder: 3         # SG polynomial order
  smoothing_sigma: 2.0                 # Mesh smoothing strength

# Quality filter parameters
quality:
  enable_filter: false                 # Enable quality filtering
  z_velocity_threshold: 0.15           # Max Z-axis movement
  total_velocity_threshold: 0.25       # Max total movement
  vertex_displacement_threshold: 0.08  # Max vertex deformation

# Metrics and logging
metrics:
  enable_metrics: true                 # Track comprehensive metrics
  enable_plots: true                   # Generate visualization plots
```

---

## 🎛️ Config Examples

### 1. Quick Test (50 frames)

**File**: `config_examples/quick_test.yaml`

```yaml
experiment_name: "quick_test"
input: "path/to/video.mp4"

processing:
  max_frames: 50
  skip_mesh_saving: true  # Faster
  export_coco_csv: true

metrics:
  enable_plots: true
```

**Usage**:
```bash
python process_video_refactored.py --config config_examples/quick_test.yaml
```

---

### 2. Batch Folder Processing

**File**: `config_examples/batch_folder.yaml`

```yaml
experiment_name: "batch_infant_videos"

# Process all videos in folder
input: "path/to/video/folder"

processing:
  max_frames: 300
  skip_mesh_saving: true
  export_coco_csv: true

quality:
  enable_filter: true  # Filter bad frames

metrics:
  enable_metrics: true
  enable_plots: false  # Disable plots for batch (faster)
```

**Usage**:
```bash
python process_video_refactored.py --config config_examples/batch_folder.yaml
```

**Result**: All `.mp4`, `.avi`, `.mov` files in the folder are processed automatically.

---

### 3. High-Quality Processing

```yaml
experiment_name: "high_quality_analysis"
input: "path/to/video.mp4"

processing:
  max_frames: -1  # All frames
  export_coco_csv: true
  bundle_adjustment: true
  temporal_smooth_window: 15  # Stronger smoothing

quality:
  enable_filter: true
  z_velocity_threshold: 0.10     # More sensitive
  total_velocity_threshold: 0.15
  vertex_displacement_threshold: 0.05

metrics:
  enable_metrics: true
  enable_plots: true
```

---

## 📊 Experiment Logging

### Automatic Tracking

Every run is automatically logged with:
- ✅ Unique run ID (timestamp-based)
- ✅ Full configuration (saved as `config.yaml`)
- ✅ Start/end time and duration
- ✅ Results (frames processed, good frames, etc.)
- ✅ Errors (if any)

### Experiment Directory Structure

```
experiments/
├── runs_log.json                     # Global log of all runs
├── my_experiment_20260213_143022/    # Run-specific directory
│   ├── config.yaml                   # Config used for this run
│   └── run_metadata.json             # Results and timing
└── batch_videos_20260213_150134/
    ├── config.yaml
    └── run_metadata.json
```

### Global Runs Log

**File**: `experiments/runs_log.json`

```json
{
  "runs": [
    {
      "run_id": "my_experiment_20260213_143022",
      "start_time": "2026-02-13T14:30:22.123456",
      "end_time": "2026-02-13T14:45:18.987654",
      "duration_seconds": 896.86,
      "status": "completed",
      "input": "videos/infant_crawling.mp4",
      "experiment_name": "my_experiment",
      "num_errors": 0,
      "run_dir": "experiments/my_experiment_20260213_143022"
    }
  ]
}
```

### Run Metadata

**File**: `experiments/{run_id}/run_metadata.json`

```json
{
  "run_id": "my_experiment_20260213_143022",
  "run_dir": "experiments/my_experiment_20260213_143022",
  "start_time": "2026-02-13T14:30:22.123456",
  "end_time": "2026-02-13T14:45:18.987654",
  "duration_seconds": 896.86,
  "status": "completed",
  "config": {
    "input": "videos/infant_crawling.mp4",
    "experiment_name": "my_experiment",
    "processing": {...},
    "quality": {...},
    "metrics": {...}
  },
  "results": {
    "infant_crawling_num_frames": 295,
    "infant_crawling_good_frames": 235,
    "infant_crawling_status": "completed"
  },
  "errors": []
}
```

---

## 🎓 Usage Patterns

### Pattern 1: Single Video with Config

```bash
# 1. Create config
cat > my_video.yaml <<EOF
experiment_name: "infant_01_baseline"
input: "videos/infant_01.mp4"
processing:
  max_frames: 300
  export_coco_csv: true
metrics:
  enable_plots: true
EOF

# 2. Run
python process_video_refactored.py --config my_video.yaml

# 3. Check results
cat experiments/infant_01_baseline_*/run_metadata.json
```

---

### Pattern 2: Batch Processing Folder

```bash
# 1. Create batch config
cat > batch_config.yaml <<EOF
experiment_name: "cohort_2024"
input: "videos/cohort_2024/"
processing:
  max_frames: 300
  skip_mesh_saving: true
  export_coco_csv: true
quality:
  enable_filter: true
metrics:
  enable_plots: false
EOF

# 2. Run (processes all videos in folder)
python process_video_refactored.py --config batch_config.yaml

# 3. Check experiment log
cat experiments/runs_log.json
```

---

### Pattern 3: Parameter Sweep

```bash
# Test different smoothing windows
for window in 7 9 11 13 15; do
  python process_video_refactored.py --config base_config.yaml \
    --temporal-smooth-window $window \
    --experiment-name "smooth_window_${window}"
done

# Compare results
ls -lh experiments/smooth_window_*/run_metadata.json
```

---

### Pattern 4: Override Single Parameter

```bash
# Config sets all defaults, override just max_frames
python process_video_refactored.py --config production.yaml \
  --max-frames 50 \
  --experiment-name "quick_test"
```

---

## 🔧 Priority Rules

When both config and command-line arguments are provided:

1. **Command-line ALWAYS takes precedence**
2. Config provides defaults for unspecified parameters
3. If neither provided, use built-in defaults

**Example**:

```bash
# Config says max_frames: 300
# CLI says --max-frames 100
# Result: 100 frames (CLI wins)

python process_video_refactored.py --config my_config.yaml --max-frames 100
```

---

## 📝 Creating Your Own Configs

### Step-by-Step

1. **Copy template**:
   ```bash
   cp config_examples/default.yaml my_project.yaml
   ```

2. **Edit required fields**:
   ```yaml
   input: "videos/my_video.mp4"  # REQUIRED
   experiment_name: "my_project"  # Optional
   ```

3. **Customize processing**:
   ```yaml
   processing:
     max_frames: 300
     export_coco_csv: true
     # ... other settings
   ```

4. **Run**:
   ```bash
   python process_video_refactored.py --config my_project.yaml
   ```

---

## 🎯 Best Practices

### 1. **Version Control Your Configs**

```bash
git add config_examples/production.yaml
git commit -m "Add production processing config"
```

### 2. **Use Experiment Names**

```yaml
# Good: descriptive experiment name
experiment_name: "infant_cohort_2024_baseline"

# Bad: no experiment name (auto-generated from video filename)
experiment_name: null
```

### 3. **Document Config Purpose**

```yaml
# Processing config for longitudinal infant study (cohort 2024)
# - Uses quality filter to remove picking up/rolling
# - Exports COCO CSV for downstream analysis
# - Disables plots for batch efficiency
experiment_name: "cohort_2024_longitudinal"
```

### 4. **Separate Configs by Use Case**

```
configs/
├── quick_test.yaml        # Fast testing
├── production.yaml        # Full processing
├── batch_processing.yaml  # Folder processing
└── high_quality.yaml      # Maximum quality
```

---

## 🔍 Troubleshooting

### Config File Not Found

```
Error: Config file not found: my_config.yaml
```

**Solution**: Check path is correct (relative to where you run the script)

---

### Input Path Not Found

```
Error: Input path not found: videos/test.mp4
```

**Solution**: Check `input:` path in config file

---

### Invalid YAML Syntax

```
Error loading config: ... yaml.scanner.ScannerError
```

**Solution**: Check YAML syntax (indentation, colons, quotes)

---

### Window Must Be Odd

```
ValueError: temporal_smooth_window must be odd (got: 10)
```

**Solution**: Use odd numbers (7, 9, 11, 13, 15)

---

## 🆚 Config vs. Command-Line

| Feature | Config File | Command-Line | Best For |
|---------|-------------|--------------|----------|
| Reproducibility | ✅ Excellent | ❌ Hard to track | Production |
| Quick changes | ❌ Edit file | ✅ Instant | Testing |
| Batch processing | ✅ Easy | ❌ Complex | Multiple videos |
| Version control | ✅ Yes | ❌ No | Research |
| Sharing | ✅ Simple | ❌ Long commands | Collaboration |

**Recommendation**: Use configs for everything except quick tests.

---

## 📚 Summary

✅ **YAML configs** for reproducible experiments
✅ **Automatic experiment logging** with timestamps and results
✅ **Batch folder processing** (all videos in folder)
✅ **Command-line override** for quick parameter changes
✅ **Version-controlled configs** for collaboration
✅ **Global runs log** to track all experiments
✅ **Backward compatible** - CLI-only usage still works

**Start using configs today**:

```bash
# Copy example
cp config_examples/default.yaml my_config.yaml

# Edit input path
nano my_config.yaml

# Run
python process_video_refactored.py --config my_config.yaml
```

**Your experiment will be automatically logged in `experiments/`!**
