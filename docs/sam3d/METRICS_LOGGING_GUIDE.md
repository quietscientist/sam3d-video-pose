# Comprehensive Metrics Logging & Visualization

## 🎯 Overview

The pipeline now includes **rigorous metrics logging** with **automatic visualization plots** at every stage. This provides complete data science tracking of:
- Sample sizes through the pipeline
- Keypoint statistics (jitter, displacement)
- Quality filter performance
- Processing stage impact

---

## 🎛️ Quick Start

### Default (Metrics + Plots Enabled)
```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```

**Output**: Complete metrics + plots in `output/video_metrics/`

### Disable Plots (Faster, Metrics Only)
```bash
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --no-plots
```

**Output**: Metrics JSON/CSV only, no plots generated

### Disable All Metrics
```bash
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --no-metrics
```

**Output**: Processing only, no metrics tracking

---

## 📊 What Gets Tracked

### 1. Sample Size Funnel
Tracks frame count at each processing stage:
- **Initial frames**: Total frames to process
- **After SAM3 segmentation**: Frames with valid masks
- **After quality filter**: Frames passing quality thresholds
- **With valid 3D keypoints**: Successful MHR extraction
- **After temporal smoothing**: Smoothed keypoints
- **After bundle adjustment**: Final adjusted keypoints

### 2. Keypoint Statistics (Per Stage)
For each processing stage, tracks per-keypoint:
- **Mean position** (x, y, z)
- **Position std deviation**
- **Mean jitter** (frame-to-frame displacement)
- **Max jitter**
- **Overall jitter** across all keypoints

### 3. Quality Filter Performance
- Total frames analyzed
- Good vs. bad frame counts
- Quality metric distributions (z_velocity, total_velocity, vertex_displacement)
- Percentiles (25th, 50th, 75th, 95th) for threshold tuning

---

## 📁 Output Structure

```
output/
└── video_metrics/
    ├── metrics_summary.json              # Complete metrics JSON
    ├── sample_sizes.csv                  # Sample size tracking table
    └── plots/
        ├── sample_size_funnel.png        # Funnel chart showing attrition
        ├── quality_metrics_timeline.png  # Quality metrics over time
        ├── quality_distributions.png     # Histograms of quality metrics
        └── keypoint_jitter_comparison.png # Jitter before/after smoothing
```

---

## 📈 Generated Plots

### 1. Sample Size Funnel
**File**: `plots/sample_size_funnel.png`

Horizontal bar chart showing frame counts at each stage:
```
initial_frames                 ████████████████ 300
after_segmentation             ███████████████  295
after_quality_filter           ████████████     235
valid_3d_keypoints             ████████████     235
after_temporal_smoothing       ████████████     235
after_bundle_adjustment        ████████████     235
```

**Purpose**: See where frames are being filtered

---

### 2. Quality Metrics Timeline
**File**: `plots/quality_metrics_timeline.png`

Three subplots showing quality metrics over time:
- **Z-velocity** (depth movement)
- **Total velocity** (overall camera movement)
- **Vertex displacement** (mesh deformation)

Each plot shows:
- Metric values per frame (line)
- Threshold (red dashed line)
- Bad frames (red dots)

**Purpose**: Visualize when artifacts occur in the video

---

### 3. Quality Distributions
**File**: `plots/quality_distributions.png`

Histograms for each quality metric:
- Distribution of z_velocity across all frames
- Distribution of total_velocity
- Distribution of vertex_displacement
- Thresholds marked as red dashed lines

**Purpose**: Understand metric distributions for threshold tuning

---

### 4. Keypoint Jitter Comparison
**File**: `plots/keypoint_jitter_comparison.png`

Grouped bar chart comparing jitter across stages:
- Raw keypoints (before smoothing)
- Smoothed keypoints
- Bundle-adjusted keypoints

**Purpose**: Quantify smoothing and bundle adjustment impact

---

## 📝 Metrics JSON Format

### `metrics_summary.json`

```json
{
  "video_name": "baby_video",
  "timestamp": "2025-01-15T14:30:22.123456",

  "sample_sizes": {
    "initial_frames": {
      "count": 300,
      "description": "Total frames to process"
    },
    "after_quality_filter": {
      "count": 235,
      "description": "Frames passing quality checks"
    }
  },

  "keypoint_stats": {
    "raw": {
      "per_keypoint": {
        "nose": {
          "mean_position": [0.5, 0.3, 1.2],
          "std_position": [0.02, 0.03, 0.01],
          "mean_jitter": 2.345,
          "max_jitter": 8.912
        }
      },
      "overall": {
        "mean_jitter_all_keypoints": 2.567,
        "max_jitter_all_keypoints": 8.912,
        "num_keypoints": 17,
        "num_frames": 235
      }
    },
    "smoothed": { ... },
    "bundle_adjusted": { ... }
  },

  "quality_stats": {
    "total_frames": 300,
    "good_frames": 235,
    "bad_frames": 65,
    "good_frame_percentage": 78.3,
    "z_velocity": {
      "mean": 0.089,
      "std": 0.045,
      "min": 0.001,
      "max": 0.456,
      "percentiles": {
        "25": 0.012,
        "50": 0.034,
        "75": 0.089,
        "95": 0.234
      }
    }
  }
}
```

---

## 🔧 Integration in Your Code

The `MetricsLogger` is ready to use programmatically:

```python
from sam3dvideo.processing import MetricsLogger

# Initialize
metrics = MetricsLogger(
    output_dir="output",
    video_name="my_video",
    enable_plots=True
)

# Log sample sizes
metrics.log_sample_size('initial_frames', 300, "Total frames to process")
metrics.log_sample_size('after_quality_filter', 235, "Passed quality checks")

# Log keypoint statistics
keypoints_list = [...]  # List of (17, 3) numpy arrays
metrics.log_keypoint_statistics(
    stage='raw',
    keypoints_3d_list=keypoints_list,
    keypoint_names=['nose', 'left_eye', ...]  # Optional
)

# Log quality statistics
quality_log = [...]  # List of quality info dicts
metrics.log_quality_statistics(quality_log)

# Generate plots
metrics.plot_sample_size_funnel()
metrics.plot_quality_metrics(quality_log)
metrics.plot_quality_distributions(quality_log)
metrics.plot_keypoint_jitter_comparison()

# Save final report
metrics.save_metrics_json()
metrics.save_metrics_csv()
metrics.generate_final_report()
```

---

## 🎓 Example Workflow

### Processing with Metrics

```bash
python process_video_refactored.py baby_video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv \
  --enable-quality-filter
```

**Console Output**:
```
[Metrics] initial_frames: 300 frames (Total frames to process)
[Metrics] after_segmentation: 295 frames (Valid SAM3 masks)
[Metrics] after_quality_filter: 235 frames (Passed quality checks)
[Metrics] valid_3d_keypoints: 235 frames (Successful MHR extraction)
[Metrics] raw keypoint stats: mean_jitter=2.567 mm
[Metrics] Quality: 235/300 good frames (78.3%)
[Metrics] smoothed keypoint stats: mean_jitter=1.234 mm
[Metrics] bundle_adjusted keypoint stats: mean_jitter=1.189 mm
[Metrics] Saved funnel plot: output/baby_video_metrics/plots/sample_size_funnel.png
[Metrics] Saved quality timeline plot: ...
[Metrics] Saved quality distributions plot: ...
[Metrics] Saved jitter comparison plot: ...

============================================================
METRICS SUMMARY
============================================================

Sample Sizes:
  initial_frames                : 300 frames
  after_segmentation            : 295 frames
  after_quality_filter          : 235 frames
  valid_3d_keypoints            : 235 frames
  after_temporal_smoothing      : 235 frames
  after_bundle_adjustment       : 235 frames

Quality Statistics:
  Good frames: 235 (78.3%)
  Bad frames:  65

Keypoint Statistics:
  raw:
    Mean jitter: 2.5670 mm
    Max jitter:  8.9120 mm
  smoothed:
    Mean jitter: 1.2340 mm
    Max jitter:  4.5670 mm
  bundle_adjusted:
    Mean jitter: 1.1890 mm
    Max jitter:  4.3210 mm

Metrics saved to: output/baby_video_metrics
============================================================
```

---

## 📊 Analysis Examples

### 1. Analyze Quality Filter Performance

```python
import json
import numpy as np

# Load metrics
with open('output/video_metrics/metrics_summary.json', 'r') as f:
    metrics = json.load(f)

# Check if thresholds are optimal
z_vel_stats = metrics['quality_stats']['z_velocity']
print(f"Z-velocity 95th percentile: {z_vel_stats['percentiles']['95']}")
print(f"Current threshold: 0.15")
print(f"Recommendation: {'Lower threshold' if z_vel_stats['percentiles']['95'] < 0.15 else 'Current is good'}")
```

### 2. Quantify Smoothing Impact

```python
# Load metrics
with open('output/video_metrics/metrics_summary.json', 'r') as f:
    metrics = json.load(f)

raw_jitter = metrics['keypoint_stats']['raw']['overall']['mean_jitter_all_keypoints']
smoothed_jitter = metrics['keypoint_stats']['smoothed']['overall']['mean_jitter_all_keypoints']

improvement = (raw_jitter - smoothed_jitter) / raw_jitter * 100
print(f"Temporal smoothing reduced jitter by {improvement:.1f}%")
```

### 3. Sample Size Attrition Analysis

```python
import pandas as pd

# Load sample sizes
df = pd.read_csv('output/video_metrics/sample_sizes.csv')
print(df)

# Calculate attrition
initial = df[df['Stage'] == 'initial_frames']['Frame Count'].values[0]
final = df[df['Stage'] == 'after_bundle_adjustment']['Frame Count'].values[0]

attrition_rate = (initial - final) / initial * 100
print(f"Overall attrition: {attrition_rate:.1f}% ({initial - final} frames filtered)")
```

---

## 🎯 Best Practices

### 1. Always Enable Metrics for Production
```bash
# DON'T disable metrics unless absolutely necessary
python process_video_refactored.py video.mp4 --export-coco-csv
```

### 2. Disable Plots for Batch Processing
```bash
# For many videos, disable plots to save time
for video in *.mp4; do
  python process_video_refactored.py "$video" \
    --export-coco-csv \
    --no-plots
done
```

### 3. Use Metrics for Quality Control
After processing, check:
- `sample_sizes.csv` - Are you losing too many frames?
- `quality_stats` - Are thresholds appropriate?
- `keypoint_stats` - Is smoothing effective?

### 4. Archive Metrics with Results
Always keep the `_metrics/` folder with your processed data:
```bash
tar -czf baby_video_processed.tar.gz \
  baby_video_all_keypoints.json \
  baby_video_3D_smoothed_adjusted.csv \
  baby_video_metrics/
```

---

## 🔍 Troubleshooting

### Plots Not Generating

**Problem**: Metrics JSON created but no plots
**Solution**: Check `enable_plots` parameter or `--no-plots` flag

### High Jitter Even After Smoothing

**Problem**: `mean_jitter` still high after temporal smoothing
**Solution**:
- Increase `--temporal-smooth-window` (e.g., 15)
- Check if tracking quality is poor (inspect quality metrics)

### Too Many Frames Filtered

**Problem**: Sample size drops significantly after quality filter
**Solution**:
- Check `quality_distributions.png` to see if thresholds are too strict
- Adjust quality thresholds using `--quality-*` flags
- Inspect `quality_metrics_timeline.png` to see when artifacts occur

---

## 📚 Related Documentation

- [QUALITY_FILTER_TUNING.md](QUALITY_FILTER_TUNING.md) - Quality threshold optimization
- [TEMPORAL_SMOOTHING.md](TEMPORAL_SMOOTHING.md) - SG filter parameters
- [BUNDLE_ADJUSTMENT_INTEGRATION.md](BUNDLE_ADJUSTMENT_INTEGRATION.md) - Bundle adjustment details

---

## 🎉 Summary

✅ **Comprehensive metrics tracking** at every stage
✅ **Automatic visualization plots** for data analysis
✅ **Sample size funnel** shows pipeline attrition
✅ **Quality performance analysis** with distributions
✅ **Keypoint jitter quantification** before/after smoothing
✅ **JSON + CSV exports** for custom analysis
✅ **Command-line control** (enable/disable metrics/plots)

**All metrics are logged automatically - no code changes needed!**
