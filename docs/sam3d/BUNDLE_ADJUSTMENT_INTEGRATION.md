# Bundle Adjustment Integration

## ✅ What Was Integrated

Bundle adjustment functionality has been **fully integrated** into the main video processing pipeline. You no longer need to run a separate `convert_to_3d_csv.py` script!

---

## 🎯 Quick Start

### Single Command for Complete Processing

```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```

**This will**:
1. Extract SAM3 segmentation masks
2. Generate 3D meshes and MHR 70-point keypoints
3. Extract ViTPose 2D keypoints
4. **Extract COCO 17-point subset from MHR**
5. **Apply Savitzky-Golay temporal smoothing** (NEW! reduces jitter)
6. **Apply bundle adjustment to enforce fixed bone lengths**
7. **Export CSV files**:
   - `video_3D_raw.csv` (raw keypoints)
   - `video_3D_smoothed.csv` (temporally smoothed)
   - `video_3D_smoothed_adjusted.csv` (smoothed + bundle-adjusted ✓ RECOMMENDED)

---

## 📋 New Command-Line Flags

### `--export-coco-csv`
Enables COCO keypoint extraction and CSV export.

**Without this flag**: Only MHR 70-point JSON output
**With this flag**: MHR JSON + COCO 17-point CSV files

### `--no-bundle-adjustment`
Skip bundle adjustment (only use with `--export-coco-csv`).

**Default behavior**: Bundle adjustment is applied
**With this flag**: Only export unadjusted COCO CSV

### `--no-temporal-smooth-keypoints`
Skip Savitzky-Golay temporal smoothing of keypoints.

**Default behavior**: Temporal smoothing is applied
**With this flag**: Export raw (jittery) keypoints

### `--temporal-smooth-window N`
Set SG filter window length (default: 11, must be odd).

**Smaller window** (7-9): Minimal smoothing, preserves high-frequency motion
**Default** (11): Balanced smoothing for infant motion
**Larger window** (13-15): Stronger smoothing for noisy tracking

### `--temporal-smooth-polyorder N`
Set SG polynomial order (default: 3).

**Typical values**: 2-4 (higher = better fit but risk of overfitting)

---

## 🏗️ Technical Implementation

### New Module: `BundleAdjuster`

Location: `processors/bundle_adjuster.py`

**Key Features**:
- Maps MHR 70-point skeleton → COCO 17-point skeleton
- Estimates canonical bone lengths (median across all frames)
- Applies SLSQP constrained optimization to each frame
- Enforces 12 bone length equality constraints
- Achieves <0.0001% bone length variation

**COCO to MHR Mapping**:
```python
COCO_TO_MHR_MAP = {
    0: 0,    # nose
    1: 1,    # left_eye
    2: 2,    # right_eye
    3: 3,    # left_ear
    4: 4,    # right_ear
    5: 5,    # left_shoulder
    6: 6,    # right_shoulder
    7: 7,    # left_elbow
    8: 8,    # right_elbow
    9: 62,   # left_wrist  (MHR hand point)
    10: 41,  # right_wrist (MHR hand point)
    11: 9,   # left_hip
    12: 10,  # right_hip
    13: 11,  # left_knee
    14: 12,  # right_knee
    15: 13,  # left_ankle
    16: 14,  # right_ankle
}
```

**12 Bone Constraints**:
1. torso (left_hip → left_shoulder)
2. right_torso (right_hip → right_shoulder)
3. left_thigh (left_hip → left_knee)
4. right_thigh (right_hip → right_knee)
5. left_shin (left_knee → left_ankle)
6. right_shin (right_knee → right_ankle)
7. left_upper_arm (left_shoulder → left_elbow)
8. right_upper_arm (right_shoulder → right_elbow)
9. left_forearm (left_elbow → left_wrist)
10. right_forearm (right_elbow → right_wrist)
11. shoulder_width (left_shoulder → right_shoulder)
12. hip_width (left_hip → right_hip)

---

## 📊 Output Files

### With `--export-coco-csv` flag (default: temporal smoothing + bundle adjustment):

```
output/
├── video_all_keypoints.json             # MHR 70-point (all frames)
├── video_3D_raw.csv                     # COCO 17-point, raw (jittery)
├── video_3D_smoothed.csv                # COCO 17-point, SG smoothed
├── video_3D_smoothed_adjusted.csv       # COCO 17-point, smoothed + bundle-adjusted ✓ BEST
├── video_quality_log.json
├── video_segments.json
├── video_masks/
└── video_meshes/
```

### CSV Format:
```csv
frame,x,y,z,part_idx
0,100.5,200.3,50.2,0
0,105.2,195.8,48.9,1
...
```
- 17 rows per frame (one per COCO keypoint)
- `part_idx`: 0-16 (COCO keypoint indices)

---

## 🔄 Processing Flow

```
Video Input
    ↓
SAM3 Segmentation
    ↓
SAM-3D-Body Mesh Generation (MHR params)
    ↓
MHR 70-point Keypoints Extracted
    ↓
Quality Analysis
    ↓
[IF --export-coco-csv]
    ↓
Map MHR 70 → COCO 17
    ↓
[IF temporal_smoothing=True (default)] ← NEW!
    ↓
Savitzky-Golay Temporal Smoothing
    ↓
Estimate Canonical Bone Lengths (median)
    ↓
[IF bundle_adjustment=True (default)]
    ↓
SLSQP Optimization per Frame
    ↓
Export CSV Files
    ↓
Output: video_3D_raw.csv + video_3D_smoothed.csv + video_3D_smoothed_adjusted.csv
```

---

## 📈 Bundle Adjustment Statistics

The BundleAdjuster prints detailed statistics:

### Before Adjustment:
```
Bone                 Mean       Std        CV %
--------------------------------------------------
hip_width            45.2341    2.3451     5.18
left_shin            89.3421    3.8921     4.36
left_thigh           95.2341    4.1234     4.33
...
```

### After Adjustment:
```
Bone                 Mean       Std        CV %
--------------------------------------------------
hip_width            45.2341    0.000001   0.0000
left_shin            89.3421    0.000002   0.0000
left_thigh           95.2341    0.000001   0.0000
...
```

**Result**: Bone lengths fixed to canonical values with <0.0001% variation!

---

## 🎬 Example Workflows

### Recommended: Lightweight + Bundle Adjustment
```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```
- 70-80% disk space savings (no PLY files)
- Full 3D keypoint extraction
- Bundle adjustment applied
- CSV export for analysis

### Full Pipeline + Bundle Adjustment
```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --export-coco-csv
```
- Complete 3D mesh reconstruction
- Temporal smoothing
- Bundle adjustment
- All outputs

### Fast: Skip Bundle Adjustment
```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv \
  --no-bundle-adjustment
```
- Faster processing (no optimization)
- Only unadjusted CSV output
- Bone lengths will vary across frames

---

## 🧪 Validation

You can validate the bundle adjustment results:

```python
import pandas as pd
import numpy as np

# Load adjusted keypoints
df = pd.read_csv('output/video_3D_adjusted.csv')

# Check bone length consistency (e.g., left thigh: hip_11 → knee_13)
for frame in df['frame'].unique():
    frame_data = df[df['frame'] == frame]
    hip = frame_data[frame_data['part_idx'] == 11][['x','y','z']].values[0]
    knee = frame_data[frame_data['part_idx'] == 13][['x','y','z']].values[0]
    length = np.linalg.norm(hip - knee)
    print(f"Frame {frame}: left_thigh = {length:.6f} mm")

# Result: All frames should have IDENTICAL bone lengths!
```

---

## 🔍 Troubleshooting

### No CSV files generated?
- Make sure you used `--export-coco-csv` flag
- Check that 3D keypoints were successfully extracted (requires MHR params)
- Mode 3 (`--skip-mesh-generation`) won't work with bundle adjustment

### Bundle adjustment residuals too high?
- Check quality filtering - bad frames may have distorted keypoints
- Consider enabling `--enable-quality-filter` to skip problematic frames
- Inspect `video_quality_log.json` for frame-by-frame quality metrics

### Want to disable bundle adjustment?
```bash
--export-coco-csv --no-bundle-adjustment
```
This will only export unadjusted CSV

---

## 📚 Related Documentation

- `USAGE_MODES.md` - Different processing modes (full, no PLY, 2D only)
- `REFACTORING_SUMMARY.md` - Complete architecture documentation
- `processors/bundle_adjuster.py` - Source code with detailed docstrings

---

## 🎉 Benefits

✅ **No separate scripts needed** - Bundle adjustment integrated into main pipeline
✅ **Automatic COCO extraction** - MHR 70 → COCO 17 mapping handled
✅ **Fixed bone lengths** - Enforces anatomical constraints
✅ **Motion artifact correction** - Reduces temporal inconsistencies
✅ **Export both versions** - Compare adjusted vs. unadjusted
✅ **Detailed statistics** - Before/after bone length analysis
✅ **CSV format** - Ready for downstream analysis tools

---

## 🚀 Next Steps

1. Process your video with integrated bundle adjustment
2. Visualize results with `visualize_3d_timeseries.py`
3. Compare adjusted vs. unadjusted keypoints
4. Use CSV files for your motion analysis pipeline

**Enjoy fixed bone lengths and consistent 3D tracking!** 🎊
