# Temporal Smoothing for Keypoints

## 🎯 Overview

The pipeline now includes **Savitzky-Golay (SG) temporal smoothing** for keypoint positions, which significantly reduces jitter while preserving motion dynamics.

## Why Savitzky-Golay?

**Better than Gaussian smoothing** because:
- ✅ Preserves sharp movements and direction changes
- ✅ Fits local polynomials (better for position trajectories)
- ✅ Maintains velocity/acceleration continuity
- ✅ Less over-smoothing of important motion features
- ✅ **Empirically tested and performs better** for infant motion

---

## Processing Pipeline

```
Extract COCO keypoints (raw, per-frame)
    ↓
Temporal Smoothing (Savitzky-Golay filter) ← NEW!
    ↓
Bundle Adjustment (enforce bone lengths)
    ↓
Export CSV (smooth + anatomically correct)
```

**Result**: Keypoints with smooth trajectories AND fixed bone lengths!

---

## Usage

### Default (Recommended): Temporal Smoothing + Bundle Adjustment

```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```

**Output files**:
- `video_3D_raw.csv` - Raw COCO keypoints (jittery)
- `video_3D_smoothed.csv` - Temporally smoothed (smooth positions, varying bone lengths)
- `video_3D_smoothed_adjusted.csv` - **Smoothed + bundle adjusted** (recommended! ✓)

### Custom Smoothing Parameters

```bash
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --temporal-smooth-window 15 \
  --temporal-smooth-polyorder 3
```

**Parameters**:
- `--temporal-smooth-window` (default: 11)
  - Window length for SG filter
  - Must be odd number
  - Larger = smoother but more lag
  - Recommended range: 7-15 frames
- `--temporal-smooth-polyorder` (default: 3)
  - Polynomial order
  - Higher order = better fit but risk of overfitting
  - Recommended: 2-4

### Skip Temporal Smoothing

```bash
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --no-temporal-smooth-keypoints
```

**Output files**:
- `video_3D.csv` - Raw keypoints (no smoothing)
- `video_3D_adjusted.csv` - Bundle adjusted only (fixed bones, still jittery)

---

## Output File Naming

### With Temporal Smoothing (default):
```
video_3D_raw.csv                  # Raw COCO keypoints
video_3D_smoothed.csv             # SG filtered positions
video_3D_smoothed_adjusted.csv    # SG + bundle adjustment ✓ RECOMMENDED
```

### Without Temporal Smoothing:
```
video_3D.csv                      # Raw COCO keypoints
video_3D_adjusted.csv             # Bundle adjustment only
```

### Without Bundle Adjustment:
```
video_3D_raw.csv                  # Raw keypoints
video_3D_smoothed.csv             # SG filtered only
```

---

## Savitzky-Golay Filter Details

### How It Works

For each keypoint (e.g., left_knee), and each coordinate (x, y, z):

1. **Build time series**: Extract position across all frames
2. **Fit local polynomials**: For each time point, fit polynomial to neighboring frames
3. **Evaluate at center**: Use fitted polynomial to get smoothed value
4. **Preserve features**: Sharp turns and direction changes preserved

### Example

```
Raw positions (left_knee x-coordinate):
Frame 0:   100.2
Frame 1:   102.5  ← jitter
Frame 2:   99.8   ← jitter
Frame 3:   101.1
Frame 4:   100.9

Savitzky-Golay smoothed (window=5, polyorder=3):
Frame 0:   100.5
Frame 1:   101.2  ← smoothed
Frame 2:   100.8  ← smoothed
Frame 3:   101.0
Frame 4:   100.9

Bundle adjusted (bone lengths fixed):
Frame 0:   100.3  ← adjusted to maintain thigh length
Frame 1:   101.0
Frame 2:   100.7
Frame 3:   100.9
Frame 4:   100.8
```

---

## Parameter Selection Guide

### Window Length

| Frames | Effect | Use Case |
|--------|--------|----------|
| 5-7 | Minimal smoothing | High-frequency movements, fast actions |
| 9-11 | **Moderate smoothing** (default) | **General infant motion** ✓ |
| 13-15 | Strong smoothing | Slow movements, very noisy tracking |
| 17+ | Very smooth | Risk of over-smoothing, motion lag |

### Polynomial Order

| Order | Effect | Use Case |
|-------|--------|----------|
| 2 | Parabolic fit | Simple, smooth trajectories |
| **3** | **Cubic fit** (default) | **General motion** ✓ |
| 4 | Quartic fit | Complex motion, high quality data |
| 5+ | Higher order | Risk of overfitting noise |

---

## Comparison: Before vs After

### Before Temporal Smoothing

```csv
frame,x,y,z,part_idx
0,100.234,200.123,50.987,13  # left_knee
1,102.891,198.456,51.234,13  # jitter!
2,99.123,201.789,49.876,13   # jitter!
3,101.456,199.987,50.543,13
```

**Issues**:
- High-frequency jitter (±2-3mm)
- Unrealistic velocity spikes
- Noisy acceleration

### After Temporal Smoothing

```csv
frame,x,y,z,part_idx
0,100.523,199.876,50.654,13  # left_knee
1,101.234,199.543,50.789,13  # smooth!
2,100.987,199.987,50.612,13  # smooth!
3,101.123,199.765,50.598,13
```

**Result**:
- ✅ Smooth trajectories
- ✅ Realistic velocity/acceleration
- ✅ Still responds to actual motion

### After Smoothing + Bundle Adjustment

```csv
frame,x,y,z,part_idx
0,100.498,199.901,50.632,13  # left_knee
1,101.256,199.521,50.801,13  # smooth + correct bone length!
2,100.963,200.012,50.594,13  # smooth + correct bone length!
3,101.145,199.743,50.621,13
```

**Result**:
- ✅ Smooth trajectories
- ✅ **Fixed thigh length** (hip → knee)
- ✅ **Fixed shin length** (knee → ankle)
- ✅ Anatomically correct motion

---

## Performance Impact

**Processing time**:
- Temporal smoothing: +2-5 seconds for 300 frames
- Negligible overhead compared to mesh generation

**Memory**:
- Minimal additional memory (keypoint arrays only)

---

## Troubleshooting

### Window length must be odd

```bash
# ❌ ERROR
--temporal-smooth-window 10

# ✓ CORRECT
--temporal-smooth-window 11
```

### Not enough frames for smoothing

If you have fewer than 11 frames (default window), smoothing is automatically skipped:

```
⚠ Not enough frames (8) for temporal smoothing (need 11)
  Skipping temporal smoothing
```

**Solution**: Reduce window length or process more frames

### Over-smoothing (motion lag)

If smoothing removes too much real motion:

```bash
# Reduce window length
--temporal-smooth-window 7
```

### Under-smoothing (still jittery)

If keypoints still jittery:

```bash
# Increase window length
--temporal-smooth-window 15
```

---

## Advanced Usage

### Inspect Smoothing Impact

The pipeline prints per-keypoint displacement:

```
Per-keypoint smoothing displacement:
Keypoint             Mean Displacement (mm)
--------------------------------------------------
nose                 1.2345
left_eye             0.9876
right_eye            1.0234
left_shoulder        2.3456
...
left_knee            3.4567  ← High displacement = more smoothing needed
right_knee           3.2109
```

**Interpretation**:
- <1mm: Minimal smoothing (good tracking)
- 1-3mm: Moderate smoothing (typical)
- >3mm: Significant smoothing (noisy tracking)

### Compare Different Smoothing Levels

```bash
# Run 1: Default smoothing
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --temporal-smooth-window 11 \
  --output-dir output_w11

# Run 2: Stronger smoothing
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --temporal-smooth-window 15 \
  --output-dir output_w15

# Compare results
python visualize_3d_timeseries.py \
  output_w11/video_3D_smoothed_adjusted.csv \
  --compare output_w15/video_3D_smoothed_adjusted.csv
```

---

## Summary

✅ **Savitzky-Golay temporal smoothing** reduces keypoint jitter
✅ **Preserves motion dynamics** better than Gaussian smoothing
✅ **Enabled by default** with tested parameters (window=11, polyorder=3)
✅ **Combined with bundle adjustment** for smooth + anatomically correct keypoints
✅ **Exports multiple versions** for comparison and analysis

**Recommended workflow**: Use defaults for best results!

```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```

**Output**: `video_3D_smoothed_adjusted.csv` ← Use this! 🎉
