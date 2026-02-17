# Video Processing Usage Modes

The refactored `process_video_refactored.py` supports **three different processing modes** to give you flexibility in what you want to extract.

---

## 🎯 Mode 1: Full Pipeline (Default)

**Extract everything: 3D meshes, 3D keypoints, 2D keypoints**

```bash
python process_video_refactored.py VIDEO_PATH \
  --max-frames 300 \
  --output-dir output
```

**What you get:**
- ✅ SAM3 segmentation masks
- ✅ 3D meshes (PLY files per frame)
- ✅ MHR 3D keypoints (70 points)
- ✅ ViTPose 2D keypoints (17 COCO points)
- ✅ Quality analysis (frame filtering)
- ✅ Temporally smoothed meshes
- ✅ All JSON outputs

**Output files:**
```
output/
├── {video}_all_keypoints.json      # All keypoints (2D + 3D)
├── {video}_quality_log.json        # Frame quality metrics
├── {video}_segments.json           # Good frame segments
├── {video}_masks/                  # SAM3 masks
├── {video}_meshes/                 # Per-frame meshes
│   └── frame_XXXX_obj_Y/
│       ├── mesh.ply                # 3D mesh
│       ├── mhr_parameters.npz      # MHR params
│       └── keypoints.json          # 2D + 3D keypoints
└── {video}_smoothed/
    ├── meshes/                     # Smoothed meshes
    └── metadata.json
```

**Use when:** You need complete 3D reconstruction + motion analysis

---

## 🎯 Mode 2: MHR Params Only (No PLY Saving)

**Extract MHR parameters and keypoints but don't save mesh PLY files**

```bash
python process_video_refactored.py VIDEO_PATH \
  --max-frames 300 \
  --skip-mesh-saving
```

**What you get:**
- ✅ SAM3 segmentation masks
- ✅ MHR 3D keypoints (70 points) + all parameters
- ✅ ViTPose 2D keypoints (17 COCO points)
- ✅ Quality analysis (frame filtering)
- ❌ No mesh PLY files saved (saves disk space!)
- ❌ No temporal smoothing (requires saved meshes)

**Output files:**
```
output/
├── {video}_all_keypoints.json      # All keypoints (2D + 3D)
├── {video}_quality_log.json        # Frame quality metrics
├── {video}_segments.json           # Good frame segments
├── {video}_masks/                  # SAM3 masks
└── {video}_meshes/
    └── frame_XXXX_obj_Y/
        ├── mhr_parameters.npz      # MHR params (vertices, pose, shape, etc.)
        └── keypoints.json          # 2D + 3D keypoints
```

**Use when:**
- You only need keypoints + MHR parameters for analysis
- Want to save disk space (PLY files can be large)
- Planning to do bundle adjustment on keypoints only

**Disk savings:** ~70-80% less space (no PLY mesh files)

---

## 🎯 Mode 3: 2D Keypoints Only (Fastest)

**Extract only 2D keypoints, skip all 3D mesh generation**

```bash
python process_video_refactored.py VIDEO_PATH \
  --max-frames 300 \
  --skip-mesh-generation
```

**What you get:**
- ✅ SAM3 segmentation masks
- ✅ ViTPose 2D keypoints (17 COCO points)
- ❌ No 3D meshes
- ❌ No MHR 3D keypoints (70 points)
- ❌ No quality analysis (requires MHR params)
- ❌ No temporal smoothing

**Output files:**
```
output/
├── {video}_all_keypoints.json      # Only 2D keypoints
├── {video}_quality_log.json        # Empty (quality requires 3D)
├── {video}_segments.json           # All frames marked as "good"
├── {video}_masks/                  # SAM3 masks
└── {video}_meshes/
    └── frame_XXXX_obj_Y/
        └── keypoints.json          # Only 2D keypoints
```

**Use when:**
- You only need 2D pose estimation
- Want fastest possible processing
- Don't need 3D reconstruction
- Working with 2D-only analysis pipeline

**Speed:** ~5-10x faster than full pipeline (no mesh generation)

---

## 📊 Comparison Table

| Feature | Mode 1 (Full) | Mode 2 (No PLY) | Mode 3 (2D Only) |
|---------|---------------|-----------------|------------------|
| SAM3 Masks | ✅ | ✅ | ✅ |
| 2D Keypoints (ViTPose 17) | ✅ | ✅ | ✅ |
| 3D Keypoints (MHR 70) | ✅ | ✅ | ❌ |
| Mesh PLY Files | ✅ | ❌ | ❌ |
| MHR Parameters (NPZ) | ✅ | ✅ | ❌ |
| Quality Filtering | ✅ | ✅ | ❌ |
| Temporal Smoothing | ✅ | ❌ | ❌ |
| Processing Speed | Slowest | Medium | **Fastest** |
| Disk Usage | **Highest** | Medium | Lowest |

---

## 🔄 Workflow Examples

### Example 1: Full 3D + Bundle Adjustment Pipeline (NEW - Integrated)

```bash
# Single command: Extract everything + COCO CSV + bundle adjustment
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --export-coco-csv

# Outputs:
# - video_all_keypoints.json (MHR 70-point)
# - video_3D.csv (COCO 17-point, unadjusted)
# - video_3D_adjusted.csv (COCO 17-point, bundle-adjusted ✓)
# - video_meshes/ (PLY files)
# - video_smoothed/ (temporally smoothed meshes)

# Visualize bundle-adjusted keypoints
python visualize_3d_timeseries.py \
  output/video_3D_adjusted.csv \
  --animate --supine
```

---

### Example 2: Lightweight Keypoint Extraction (Recommended)

```bash
# Extract keypoints only (no PLY files) with integrated bundle adjustment
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv

# Outputs:
# - video_all_keypoints.json (MHR 70-point)
# - video_3D.csv (COCO 17-point, unadjusted)
# - video_3D_adjusted.csv (COCO 17-point, bundle-adjusted ✓)
# - video_meshes/ (MHR params only, no PLY files)

# Visualize
python visualize_3d_timeseries.py \
  output/video_3D_adjusted.csv \
  --animate --supine
```

**Benefits:**
- 70-80% less disk space
- Still get full 3D keypoint data (both MHR 70 and COCO 17)
- Bundle adjustment integrated (no separate script needed)
- Much faster storage/backup

---

### Example 3: Legacy Workflow (Two-Step Process)

If you prefer the old two-step workflow:

```bash
# Step 1: Extract keypoints
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving

# Step 2: Separate bundle adjustment (if you have convert_to_3d_csv.py)
python convert_to_3d_csv.py output/video_all_keypoints.json

# Step 3: Visualize
python visualize_3d_timeseries.py \
  output/video_3D_adjusted.csv \
  --animate --supine
```

---

### Example 3: 2D-Only Analysis

```bash
# Extract 2D keypoints only (fastest)
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-generation

# Your 2D keypoints are in:
# output/video_all_keypoints.json
```

Use with your own 2D pose analysis tools!

---

## 💡 Tips

1. **Start with Mode 2** (`--skip-mesh-saving`) for most workflows
   - Gets you all the data you need for keypoint analysis
   - Saves massive disk space
   - Can always regenerate PLY meshes later if needed

2. **Use Mode 3** (`--skip-mesh-generation`) for:
   - Quick 2D pose extraction
   - Testing/prototyping
   - When you only need 2D tracking

3. **Use Mode 1** (full) only when:
   - You need the actual 3D mesh geometry
   - You want temporal mesh smoothing
   - You're doing mesh-based analysis (not just keypoints)

---

## 🚀 Performance Impact

Based on 300 frame video:

| Mode | Processing Time | Disk Space | Memory Usage |
|------|----------------|------------|--------------|
| Full | ~15-20 min | ~2-3 GB | High |
| No PLY | ~12-15 min | ~500 MB | Medium |
| 2D Only | ~3-5 min | ~100 MB | Low |

*Actual times depend on GPU, video resolution, and frame rate*

---

## 🎓 Additional Options

All modes support these optional flags:

```bash
--text-prompt "a baby"              # Segmentation prompt (default: "a baby")
--enable-quality-filter             # Skip bad frames (rolling, being picked up)
--smoothing-sigma 2.0               # Temporal smoothing strength (default: 2.0)
--output-dir output                 # Output directory (default: "output")
--max-frames -1                     # Process all frames (-1 = all)
--export-coco-csv                   # Export COCO 17-point keypoints to CSV
--no-bundle-adjustment              # Skip bundle adjustment (when using --export-coco-csv)
```

### 🔧 COCO CSV Export & Bundle Adjustment

When you use `--export-coco-csv`, the pipeline will:

1. **Extract COCO keypoints**: Map MHR 70-point skeleton → COCO 17-point skeleton
2. **Apply bundle adjustment** (default): Enforce fixed bone lengths using constrained optimization
3. **Export CSV files**:
   - `{video}_3D.csv` - Unadjusted COCO keypoints
   - `{video}_3D_adjusted.csv` - Bundle-adjusted COCO keypoints (fixed bone lengths)

**Example**:
```bash
python process_video_refactored.py video.mp4 \
  --max-frames 300 \
  --skip-mesh-saving \
  --export-coco-csv
```

**Output**:
- `video_3D.csv` - Raw COCO keypoints from MHR
- `video_3D_adjusted.csv` - Optimized keypoints with fixed bone lengths (recommended)

**Skip bundle adjustment** (faster, but bone lengths vary):
```bash
python process_video_refactored.py video.mp4 \
  --export-coco-csv \
  --no-bundle-adjustment
```

---

## 📝 Notes

- **Bundle adjustment** is now integrated into the main pipeline with `--export-coco-csv`
  - Works with all modes that generate 3D keypoints (Mode 1 & 2)
  - Automatically extracts COCO 17-point subset from MHR 70-point skeleton
  - Enforces fixed bone lengths using SLSQP optimization
  - Exports both unadjusted and adjusted CSV files
- **Temporal smoothing** only works with Mode 1 (requires saved PLY meshes)
- **Quality filtering** only works with Mode 1 & 2 (requires MHR parameters)
- All modes produce `{video}_all_keypoints.json` (MHR 70-point format)
- Use `--export-coco-csv` to additionally generate COCO 17-point CSV files
